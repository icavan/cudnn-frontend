# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set TRTLLM_MEGAMOE_KERNELS to the pinned checkout's _torch/moe/kernels directory."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_nvfp4_training import (
    fixture,
    buffers,
    reference_pack,
    physical_scales,
    MoeEp,
    MoeEpConfig,
    MoeEpDataPathConfig,
    MoeEpModelConfig,
    MoeEpParallelConfig,
    MoeEpTrainingForwardOutputs,
    MoeEpNvfp4ForwardWeights,
    MoeFormat,
)


pytestmark = pytest.mark.L0


def assert_combined_output(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.005)
    target = expected.float()
    relative = (actual.float() - target).norm(dim=1) / target.norm(dim=1).clamp_min(1e-12)
    assert torch.quantile(relative, 0.99) < 0.03 and relative.max() < 0.05


def test_pinned_bf16_input_quantization():
    """Frozen cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false> outputs from the pinned CUDA source."""
    from cudnn.moe_ep._nvfp4_pack import PackTile

    with np.load(Path(__file__).with_name("data") / "trtllm_933366a_input.npz", allow_pickle=False) as frozen:
        rows = torch.from_numpy(frozen["x"]).cuda().view(torch.bfloat16)
        active, width = rows.shape
        x = torch.zeros((256, width), device=rows.device, dtype=rows.dtype)
        x[:active].copy_(rows)
        pack = PackTile(width, torch.bfloat16, False, False, forward=True)
        data = torch.empty((256, width // 2), device=rows.device, dtype=torch.uint8)
        sf = torch.empty((256, width // 16), device=rows.device, dtype=torch.float8_e4m3fn)
        for j, norm in enumerate(frozen["norms"]):
            pack.run(x, torch.tensor([norm], device=rows.device), data, sf)
            assert torch.equal(data[:active], torch.from_numpy(frozen[f"payload{j}"]).cuda())
            assert torch.equal(sf[:active].view(torch.uint8), torch.from_numpy(frozen[f"sf{j}"]).cuda())


@pytest.mark.parametrize("fused", [False, True], ids=["logical", "fused_checkpoint"])
def test_frozen_blackwell_forward(fused):
    """Replay the operands and workspace outputs of the pinned TRT-LLM device kernel."""
    device = torch.device("cuda", torch.cuda.current_device())
    with np.load(Path(__file__).with_name("data") / "trtllm_933366a_nvfp4.npz", allow_pickle=False) as frozen:
        tensors = {key: torch.from_numpy(frozen[key]).to(device) for key in frozen.files}
    from cudnn.moe_ep import MoeEpNvfp4FusedForwardWeights

    bank_type = MoeEpNvfp4FusedForwardWeights if fused else MoeEpNvfp4ForwardWeights
    fw = bank_type.from_trtllm(SimpleNamespace(**tensors), tensors["input_norm"], input_channel_scale=None)
    x = tensors["x"].view(torch.bfloat16)
    ids, p = tensors["ids"], tensors["p"]
    t, h = x.shape
    e, _, packed_i = tensors["fc2_weight"].shape
    i = 2 * packed_i
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=e, hidden_size=h, intermediate_size=i, top_k=ids.shape[1]),
            parallel=MoeEpParallelConfig(max_tokens_per_rank=t),
            data_path=MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0, nvfp4_fused_forward=fused),
        )
    )
    rec = buffers(op.prepare_training(device=device), MoeEpTrainingForwardOutputs, device)
    y = op.training_forward(x, ids, p, weights=fw, out=rec)
    assert_combined_output(y, tensors["y"].view(torch.bfloat16))
    for expert in range(e):
        rows = slice(expert * 256, expert * 256 + t)
        source_rows = slice(0, t) if fused else rows
        assert torch.equal(rec.fc1_a[source_rows], tensors["input_payload"])
        assert torch.equal(rec.fc1_sfa[source_rows].view(torch.uint8), tensors["input_sf"])
        assert torch.equal(rec.saved_a[rows], tensors["a_payload"][expert].expand(t, -1))
        assert torch.equal(rec.saved_sfa[rows].view(torch.uint8), tensors["a_sf"][expert].expand(t, -1))
    for slot in range(ids.shape[1]):
        isolated = torch.zeros_like(p)
        isolated[:, slot] = p[:, slot]
        route = op.training_forward(x, ids, isolated, weights=fw, out=rec)
        torch.testing.assert_close(route, tensors["route_y"].view(torch.bfloat16)[:, slot], rtol=0.01, atol=0.005)
    op.close()


def test_pinned_blackwell_forward(monkeypatch):
    root = os.environ.get("TRTLLM_MEGAMOE_KERNELS")
    if not root:
        pytest.skip("set TRTLLM_MEGAMOE_KERNELS for the optional pinned forward comparison")
    source = Path(root) / "cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "c186861316e70c9995a0f93c3cfe5dddf6fc4d31666553daf84afd9824ae773b"
    monkeypatch.syspath_prepend(root)
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from cutlass.cute.nvgpu import OperandMajorMode
    from cuda.bindings import driver as cuda
    from cutedsl_megamoe.api import ProblemDesc, ImplDesc
    from cutedsl_megamoe.quant_def import CombineFormat
    from cutedsl_megamoe.communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost
    from cutedsl_megamoe.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import BlockScaledSwapAbMegaMoeKernel

    e, h, i, t, k = 2, 256, 256, 16, 2
    device = torch.device("cuda", torch.cuda.current_device())
    fw, _, _, _ = fixture(e, h, i, device)

    def native(data, interleave):
        def order(v):
            return v.reshape(e, 2, -1, 16, v.shape[-1]).transpose(1, 2).reshape_as(v) if interleave else v

        return order(data.data), torch.stack([physical_scales(v).flatten() for v in order(data.scale.view(torch.uint8))])

    w1, sf1 = native(fw.fc1, True)
    w2, sf2 = native(fw.fc2, False)
    fw = MoeEpNvfp4ForwardWeights.from_trtllm(
        SimpleNamespace(
            fc1_weight=w1,
            fc1_weight_sf=sf1,
            fc2_weight=w2,
            fc2_weight_sf=sf2,
            fc31_alpha=fw.fc31_alpha,
            fc2_alpha=fw.fc2_alpha,
            fc1_norm_const=fw.fc1_norm_const,
        ),
        fw.input_norm,
        input_channel_scale=None,
    )
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0)
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=e, hidden_size=h, intermediate_size=i, top_k=k),
            parallel=MoeEpParallelConfig(max_tokens_per_rank=t),
            data_path=path,
        )
    )
    rec = buffers(op.prepare_training(device=device), MoeEpTrainingForwardOutputs, device)
    x = torch.linspace(-1.5, 1.5, h, device=device).to(torch.bfloat16).repeat(t, 1)
    ids = torch.arange(e, device=device, dtype=torch.int32).repeat(t, 1)
    p = torch.tensor([0.375, 0.75], device=device).repeat(t, 1)
    y = op.training_forward(x, ids, p, weights=fw, out=rec)
    xp, sfx, _ = reference_pack(x, fw.input_norm)
    if hasattr(torch.ops.trtllm, "fp4_quantize"):
        trt_x, trt_sf = torch.ops.trtllm.fp4_quantize(x, fw.input_norm, 16, False, False)
        assert torch.equal(xp, trt_x.view(torch.uint8))
        assert torch.equal(sfx.view(torch.uint8), trt_sf.view(torch.uint8).reshape_as(sfx))

    problem = ProblemDesc(
        dict(
            expert_count=e,
            intermediate_gateup_size=2 * i,
            hidden_size=h,
            quant_kind="nvfp4",
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
            combine_format=CombineFormat.parse("bf16"),
            gate_up_clamp=2.0,
            world_size=1,
            topk=k,
            topk_index_dtype=cutlass.Int64,
            max_tokens_per_rank=t,
            apply_topk_at_fc1=True,
        )
    )
    impl = ImplDesc(
        dict(
            mma_tiler_mnk=(256, 128, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            schedule_policy=("grouped", 512),
            token_padding_block=128,
            sf_padding_block=128,
            work_id_mode="grid_stride",
            fc2_use_bulk=True,
            epi_flag_batches=(1, 1),
            launch_cluster_count=8,
            token_in_flag_batch=1,
            token_back_mode="epi_warps",
            reduce_topk_in_kernel=False,
        )
    )
    kernel = BlockScaledSwapAbMegaMoeKernel(problem, impl)
    local, shared = [torch.zeros(n, device=device, dtype=torch.uint8) for n in kernel.get_workspace_sizes()]
    target = torch.empty_like(x)
    fp4 = torch.float4_e2m1fn_x2
    # Keep every allocation alive until the launch has completed (including converted indices).
    args = dict(
        activation=xp.view(fp4),
        activation_sf=sfx,
        fc1_weight=w1.view(fp4).transpose(1, 2),
        fc2_weight=w2.view(fp4).transpose(1, 2),
        fc1_weight_sf=sf1,
        fc2_weight_sf=sf2,
        fc1_alpha=fw.fc31_alpha,
        fc2_alpha=fw.fc2_alpha,
        fc1_norm_const=fw.fc1_norm_const,
        output_activation=target,
        topk_indices=ids.long(),
        topk_scores=p,
    )
    runtime = {
        n: ct.from_dlpack(v, assumed_align=4 if n in ("fc1_alpha", "fc2_alpha", "fc1_norm_const") else 16).mark_layout_dynamic(
            leading_dim=ct.get_leading_dim(v)
        )
        for n, v in args.items()
    }
    ptr = lambda v: cute.runtime.make_ptr(cutlass.Uint8, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)
    runtime.update(
        local_workspace=ptr(local),
        shared_workspace=ptr(shared),
        peer_rank_ptr_mapper_host=SymmetricBufferHost(base_address=xp.data_ptr(), offsets=(0,), rank=0, max_ranks=1),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel(**runtime)
    torch.cuda.synchronize()
    ws = kernel._device_workspace

    def region(heap, name, dtype):
        off = ws.offset(name)
        return heap[off : off + ws.nbytes(name)].view(dtype)

    partial = region(shared, kernel.token_comm.pre_reduced_activation_region, torch.bfloat16).reshape(t, k, h)
    output = region(local, kernel.fc1_output_region, torch.uint8).view(-1, i // 2)
    sfout = region(local, kernel.fc1_output_sf_region, torch.uint8)
    # The router pads each expert separately. Rows in this fixture are identical within an expert.
    for ex in range(e):
        row = ex * 256
        assert torch.equal(xp, rec.fc1_a[row : row + t])
        assert torch.equal(sfx.view(torch.uint8), rec.fc1_sfa[row : row + t].view(torch.uint8))
        assert torch.equal(output[ex * 128 : ex * 128 + t], rec.saved_a[row : row + t])
        blocked = sfout.view(-1, (i // 16) // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(-1, i // 16)
        assert torch.equal(blocked[ex * 128 : ex * 128 + t], rec.saved_sfa[row : row + t].view(torch.uint8))
        dense_a = unpack_ref(rec.saved_a[row : row + t], rec.saved_sfa[row : row + t])
        dense_w = unpack_ref(fw.fc2.data[ex], fw.fc2.scale[ex])
        torch.testing.assert_close(partial[:, ex], ((dense_a @ dense_w.T) * fw.fc2_alpha[ex]).bfloat16(), rtol=0.01, atol=0.005)
    assert_combined_output(y, target)
    op.close()


def unpack_ref(data, sf):
    from cudnn.moe_ep import BlockScaledTensor

    return BlockScaledTensor(data, sf, MoeFormat.NVFP4, (data.shape[0], data.shape[1] * 2)).dequantize()
