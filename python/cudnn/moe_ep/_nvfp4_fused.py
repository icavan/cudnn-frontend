# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared TRT-derived single-domain Blackwell fused training forward."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import cutlass
import cutlass.cute as cute
import cutlass.torch as ct
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode
from cuda.bindings import driver as cuda

from ._trt_blackwell.api import ProblemDesc, ImplDesc
from ._trt_blackwell.quant_def import CombineFormat
from ._trt_blackwell.communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost
from ._trt_blackwell.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import BlockScaledSwapAbMegaMoeKernel


def make_native_banks(experts, hidden, intermediate, device):
    if min(experts, hidden, intermediate) < 1 or hidden % 128 or intermediate % 128:
        raise ValueError("native weight banks require positive dimensions and H and I divisible by 128")
    return [
        (torch.empty((experts, n, k // 2), device=device, dtype=torch.uint8), torch.empty((experts, n, k // 16), device=device, dtype=torch.float8_e4m3fn))
        for n, k in ((2 * intermediate, hidden), (hidden, intermediate))
    ]


def pack_weights(weights, banks):
    for index, (weight, (dst, sf)) in enumerate(zip((weights.fc1, weights.fc2), banks)):
        e, n, b = weight.data.shape
        data = weight.data
        scales = weight.scale.view(torch.uint8)
        if index == 0:
            dst.view(e, -1, 2, 16, b).copy_(data.reshape(e, 2, -1, 16, b).transpose(1, 2))
            scales = scales.reshape(e, 2, -1, 16, scales.shape[-1]).transpose(1, 2).reshape_as(scales)
        else:
            dst.copy_(data)
        cols = scales.shape[-1]
        sf.view(torch.uint8).view(e, n // 128, cols // 4, 32, 4, 4).copy_(scales.reshape(e, n // 128, 4, 32, cols // 4, 4).permute(0, 1, 4, 3, 2, 5))


class FusedForward:
    def __init__(self, state):
        dev, self.group = state.device, state.group
        self.t, self.h, self.i, self.k = t, h, i, k = state.max_tokens, state.h, state.i, state.k
        topo = state.topology
        self.ep = topo.ep_size
        self.strict = state.resolved.public_config.validation_mode == "strict"
        local = topo.experts_per_rank
        if h % 128 or i % 128 or k > 16:
            raise ValueError("fused NVFP4 forward requires H and I divisible by 128 and top_k <= 16")
        problem = ProblemDesc(
            dict(
                expert_count=local * self.ep,
                intermediate_gateup_size=2 * i,
                hidden_size=h,
                quant_kind="nvfp4",
                a_major_mode=OperandMajorMode.K,
                b_major_mode=OperandMajorMode.K,
                combine_format=CombineFormat.parse("bf16"),
                gate_up_clamp=state.path.gate_up_clamp,
                world_size=self.ep,
                topk=k,
                topk_index_dtype=cutlass.Int64,
                max_tokens_per_rank=t,
                apply_topk_at_fc1=True,
            )
        )
        impl = ImplDesc(
            dict(
                save_training=True,
                mma_tiler_mnk=(256, 128 if t <= 1024 else 256, 256),
                cluster_shape_mn=(2, 1),
                use_2cta_instrs=True,
                schedule_policy=("grouped", 512),
                token_padding_block=64,
                sf_padding_block=128,
                work_id_mode="grid_stride",
                fc2_use_bulk=True,
                epi_flag_batches=(1, 1),
                launch_cluster_count=int(cutlass.utils.HardwareInfo().get_max_active_clusters(2)),
                token_in_flag_batch=1 if t <= 1024 else 4,
                token_back_mode="epi_warps",
                reduce_topk_in_kernel=False,
            )
        )
        self.kernel = BlockScaledSwapAbMegaMoeKernel(problem, impl)
        lbytes, sbytes = self.kernel.get_workspace_sizes()
        sf_pitch = self.kernel.token_comm.activation_sf_hidden_padded
        sizes = (t * h // 2, t * sf_pitch, t * k * 4, t * h * 2, sbytes)
        arena_size = sum((s + 255) // 256 * 256 for s in sizes)
        self.handle = None
        if self.ep > 1:
            self.arena = symm.empty(arena_size, dtype=torch.uint8, device=dev)
            self.handle = symm.rendezvous(self.arena, self.group)
            offsets = tuple(self.handle.get_buffer(r, (arena_size,), torch.uint8).data_ptr() - self.arena.data_ptr() for r in range(self.ep))
        else:
            self.arena = torch.empty(arena_size, dtype=torch.uint8, device=dev)
            offsets = (0,)
        self.arena.zero_()
        self.local = torch.zeros(lbytes, dtype=torch.uint8, device=dev)
        chunks = []
        offset = 0
        for nbytes in sizes:
            chunks.append(self.arena[offset : offset + nbytes])
            offset += (nbytes + 255) // 256 * 256
        data, scale, prob, output, shared = chunks
        self.data = data.view(t, h // 2)
        self.wire_sf = scale.view(torch.float8_e4m3fn).view(t, sf_pitch)
        self.scale = self.wire_sf if sf_pitch == h // 16 else torch.empty((t, h // 16), device=dev, dtype=torch.float8_e4m3fn)
        self.prob = prob.view(torch.float32).view(t, k)
        self.output = output.view(torch.bfloat16).view(t, h)
        self.indices = torch.empty((t, k), dtype=torch.int64, device=dev)
        self.x_pad = torch.empty((((t + 255) // 256) * 256, h), dtype=torch.bfloat16, device=dev)
        self.tail_data = torch.empty((256, h // 2), dtype=torch.uint8, device=dev)
        self.tail_sf = torch.empty((256, h // 16), dtype=torch.float8_e4m3fn, device=dev)
        self.inverse = torch.empty(t * self.ep * k + 1, device=dev, dtype=torch.int32)
        self.row_ids = torch.arange(state.rows, device=dev, dtype=torch.int32)
        self.norms = torch.empty(self.ep, device=dev, dtype=torch.float32)
        self.alpha = torch.empty((3, local), device=dev, dtype=torch.float32)
        self.native_weights = make_native_banks(local, h, i, dev)
        (w1, s1), (w2, s2) = self.native_weights
        fp4 = torch.float4_e2m1fn_x2
        args = dict(
            activation=self.data.view(fp4),
            activation_sf=self.wire_sf,
            fc1_weight=w1.view(fp4).transpose(1, 2),
            fc2_weight=w2.view(fp4).transpose(1, 2),
            fc1_weight_sf=s1.view(local, -1),
            fc2_weight_sf=s2.view(local, -1),
            fc1_alpha=self.alpha[0],
            fc2_alpha=self.alpha[1],
            fc1_norm_const=self.alpha[2],
            output_activation=self.output,
            topk_indices=self.indices,
            topk_scores=self.prob,
        )
        self.args = {
            n: ct.from_dlpack(v, assumed_align=4 if n in ("fc1_alpha", "fc2_alpha", "fc1_norm_const") else 16).mark_layout_dynamic(
                leading_dim=ct.get_leading_dim(v)
            )
            for n, v in args.items()
        }

        def ptr(v):
            return cute.runtime.make_ptr(cutlass.Uint8, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)

        self.args.update(
            local_workspace=ptr(self.local),
            shared_workspace=ptr(shared),
            peer_rank_ptr_mapper_host=SymmetricBufferHost(base_address=self.arena.data_ptr(), offsets=offsets, rank=topo.ep_rank, max_ranks=self.ep),
        )
        self.tape = self.local.narrow(0, self.kernel._device_workspace.offset("fe.training_ptrs"), 32).view(torch.int64)
        self.tape[0].fill_(self.inverse.data_ptr())
        self.compiled = cute.compile(self.kernel, **self.args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def gather(self, dest, source):
        if self.ep == 1:
            dest.copy_(source)
        else:
            dist.all_gather_into_tensor(dest, source.contiguous(), group=self.group)

    def bind_weights(self, weights):
        from ._types import MoeEpNvfp4FusedForwardWeights

        args = self.args
        if isinstance(weights, MoeEpNvfp4FusedForwardWeights):
            args = dict(args)
            for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
                value = getattr(weights, name)
                v = value.view(torch.float8_e4m3fn) if name.endswith("_sf") else value.view(torch.float4_e2m1fn_x2).transpose(1, 2)
                args[name] = ct.from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=ct.get_leading_dim(v))
        else:
            pack_weights(weights, self.native_weights)
        for dst, src in zip(self.alpha, (weights.fc31_alpha, weights.fc2_alpha, weights.fc1_norm_const)):
            dst.copy_(src)
        return args

    def forward(self, state, x, ids, prob, weights, record):
        from ._nvfp4_training import route_table

        n = x.shape[0]
        self.indices[:n].copy_(ids)
        self.indices[n:].fill_(-1)
        self.prob[:n].copy_(prob)
        self.prob[n:].zero_()
        self.x_pad[:n].copy_(x)
        self.x_pad[n:].zero_()
        for start in range(0, self.t, 256):
            count = min(256, self.t - start)
            rows = slice(start, start + count)
            dst = (self.data[rows], self.scale[rows]) if count == 256 else (self.tail_data, self.tail_sf)
            state.pack_x.run(self.x_pad[start : start + 256], weights.input_norm, *dst)
            if count < 256:
                self.data[rows].copy_(dst[0][:count])
                self.scale[rows].copy_(dst[1][:count])
        if self.scale is not self.wire_sf:
            self.wire_sf[:, : self.h // 16].copy_(self.scale)
        if self.strict and self.ep > 1:
            self.gather(self.norms, weights.input_norm.reshape(1))
            torch._assert_async((self.norms == self.norms[:1]).all(), "TRT NVFP4 forward requires one checkpoint input scale across EP")
        self.gather(record.fc1_a, self.data)
        self.gather(record.fc1_sfa.view(torch.uint8), self.scale.view(torch.uint8))
        all_ids = state.gather(self.indices, -1)
        slots, counts, ends = route_table(all_ids, state.topology.ep_rank * state.topology.experts_per_rank, state.topology.experts_per_rank, state.rows)
        record.route_slots.copy_(slots)
        record.valid_route_counts.copy_(counts)
        record.expert_offsets.copy_(ends)
        self.inverse.scatter_(0, torch.where(slots >= 0, slots, self.inverse.numel() - 1), self.row_ids)
        record.activation_norms[:, 0].copy_(weights.input_norm.reshape(()))
        record.activation_norms[:, 1].copy_(weights.fc1_norm_const)
        for ix, dest in enumerate((record.fc1_preact, record.saved_a, record.saved_sfa), 1):
            self.tape[ix].fill_(dest.data_ptr())
        args = self.bind_weights(weights)
        self.compiled(**args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        record.output[:n].copy_(self.output[:n])
        return record.output[:n]
