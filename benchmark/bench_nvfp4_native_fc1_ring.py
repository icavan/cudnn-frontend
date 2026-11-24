# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare original TRT-LLM and FE FC1 pool geometries without a training tape on one EP8 domain."""

import argparse
import importlib
import json
import os
import statistics
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.torch as ct
import cutlass.utils
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from bench_nvfp4_moe_ep import workload
from cudnn.moe_ep._nvfp4_pack import PackTile


@dataclass
class CapturedForward:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    workspace_bytes: dict
    # Graph nodes retain raw device addresses. Own all bindings until the last replay finishes.
    keepalive: tuple


def prepared(package, slots, x, ids, scores, weights, topk, hidden, intermediate, experts, clusters):
    def module(path):
        return importlib.import_module(f"{package}.{path}")

    desc = module("api")
    rank = dist.get_rank()
    t = x.shape[0]
    device = x.device
    problem = desc.ProblemDesc(
        dict(
            expert_count=experts,
            intermediate_gateup_size=2 * intermediate,
            hidden_size=hidden,
            quant_kind="nvfp4",
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
            combine_format=module("quant_def").CombineFormat.parse("bf16"),
            gate_up_clamp=7.0,
            world_size=8,
            topk=topk,
            topk_index_dtype=cutlass.Int64,
            max_tokens_per_rank=t,
            apply_topk_at_fc1=True,
        )
    )
    settings = dict(
        mma_tiler_mnk=(256, 128 if t <= 1024 else 256, 256),
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        schedule_policy=("grouped", 512),
        token_padding_block=64,
        sf_padding_block=128,
        work_id_mode="grid_stride",
        fc2_use_bulk=True,
        epi_flag_batches=(1, 1),
        launch_cluster_count=clusters,
        token_in_flag_batch=1 if t <= 1024 else 4,
        token_back_mode="epi_warps",
        reduce_topk_in_kernel=False,
    )
    if slots:
        settings["fc1_ring_slots"] = slots
    cls = module("kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel").BlockScaledSwapAbMegaMoeKernel
    kernel = cls(problem, desc.ImplDesc(settings))
    private, shared = kernel.get_workspace_sizes()
    sf_pitch = kernel.token_comm.activation_sf_hidden_padded
    sizes = (t * hidden // 2, t * sf_pitch, t * topk * 4, t * hidden * 2, shared)
    arena_bytes = sum((n + 255) // 256 * 256 for n in sizes)
    arena = symm.empty(arena_bytes, dtype=torch.uint8, device=device)
    handle = symm.rendezvous(arena, dist.group.WORLD)
    arena.zero_()
    offsets = tuple(handle.get_buffer(r, (arena_bytes,), torch.uint8).data_ptr() - arena.data_ptr() for r in range(8))
    local = torch.zeros(private, dtype=torch.uint8, device=device)
    off = 0
    chunks = []
    for n in sizes:
        chunks.append(arena[off : off + n])
        off += (n + 255) // 256 * 256
    payload, scale, p, y, workspace = chunks
    payload = payload.view(t, hidden // 2)
    scale = scale.view(torch.float8_e4m3fn).view(t, sf_pitch)
    y = y.view(torch.bfloat16).view(t, hidden)
    p = p.view(torch.float32).view_as(scores)
    ids_buf = torch.empty_like(ids, dtype=torch.int64)
    norm = torch.ones(1, device=device)
    alpha = torch.ones(experts // 8, device=device)
    pack = PackTile(hidden, torch.bfloat16, False, False, forward=True)

    def stage():
        for start in range(0, t, 256):
            sl = slice(start, start + 256)
            pack.run(x[sl], norm, payload[sl], scale[sl])
        ids_buf.copy_(ids)
        p.copy_(scores)

    args = dict(
        activation=payload.view(torch.float4_e2m1fn_x2),
        activation_sf=scale,
        topk_indices=ids_buf,
        topk_scores=p,
        fc1_alpha=alpha,
        fc2_alpha=alpha,
        fc1_norm_const=alpha,
        output_activation=y,
        **weights,
    )
    args = {
        key: ct.from_dlpack(v, assumed_align=4 if key in ("fc1_alpha", "fc2_alpha", "fc1_norm_const") else 16).mark_layout_dynamic(
            leading_dim=ct.get_leading_dim(v)
        )
        for key, v in args.items()
    }

    def ptr(v):
        return cute.runtime.make_ptr(cutlass.Uint8, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=128)

    args.update(
        local_workspace=ptr(local),
        shared_workspace=ptr(workspace),
        peer_rank_ptr_mapper_host=module("communication.nvlink_domain.symmetric_buffer").SymmetricBufferHost(
            base_address=arena.data_ptr(), offsets=offsets, rank=rank, max_ranks=8
        ),
    )
    compiled = cute.compile(kernel, **args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def forward():
        stage()
        compiled(**args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    comm = kernel.token_comm
    pool_bytes = sum(kernel._device_workspace.nbytes(name) for name in (comm.fc1_activation_region, comm.fc1_activation_sf_region))
    return CapturedForward(
        graph,
        y,
        dict(pool_bytes=pool_bytes, local_workspace_bytes=private, symmetric_arena_bytes=arena_bytes),
        (handle, local, arena, alpha, args, compiled, forward),
    )


def measure(t, options):
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    h, i, e, k = 7168, 2048, 256, 8
    x, _, ids, scores = workload(t, h, e, k, rank, device)
    original_ids = ids.clone()
    gen = torch.Generator(device=device).manual_seed(8181 + rank)
    weights = {}
    for name, n, width in (("fc1", 2 * i, h), ("fc2", h, i)):
        data = torch.randint(0, 256, (e // 8, n, width // 2), device=device, dtype=torch.uint8, generator=gen)
        sf = torch.randint(28, 42, (e // 8, n * width // 16), device=device, dtype=torch.uint8, generator=gen)
        weights[name + "_weight"] = data.view(torch.float4_e2m1fn_x2).transpose(1, 2)
        weights[name + "_weight_sf"] = sf.view(torch.float8_e4m3fn)
    clusters = int(cutlass.utils.HardwareInfo().get_max_active_clusters(2))
    plans = {}
    for name, package, s in [("trt", "cutedsl_megamoe", 0)] + [(str(s), "cudnn.moe_ep._trt_blackwell", s) for s in options.slots]:
        plans[name] = prepared(package, s, x, ids, scores, weights, k, h, i, e, clusters)
    for mode in ("balanced", "hot", "duplicates", "balanced"):
        if mode == "balanced":
            ids.copy_(original_ids)
        elif mode == "hot":
            ids.copy_(torch.arange(k, device=device).expand_as(ids))
        else:
            ids.copy_(torch.arange(t, device=device)[:, None] % e)
        for plan in plans.values():
            plan.graph.replay()
        torch.cuda.synchronize()
        for name, plan in plans.items():
            assert torch.equal(plan.output, plans["trt"].output), (rank, t, name, mode)
    names = list(plans)
    times = {name: [] for name in names}
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for it in range(options.repeats + 3):
        for j in range(len(plans)):
            name = names[(it + j) % len(plans)]
            dist.barrier()
            torch.cuda.synchronize()
            begin.record()
            plans[name].graph.replay()
            end.record()
            end.synchronize()
            ms = torch.tensor(begin.elapsed_time(end), device=device)
            dist.all_reduce(ms, op=dist.ReduceOp.MAX)
            if it >= 3:
                times[name].append(ms.item())
    if rank == 0:
        print(
            json.dumps(
                dict(
                    tokens=t,
                    verified=True,
                    routing_tests=["balanced", "hot", "duplicates", "balanced"],
                    results={name: dict(median_ms=statistics.median(times[name]), samples=times[name], **plan.workspace_bytes) for name, plan in plans.items()},
                )
            ),
            flush=True,
        )
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--slots", type=int, nargs="+", default=[0, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=15)
    options = parser.parse_args()
    if any(t <= 0 or t % 256 for t in options.tokens):
        parser.error("the benchmark packs whole 256-token tiles")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    if dist.get_world_size() != 8:
        raise ValueError("the native FC1 ring benchmark requires an EP8 process group")
    for t in options.tokens:
        measure(t, options)
    dist.destroy_process_group()
