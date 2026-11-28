# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure full pre-FC2 W4A4 backward with torchrun --nproc-per-node=8."""

import argparse
import json
import os
import statistics
from dataclasses import fields

import torch
import torch.distributed as dist

V4_PRESETS = {
    # H, intermediate width, routed experts, experts per token, routing multiplier.
    "v4-flash": (4096, 2048, 256, 6, 1.5),
    "v4-pro": (7168, 3072, 384, 6, 2.5),
}


def _profiler_call(action, rank):
    """Check CUDA profiler status and report a rank-0 failure to every rank."""
    error = [None]
    if rank == 0:
        try:
            getattr(torch.cuda.profiler, action)()
        except Exception as exc:
            error[0] = f"CUDA profiler {action} failed: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])


def profile_replay(run, rank):
    """One untimed replay; profiling failures abort before timing collectives."""
    dist.barrier()
    _profiler_call("start", rank)
    try:
        run()
        torch.cuda.synchronize()
    finally:
        _profiler_call("stop", rank)


def configure_model(args):
    args.clamp, args.route_scale = 7.0, None
    if args.model is not None:
        args.hidden, args.intermediate, args.experts, args.top_k, args.route_scale = V4_PRESETS[args.model]
        args.clamp = 10.0


def workload(t, h, e, k, rank, device, route_scale=None):
    gen = torch.Generator(device=device).manual_seed(314 + rank)
    x, dy = [torch.randn((t, h), dtype=torch.bfloat16, generator=gen, device=device) for _ in range(2)]
    if route_scale is not None:
        # Score-based layer following the first three hash-routed V4 layers.
        # Every source rank uses the same synthetic router weights and bias.
        router = torch.Generator(device=device).manual_seed(7781)
        weight = torch.randn((e, h), device=device, generator=router) * h**-0.5
        bias = torch.randn(e, device=device, generator=router) * 0.02
        scores = torch.nn.functional.softplus(torch.nn.functional.linear(x.float(), weight)).sqrt()
        ids = (scores + bias).topk(k, dim=-1).indices
        p = scores.gather(1, ids)
        p = (p / p.sum(dim=1, keepdim=True)) * route_scale
        return x, dy, ids.int(), p
    p, ids = torch.randn((t, e), generator=gen, device=device).topk(k)
    ids = ids.int()
    p = p.softmax(-1)
    return x, dy, ids, p


def measure_training_pair(forward, backward, repeats, device):
    start, boundary, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    samples = []
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        forward()
        boundary.record()
        backward()
        end.record()
        end.synchronize()
        values = torch.tensor(
            [start.elapsed_time(boundary), boundary.elapsed_time(end), start.elapsed_time(end), torch.cuda.max_memory_allocated() - resident],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        samples.append(values.tolist())
    return dict(
        forward_ms=statistics.median(v[0] for v in samples),
        backward_ms=statistics.median(v[1] for v in samples),
        forward_backward_ms=statistics.median(v[2] for v in samples),
        temporary_torch_peak_bytes=int(max(v[3] for v in samples)),
        samples=samples,
    )


def measure(args, t, rank, device):
    from cudnn.moe_ep import (
        BlockScaledTensor,
        MoeEp,
        MoeEpConfig,
        MoeEpDataPathConfig,
        MoeEpModelConfig,
        MoeEpParallelConfig,
        MoeFormat,
        MoeEpNvfp4ForwardWeights,
        MoeEpNvfp4BackwardWeights,
        MoeEpTrainingForwardOutputs,
        MoeEpTrainingBackwardOutputs,
    )

    e, h, i, k = args.experts, args.hidden, args.intermediate, args.top_k
    local = e // dist.get_world_size()
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=args.clamp)
    config = MoeEpConfig(
        model=MoeEpModelConfig(num_experts=e, hidden_size=h, intermediate_size=i, top_k=k),
        parallel=MoeEpParallelConfig(ep_group=dist.group.WORLD, max_tokens_per_rank=t),
        data_path=path,
        validation_mode="trusted",
    )
    op = MoeEp(config)
    before = torch.cuda.memory_allocated()
    requirements = op.prepare_training(device=device)
    prepared = torch.cuda.memory_allocated() - before

    def bundle(cls):
        return cls(**{f.name: torch.empty(requirements[f.name][0], dtype=requirements[f.name][2], device=device) for f in fields(cls)})

    rec = bundle(MoeEpTrainingForwardOutputs)
    out = bundle(MoeEpTrainingBackwardOutputs)

    # Constant master weights have exact independent packs in both directions. E2M1 code 7
    # represents 6, and SF=2^-8 gives master value 6/256 in each position.
    def matrix(n, k):
        data = torch.full((local, n, k // 2), 0x77, dtype=torch.uint8, device=device)
        sf = torch.full((local, n, k // 16), 2**-8, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        return BlockScaledTensor(data, sf, MoeFormat.NVFP4, (local, n, k))

    one = torch.ones(local, dtype=torch.float32, device=device)
    fw = MoeEpNvfp4ForwardWeights(matrix(2 * i, h), matrix(h, i), one[0], one, one, one)
    bw = MoeEpNvfp4BackwardWeights(matrix(h, 2 * i), matrix(i, h), 1.0, 1.0)
    x, dy, ids, p = workload(t, h, e, k, rank, device, args.route_scale)
    w1, w2 = [torch.empty(shape, dtype=torch.float32, device=device) for shape in ((local, 2 * i, h), (local, h, i))]
    op.training_forward(x, ids, p, weights=fw, out=rec)
    bundle_bytes = sum(getattr(out, f.name).nbytes for f in fields(out))
    peer_bytes = op._training_state.exchange.heap.nbytes
    saved = sum(getattr(rec, f.name).nbytes for f in fields(rec) if f.name != "output")
    active_tiles = int(((rec.valid_route_counts + 255) // 256).sum())
    reports = []
    serial_outputs = None
    if args.graph:
        op._training_state.dx_ready = torch.cuda.Event(enable_timing=True, external=True)
        op._training_state.weight_done = torch.cuda.Event(enable_timing=True, external=True)
    for overlap in (False, True):

        def run():
            op.training_backward(dy, ids, p, weights=bw, forward_record=rec, out=out, dw1=w1, dw2=w2, overlap=overlap)

        run()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph = None
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        samples = []
        if args.profile and overlap:
            profile_replay(run if graph is None else graph.replay, rank)
        for iteration in range(args.repeats):
            dist.barrier()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            start.record()
            if graph is None:
                run()
            else:
                graph.replay()
            stop.record()
            stop.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"] - allocations
            stats = torch.tensor(
                [
                    start.elapsed_time(op._training_state.dx_ready),
                    start.elapsed_time(stop),
                    start.elapsed_time(op._training_state.weight_done),
                    torch.cuda.max_memory_allocated() - resident,
                    allocations,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            samples.append(stats.tolist())
        reports.append(
            dict(
                overlap=overlap,
                dx_ms=statistics.median(v[0] for v in samples),
                backward_ms=statistics.median(v[1] for v in samples),
                wgrad_ms=statistics.median(v[2] for v in samples),
                samples=[dict(dx_ms=v[0], backward_ms=v[1], wgrad_ms=v[2]) for v in samples],
                temporary_torch_peak_bytes=int(max(v[3] for v in samples)),
                torch_allocations=int(max(v[4] for v in samples)),
            )
        )
        if overlap and args.measure_step:
            reports[-1]["training_pair"] = measure_training_pair(
                lambda: op.training_forward(x, ids, p, weights=fw, out=rec), graph.replay if graph is not None else run, args.repeats, device
            )
        if args.verify:
            outputs = (out.grad_activation, out.dprob, w1, w2)
            if not overlap:
                serial_outputs = tuple(t.clone() for t in outputs)
            else:
                for name, actual, reference in zip(("dx", "dp", "dw1", "dw2"), outputs, serial_outputs):
                    if not torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)):
                        raise AssertionError(f"rank {rank}: {name} differs between serial and overlapped execution")
        del graph
    if rank == 0:
        print(
            json.dumps(
                dict(
                    graph=args.graph,
                    bitwise_verified=args.verify,
                    active_tiles_rank0=active_tiles,
                    capacity_tiles=rec.route_slots.numel() // 256,
                    model=args.model,
                    tokens_per_rank=t,
                    hidden=h,
                    intermediate=i,
                    experts=e,
                    top_k=k,
                    prepared_torch_bytes=prepared,
                    peer_heap_bytes=peer_bytes,
                    backward_bundle_bytes=bundle_bytes,
                    saved_bytes=saved,
                    runs=reports,
                )
            ),
            flush=True,
        )
    op.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=V4_PRESETS, help="Use the named DeepSeek-V4 routed expert configuration and synthetic score-based router")
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify", action="store_true", help="Compare every gradient byte after timing both schedules")
    parser.add_argument("--graph", action="store_true", help="Measure CUDA Graph replay")
    parser.add_argument("--measure-step", action="store_true", help="Also time eager forward followed by the overlapped backward")
    parser.add_argument("--profile", action="store_true", help="Profile an extra untimed overlapped replay on rank 0")
    args = parser.parse_args()
    configure_model(args)
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    for t in args.tokens:
        measure(args, t, rank, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
