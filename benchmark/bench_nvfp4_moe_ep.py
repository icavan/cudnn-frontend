# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure full pre-FC2 W4A4 backward with torchrun --nproc-per-node=8."""

import argparse
import json
import os
import statistics
from dataclasses import fields, replace

import torch
import torch.distributed as dist


def workload(t, h, e, k, rank, device):
    gen = torch.Generator(device=device).manual_seed(314 + rank)
    x, dy = [torch.randn((t, h), dtype=torch.bfloat16, generator=gen, device=device) for _ in range(2)]
    p, ids = torch.randn((t, e), generator=gen, device=device).topk(k)
    ids = ids.int()
    p = p.softmax(-1)
    return x, dy, ids, p


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
    path = MoeEpDataPathConfig(
        compute_format=MoeFormat.NVFP4,
        gate_up_clamp=7.0,
        fc1_ring_slots=args.fc1_ring_slots,
        fc1_ring_overlap=not args.serial_ring,
        nvfp4_fused_forward=args.trt_fused_forward,
    )
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

    def bundle(cls, spec=requirements):
        return cls(**{f.name: torch.empty(spec[f.name][0], dtype=spec[f.name][2], device=device) for f in fields(cls)})

    rec = bundle(MoeEpTrainingForwardOutputs)

    # Constant master weights have exact independent packs in both directions. E2M1 code 7
    # represents 6, and SF=2^-8 gives master value 6/256 in each position.
    def matrix(n, k):
        data = torch.full((local, n, k // 2), 0x77, dtype=torch.uint8, device=device)
        sf = torch.full((local, n, k // 16), 2**-8, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        return BlockScaledTensor(data, sf, MoeFormat.NVFP4, (local, n, k))

    one = torch.ones(local, dtype=torch.float32, device=device)
    fw = MoeEpNvfp4ForwardWeights(matrix(2 * i, h), matrix(h, i), one[0], one, one, one)
    logical_fw = fw
    if args.prepack_weights:
        from cudnn.moe_ep import MoeEpNvfp4FusedForwardWeights

        fw = MoeEpNvfp4FusedForwardWeights.from_logical(fw)
    bw = MoeEpNvfp4BackwardWeights(matrix(h, 2 * i), matrix(i, h), 1.0, 1.0)
    x, dy, ids, p = workload(t, h, e, k, rank, device)
    if args.forward_only:

        def forward():
            return op.training_forward(x, ids, p, weights=fw, out=rec)

        forward()
        graph = None
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for iteration in range(args.repeats):
            dist.barrier()
            profile = args.profile and rank == 0 and iteration == 0
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.reset_peak_memory_stats()
            start.record()
            if graph is None:
                forward()
            else:
                graph.replay()
            end.record()
            end.synchronize()
            if profile:
                torch.cuda.cudart().cudaProfilerStop()
            metrics = torch.tensor([start.elapsed_time(end), torch.cuda.max_memory_allocated()], device=device, dtype=torch.float64)
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            samples.append(metrics.tolist())
        ring = op._training_state.fc1_ring
        fused = op._training_state.fused
        saved_x = rec.fc1_a.nbytes + rec.fc1_sfa.nbytes
        pool_bytes = saved_x
        if fused is not None:
            comm = fused.kernel.token_comm
            pool_bytes = sum(fused.kernel._device_workspace.nbytes(region) for region in (comm.fc1_activation_region, comm.fc1_activation_sf_region))
        elif ring is not None:
            pool_bytes = ring.payload.nbytes + ring.sf.nbytes
        if args.verify and (ring is not None or fused is not None):
            oracle = MoeEp(replace(config, data_path=replace(path, fc1_ring_slots=0, nvfp4_fused_forward=False)))
            ref = bundle(MoeEpTrainingForwardOutputs, oracle.prepare_training(device=device))
            oracle.training_forward(x, ids, p, weights=logical_fw, out=ref)
            assert torch.equal(ref.route_slots, rec.route_slots)
            valid_rows = torch.where(ref.route_slots >= 0)[0]
            src = ref.route_slots[valid_rows] // k
            for name in ("fc1_a", "fc1_sfa", "fc1_preact", "saved_a", "saved_sfa"):
                index = src if name in ("fc1_a", "fc1_sfa") else valid_rows
                expected = getattr(ref, name).view(torch.uint8).index_select(0, valid_rows)
                actual = getattr(rec, name).view(torch.uint8).index_select(0, index)
                assert torch.equal(actual, expected), (rank, t, name)
            assert torch.equal(rec.output, ref.output), (rank, t, "output")
            oracle.close()
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        tokens_per_rank=t,
                        slots=args.fc1_ring_slots,
                        overlap=not args.serial_ring,
                        forward_ms=statistics.median(v[0] for v in samples),
                        peak_torch_bytes=int(max(v[1] for v in samples)),
                        prepared_bytes=prepared,
                        fc1_pool_bytes=pool_bytes,
                        fused_forward=fused is not None,
                        fused_local_workspace_bytes=fused.local.nbytes if fused else None,
                        fused_symmetric_arena_bytes=fused.arena.nbytes if fused else None,
                        saved_input_bytes=saved_x,
                        saved_record_bytes=sum(getattr(rec, f.name).nbytes for f in fields(rec) if f.name != "output"),
                        full_pool_verified=bool(args.verify and (ring is not None or fused is not None)),
                        live_blocks_rank0=int(rec.expert_offsets[-1]) // 256,
                        full_capacity_rows=rec.route_slots.numel(),
                    )
                ),
                flush=True,
            )
        del graph
        op.close()
        return

    out = bundle(MoeEpTrainingBackwardOutputs)
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
        for iteration in range(args.repeats):
            dist.barrier()
            profile = args.profile and overlap and iteration == 0 and rank == 0
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
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
            if profile:
                torch.cuda.cudart().cudaProfilerStop()
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
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--fc1-ring-slots", type=int, default=0)
    parser.add_argument("--serial-ring", action="store_true")
    parser.add_argument("--prepack-weights", action="store_true")
    parser.add_argument("--trt-fused-forward", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify", action="store_true", help="Compare every gradient byte after timing both schedules")
    parser.add_argument("--graph", action="store_true", help="Measure CUDA Graph replay")
    parser.add_argument("--profile", action="store_true", help="Mark rank 0's first overlapped iteration with cudaProfilerStart/Stop")
    args = parser.parse_args()
    if args.prepack_weights and not args.trt_fused_forward:
        parser.error("--prepack-weights requires --trt-fused-forward")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    for t in args.tokens:
        measure(args, t, rank, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
