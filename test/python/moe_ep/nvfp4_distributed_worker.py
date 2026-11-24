# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eight-rank NVLink equivalence and mutable-routing CUDA Graph regression."""

import argparse
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from test_nvfp4_training import (
    MoeEp,
    MoeEpConfig,
    MoeEpModelConfig,
    MoeEpDataPathConfig,
    MoeEpParallelConfig,
    MoeFormat,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingBackwardOutputs,
    buffers,
    fixture,
    oracle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-fused-forward", action="store_true")
    parser.add_argument("--fc1-ring-slots", type=int, default=0)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    t, h, i, k = 33, 128, 128, 2
    local_experts = 2
    experts = 8 * local_experts
    path = MoeEpDataPathConfig(
        compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0, fc1_ring_slots=args.fc1_ring_slots, nvfp4_fused_forward=args.trt_fused_forward
    )
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=experts, hidden_size=h, intermediate_size=i, top_k=k),
            parallel=MoeEpParallelConfig(ep_group=dist.group.WORLD, max_tokens_per_rank=t),
            data_path=path,
            validation_mode="trusted",
        )
    )
    req = op.prepare_training(device=device)
    rec = buffers(req, MoeEpTrainingForwardOutputs, device)
    out = buffers(req, MoeEpTrainingBackwardOutputs, device)
    fw, bw, w1, w2 = fixture(local_experts, h, i, device)
    gen = torch.Generator(device=device).manual_seed(523)
    tmp = [torch.randn((8 * t, h), generator=gen, device=device, dtype=torch.bfloat16) for _ in range(2)]
    src = slice(rank * t, (rank + 1) * t)
    x, dy = [v[src] for v in tmp]
    prob = torch.rand((8 * t, k), device=device, generator=gen)
    prob[::5, 0] = 0
    prob[::7, 1] = 1
    ids = torch.zeros((8 * t, k), device=device, dtype=torch.int32)
    dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)

    def run(overlap):
        y = op.training_forward(x, ids[src], prob[src], weights=fw, out=rec)
        debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            dx, dp, _ = op.training_backward(dy, ids[src], prob[src], weights=bw, forward_record=rec, out=out, overlap=overlap, dw1=dw1, dw2=dw2)
        finally:
            torch.cuda.set_sync_debug_mode(debug_mode)
        return y, dx, dp, dw1, dw2

    actual = run(True)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    exchange = op._training_state.exchange
    publish = exchange.put
    # Delay publication on one rank inside the captured schedule. Other
    # ranks can publish newer tiles before that rank's older tile arrives.

    def delayed(*args):
        if rank == 7:
            with torch.cuda.stream(exchange.writer):
                torch.cuda._sleep(500_000)
        publish(*args)

    exchange.put = delayed
    with torch.cuda.graph(g):
        run(True)
    exchange.put = publish
    # The first two calls cross zero. Later replays also cross signed overflow.
    exchange.gen.fill_(-2)
    for hot in (0, experts - 1, 3):
        if hot == experts - 1:
            exchange.gen.fill_(2**31 - 2)
        ids[:, 0].fill_(hot)
        if hot == 3:
            ids[:, 1].fill_(hot)  # Seven ranks own no routes; both slots target the same expert.
        else:
            ids[:, 1].copy_(torch.arange(8 * t, device=device) % experts)
        exchange.combined.fill_(float("nan"))
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        profiling = rank == 0 and hot == 0 and bool(os.environ.get("NVFP4_PROFILE_REPLAY"))
        if profiling:
            torch.cuda.cudart().cudaProfilerStart()
        g.replay()
        torch.cuda.synchronize()
        if profiling:
            torch.cuda.cudart().cudaProfilerStop()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        assert torch.all(exchange.flags == (exchange.gen + 1) * k), (rank, hot, "arrival count")
        recorded = [a.clone() for a in actual]
        serial = run(False)
        expected = oracle(tmp[0], tmp[1], ids, prob, w1.repeat(8, 1, 1), w2.repeat(8, 1, 1), 4.0, fw.fc1_norm_const.repeat(8), 8.0, 4.0, path)
        for idx, (r, s, ref) in enumerate(zip(recorded, serial, expected[:5])):
            assert torch.equal(r.view(torch.uint8), s.view(torch.uint8)), (rank, hot, idx)
            torch.testing.assert_close(r, ref[src] if idx < 3 else ref[rank * local_experts : (rank + 1) * local_experts], rtol=5e-5, atol=5e-5)
    if args.trt_fused_forward:
        for live in (rank, t - rank):
            observed = op.training_forward(x[:live], ids[src][:live], prob[src][:live], weights=fw, out=rec)
            torch.testing.assert_close(observed, expected[0][src][:live], rtol=5e-5, atol=5e-5)
    dist.barrier()
    if rank == 0:
        print("EP8 serial/overlap bitwise equal; three changed-route graph replays match independent FP4 oracle", flush=True)
    del g
    torch.cuda.synchronize()

    op.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
