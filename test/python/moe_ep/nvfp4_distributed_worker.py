# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eight-rank NVLink equivalence and mutable-routing CUDA Graph regression."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from unittest.mock import patch
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from nvfp4_test_support import owned_graph
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
    physical_scales,
)


def full_k_wgrad(a, sfa, b, sfb, norm):
    """Independent per-expert cuBLAS reference on the oracle-checked NVFP4 bytes."""
    from torch.nn import functional as F

    def operand(data, scales):
        # Public tiles are [tile, feature, packed K]. cuBLAS consumes one
        # contiguous K per feature and uses 128-row scale atoms.
        payload = data.transpose(0, 1).flatten(1).contiguous().view(torch.float4_e2m1fn_x2)
        sf = scales.view(torch.uint8).transpose(0, 1).flatten(1).contiguous().view(torch.float8_e4m3fn)
        return payload, physical_scales(sf).flatten()

    lhs, scale_l = operand(a, sfa)
    rhs, scale_r = operand(b, sfb)
    recipe = F.ScalingType.BlockWise1x16
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    accum = F.scaled_mm(lhs, rhs.T, scale_l, recipe, scale_r, recipe, swizzle, swizzle, output_dtype=torch.float32)
    return (accum.double() / norm).float()


@contextmanager
def delayed_transfers(state):
    exchange = state.exchange
    with ExitStack() as patches:

        def delay(owner, name, stream):
            launch = getattr(owner, name)

            def delayed(*args):
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(500_000)
                return launch(*args)

            patches.enter_context(patch.object(owner, name, delayed))

        delay(exchange, "get", exchange.get_stream)
        delay(exchange, "put", exchange.put_stream)
        # Window widths can share a batch. The direct-store variant publishes
        # readiness through notify after the MMA has written its peer slots.
        for batch in set(state.batches.values()):
            if batch.packed is None:
                delay(batch, "get", exchange.get_stream)
            else:
                delay(batch.fc2.pack, "receive", exchange.get_stream)
            delay(batch, "notify" if batch.notify is not None else "put", exchange.put_stream)
        yield


def main():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--nonunit-norms", action="store_true", help="Exercise FP32 division after each GEMM")
    args = parser.parse_args()
    t, h, i, k = args.tokens, args.hidden, args.intermediate, args.top_k
    local_experts = 2
    experts = 8 * local_experts
    if not 2 <= k <= experts:
        parser.error(f"the duplicate-slot regression requires top-k between 2 and {experts}")
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0)
    if args.nonunit_norms:
        path = replace(path, dy_norm=3.0, dgu_norm=5.0, x_wgrad_norm=1.25, a_wgrad_norm=7.0)
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=experts, hidden_size=h, intermediate_size=i, top_k=k),
            parallel=MoeEpParallelConfig(ep_group=dist.group.WORLD, max_tokens_per_rank=t),
            data_path=path,
            validation_mode="trusted",
        )
    )
    # A locally unsupported layout must still enter the common preflight:
    # otherwise the other seven ranks wait in all_gather_object.
    mixed_configs = [
        replace(op.config, data_path=MoeEpDataPathConfig() if rank == 0 else path),
        replace(op.config, parallel=replace(op.config.parallel, token_padding_size=((8 * t * k + 255) // 256 + 1) * 256 if rank == 0 else 128)),
    ]
    for config in mixed_configs:
        with MoeEp(config) as mixed:
            try:
                mixed.prepare_training(device=device)
            except RuntimeError as exc:
                assert "must match on every" in str(exc)
            else:
                raise AssertionError("mixed training precision or resolved capacity was accepted")
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
    with owned_graph() as g, patch("cutlass.cute.compile", side_effect=AssertionError("unexpected runtime compilation")):
        exchange = op._training_state.exchange

        with delayed_transfers(op._training_state) if rank == 7 else nullcontext():
            with torch.cuda.graph(g):
                run(True)
        # The first two calls cross zero. Later replays also cross signed overflow.
        exchange.gen.fill_(-2)
        for hot in (0, experts - 1, 3, 2):
            if hot == experts - 1:
                exchange.gen.fill_(2**31 - 2)
            ids.fill_(hot)
            if hot == 2:
                # Put two different local experts in one dgrad window.
                ids[:, 1].fill_(hot + 1)
            elif hot == 3:
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
            first = rank * local_experts
            bounds = [torch.zeros_like(dw1), torch.zeros_like(dw2)] if k > 2 else []

            def check_operands(expert, begin, gu, xx, yy, aa):
                if not first <= expert < first + local_experts:
                    return
                local = expert - first
                start = int(rec.expert_offsets[local - 1]) if local else 0
                tile = (start + begin) // 256
                for name, data, sf, expected_pack in (
                    ("dGU", out.fc1_b, out.fc1_sfb, gu),
                    ("X", out.fc1_wgrad_a, out.fc1_wgrad_sfa, xx),
                    ("dY", out.fc2_b, out.fc2_sfb, yy),
                    ("A", out.fc2_a, out.fc2_sfa, aa),
                ):
                    for actual_bytes, expected_bytes in ((data[tile], expected_pack[0]), (sf[tile], expected_pack[1])):
                        torch.testing.assert_close(
                            actual_bytes.view(torch.uint8),
                            expected_bytes.view(torch.uint8),
                            rtol=0,
                            atol=0,
                            msg=lambda m: f"{rank=} {hot=} {local=} {tile=} {name=}\n{m}",
                        )
                # Compare long K-grouped accumulations with the sum of 256-route
                # reference GEMMs. Keep cancellation-sensitive error in units of
                # the absolute product sum, alongside the full-K cuBLAS check.
                for bound, lhs, rhs, divisor in ((bounds[0], gu, xx, path.dgu_norm * path.x_wgrad_norm), (bounds[1], yy, aa, path.dy_norm * path.a_wgrad_norm)):
                    bound[local] += (lhs[2].abs() @ rhs[2].abs().T) / divisor

            expected = oracle(
                tmp[0],
                tmp[1],
                ids,
                prob,
                w1.repeat(8, 1, 1),
                w2.repeat(8, 1, 1),
                4.0,
                fw.fc1_norm_const.repeat(8),
                8.0,
                4.0,
                path,
                check_operands=check_operands if k > 2 else None,
            )
            for idx, (r, s, ref) in enumerate(zip(recorded, serial, expected[:5])):
                assert torch.equal(r.view(torch.uint8), s.view(torch.uint8)), (rank, hot, idx)
                ref = ref[src] if idx < 3 else ref[first : first + local_experts]
                if bounds and idx >= 3:
                    error = (r - ref).abs()
                    tiles = rec.expert_offsets.diff(prepend=rec.expert_offsets.new_zeros(1)) // 256
                    tol = 2 * torch.finfo(torch.float32).eps * tiles[:, None, None] * bounds[idx - 3]
                    budget = 5e-5 + 5e-5 * ref.abs() + tol
                    assert (error <= budget).all(), (rank, hot, idx)
                    # A long continuous Tensor Core accumulation rounds differently
                    # from summing the oracle's 256-row partial GEMMs. Compare the
                    # grouped result to a separate full-K GEMM for each expert.
                    args = (
                        (out.fc1_b, out.fc1_sfb, out.fc1_wgrad_a, out.fc1_wgrad_sfa, path.dgu_norm * path.x_wgrad_norm)
                        if idx == 3
                        else (out.fc2_b, out.fc2_sfb, out.fc2_a, out.fc2_sfa, path.dy_norm * path.a_wgrad_norm)
                    )
                    begin = 0
                    for expert, end in enumerate((rec.expert_offsets // 256).tolist()):
                        if begin < end:
                            native = full_k_wgrad(*(v[begin:end] for v in args[:4]), args[4])
                            # cuBLAS may choose a different long-K accumulation
                            # tree. Use the same FP32 budget as the independent
                            # tiled oracle; serial/overlap equality above remains
                            # byte-exact. This is not an extra error allowance.
                            assert ((r[expert] - native).abs() <= budget[expert]).all(), (rank, hot, idx, expert, "full-K reference")
                        else:
                            assert torch.count_nonzero(r[expert]) == 0
                        begin = end
                else:
                    torch.testing.assert_close(r, ref, rtol=5e-5, atol=5e-5, msg=lambda m: f"{rank=} {hot=} {idx=}\n{m}")

            if hot == 3:
                compact = op.compact_training_record(rec)
                assert compact.route_slots.numel() == max(256, int(rec.expert_offsets[-1]))
                dx, dp, _ = op.training_backward(dy, ids[src], prob[src], weights=bw, forward_record=compact, out=out, dw1=dw1, dw2=dw2)
                for value, expected_value in zip((dx, dp, dw1, dw2), recorded[1:]):
                    assert torch.equal(value.view(torch.uint8), expected_value.view(torch.uint8)), (rank, hot, "retained record")

        # Exercise successive backward-only generations without a forward collective
        # or an intervening host synchronization, with a deliberately slow consumer.
        def backward_only(overlap):
            return op.training_backward(dy, ids[src], prob[src], weights=bw, forward_record=rec, out=out, overlap=overlap)[:2]

        references = []
        for _ in range(2):
            dy.neg_()
            references.append(tuple(v.clone() for v in backward_only(False)))
        drain = exchange.combine_ready

        producers_done = torch.cuda.Event(external=True)
        consumer_done = torch.cuda.Event(external=True)

        def slow_consumer(stream):
            if rank == 0 and stream != exchange.put_stream:
                with torch.cuda.stream(stream):
                    # finish() has already joined all producers at its first
                    # peer barrier. Keep the consumer pending before reuse.
                    producers_done.record(stream)
                    torch.cuda._sleep(200_000_000)
            drain(stream)
            if rank == 0 and stream != exchange.put_stream:
                consumer_done.record(stream)

        with owned_graph() as backward_graph:
            with patch.object(exchange, "combine_ready", slow_consumer):
                with torch.cuda.graph(backward_graph):
                    dx_dp = backward_only(True)
            samples = [tuple(torch.empty_like(v) for v in dx_dp) for _ in range(4)]
            for outputs in samples:
                dy.neg_()
                backward_graph.replay()
                if rank == 0:
                    producers_done.synchronize()
                    assert not consumer_done.query(), "delayed consumer completed before the lifetime checkpoint"
                for dest, value in zip(outputs, dx_dp):
                    dest.copy_(value)
            torch.cuda.synchronize()
            for iteration, outputs in enumerate(samples):
                for actual_value, expected_value in zip(outputs, references[iteration % 2]):
                    assert torch.equal(actual_value.view(torch.uint8), expected_value.view(torch.uint8)), (rank, iteration, "peer reuse")
        # The sole source token is remote to its expert. Padding must never
        # consume rank zero's stale probability, even when it is NaN.
        count = int(rank == 1)
        ids.zero_()
        exchange.prob.fill_(float("nan"))
        y = op.training_forward(x[:count], ids[src][:count], prob[src][:count], weights=fw, out=rec)
        dx, dp, _ = op.training_backward(dy[:count], ids[src][:count], prob[src][:count], weights=bw, forward_record=rec, out=out, dw1=dw1, dw2=dw2)
        if rank == 0:
            assert rec.valid_route_counts.tolist() == [k, 0]
            assert not torch.count_nonzero(rec.saved_sfa[k:256].view(torch.uint8))
            assert not torch.count_nonzero(rec.saved_a[k:256])
        for value in (y, dx, dp, dw1, dw2):
            assert torch.isfinite(value).all()
        dist.barrier()
        if rank == 0:
            print("EP8 serial/overlap bitwise equal; four changed-route graph replays match independent FP4 oracle", flush=True)
    torch.cuda.synchronize()

    op.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
