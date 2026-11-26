# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 or native W4A8 MoE backward using DeepGEMM grouped GEMMs and NCCL all-to-all."""

import argparse
import json
import os
import statistics

import deep_gemm as dg
import torch
import torch.distributed as dist

from bench_nvfp4_moe_ep import V4_PRESETS, configure_model, workload


def gather(t):
    result = t.new_empty((dist.get_world_size() * len(t), *t.shape[1:]))
    dist.all_gather_into_tensor(result, t.contiguous())
    return result


@torch.compile(fullgraph=True)
def activation_backward(da, pre, p, limit):
    g, u = pre.float().chunk(2, -1)
    gc, uc = g.clamp(max=limit), u.clamp(-limit, limit)
    sigmoid = gc.sigmoid()
    s = gc * sigmoid
    ds = da.float() * p[:, None]
    dg = torch.where(g < limit, ds * uc * (sigmoid + s * (1 - sigmoid)), 0.0)
    du = torch.where(u.abs() < limit, ds * s, 0.0)
    return torch.cat((dg, du), -1).bfloat16(), (da.float() * (s * uc)).sum(-1)


@torch.compile(fullgraph=True)
def combine(dx, dp, inverse):
    return dx[inverse].float().sum(1).bfloat16(), dp[inverse]


@torch.compile(fullgraph=True)
def fp8_operand(value, token_axis):
    # Expert starts are aligned to 32 routes. The token-axis groups therefore
    # never share a scale between experts, including when an expert is empty.
    rows, cols = value.shape
    if token_axis:
        blocks = value.view(-1, 32, cols).float()
        axis = 1
    else:
        blocks = value.view(rows, -1, 128).float()
        axis = 2
    sf = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(axis).clamp_min(1e-4) / 448)))
    return (blocks / sf.unsqueeze(axis)).to(torch.float8_e4m3fn).view_as(value), sf


def fp4_weights(weights):
    from deep_gemm.utils.math import per_token_cast_to_fp4

    packed = [per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32) for w in weights]
    codes, scales = [torch.stack(parts) for parts in zip(*packed)]
    groups, n, k = weights.shape
    return codes, dg.transform_sf_into_required_layout(scales, n, k, (1, 32), groups)


def saved_fp8_operand(value, ks, ks_cpu, alignment):
    data, sf = fp8_operand(value, True)
    # Forward-owned operand. Cache the same packed scale layout that the native
    # K-grouped call otherwise regenerates from FP32 scales on every backward.
    return data, dg._C.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks, ks_cpu, 32, alignment)


def mixed_dgrad(value, bank, out, layout):
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(fp8_operand(value, False), bank, out, layout, recipe_a=(1, 128), recipe_b=(1, 32))


def mixed_wgrad(value, saved, out, ks_cpu, ks):
    out.zero_()
    dg.k_grouped_fp8_gemm_tn_contiguous(fp8_operand(value, True), saved, out, ks_cpu, ks, out, recipe=(1, 1, 32))


def ref_fp8(value, *, token_axis=False):
    group = 32 if token_axis else 128
    if token_axis:
        value = value.T
    width = value.shape[1]
    blocks = torch.nn.functional.pad(value.float(), (0, -width % group)).reshape(len(value), -1, group)
    sf = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    decoded = ((blocks / sf).to(torch.float8_e4m3fn).float() * sf).flatten(1)[:, :width]
    return decoded.T if token_axis else decoded


def ref_fp4(value):
    from deep_gemm.utils.math import per_token_cast_to_fp4

    packed, sf = per_token_cast_to_fp4(value, use_ue8m0=True, gran_k=32)
    codes = torch.stack((packed & 15, (packed >> 4) & 15), -1).flatten(1).long()
    lut = value.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    return lut[codes & 7] * torch.where(codes < 8, 1, -1) * sf.repeat_interleave(32, dim=1)


def measure(args, t, rank, device):
    mixed = args.precision == "w4a8"
    h, i, e, k = args.hidden, args.intermediate, args.experts, args.top_k
    world = dist.get_world_size()
    assert e % world == 0
    local = e // world
    x, dy, ids, p = workload(t, h, e, k, rank, device, args.route_scale)
    all_ids, all_x, all_p = gather(ids).flatten(), gather(x), gather(p).flatten()
    # The forward pass owns the route handle. All-to-all preserves source rank and
    # within-source order, including distinct top-k slots for the same expert.
    send_order = (ids.flatten() // local).argsort(stable=True)
    send_counts = torch.bincount((ids.flatten() // local).long(), minlength=world).tolist()
    send_tokens = send_order // k
    inverse = send_order.argsort().view(t, k)
    slots = (all_ids // local == rank).nonzero().flatten()
    recv_counts = torch.bincount(slots // (t * k), minlength=world).tolist()
    sorted_expert, order = (all_ids[slots] % local).sort(stable=True)
    counts = sorted_expert.bincount(minlength=local)
    align = dg.get_theoretical_mk_alignment_for_contiguous_layout(t * k * world // e)
    dg.set_mk_alignment_for_contiguous_layout(align)
    ks = ((counts + align - 1) // align * align).int()
    ks_cpu = ks.tolist()
    extent = sum(ks_cpu)
    sorted_rows = torch.arange(len(slots), device=device) + ((ks - counts).cumsum(0) - (ks - counts))[sorted_expert.long()]
    recv_rows = torch.empty_like(order)
    recv_rows[order] = sorted_rows
    layout = torch.arange(local, device=device, dtype=torch.int32).repeat_interleave(ks.long())
    expert_x = x.new_zeros((extent, h))
    expert_x[recv_rows] = all_x[slots // k]
    prob = p.new_zeros(extent)
    prob[recv_rows] = all_p[slots]

    def weight(shape):
        if args.check:
            return (torch.randn(shape, generator=weight_gen, device=device) / h**0.5).bfloat16()
        return torch.full(shape, 6 / 256, device=device, dtype=torch.bfloat16)

    weight_gen = torch.Generator(device=device).manual_seed(759 + rank)
    w1, w2 = weight((local, 2 * i, h)), weight((local, h, i))
    pre, da, dx = [x.new_empty((extent, n)) for n in (2 * i, i, h)]
    if mixed:
        mixed_dgrad(expert_x, fp4_weights(w1), pre, layout)
    else:
        dg.m_grouped_bf16_gemm_nt_contiguous(expert_x, w1, pre, layout)
    gate, up = pre.float().chunk(2, -1)
    a = (torch.nn.functional.silu(gate.clamp(max=args.clamp)) * up.clamp(-args.clamp, args.clamp) * prob[:, None]).bfloat16()
    # Pack dgrad weights once per weight update, like the FE baseline.
    w1t, w2t = [w.transpose(1, 2).contiguous() for w in (w1, w2)]
    if mixed:
        w1t, w2t = fp4_weights(w1t), fp4_weights(w2t)
        saved_x, saved_a = (saved_fp8_operand(v, ks, ks_cpu, align) for v in (expert_x, a))
    dgrad = mixed_dgrad if mixed else dg.m_grouped_bf16_gemm_nt_contiguous
    dw1, dw2 = [torch.empty_like(w, dtype=torch.float32) for w in (w1, w2)]
    yt = torch.zeros_like(expert_x)
    received = x.new_empty((len(slots), h))
    returned = x.new_empty((t * k, h))
    returned_p = p.new_empty(t * k)
    worker = torch.cuda.Stream(device=device)
    dy_ready, gu_ready, join = [torch.cuda.Event() for _ in range(3)]
    dx_time, w_time = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]

    def run():
        main = torch.cuda.current_stream()
        dist.all_to_all_single(received, dy.index_select(0, send_tokens), recv_counts, send_counts)
        yt.index_copy_(0, recv_rows, received)
        dy_ready.record(main)
        with torch.cuda.stream(worker):
            worker.wait_event(dy_ready)
            if mixed:
                mixed_wgrad(yt, saved_a, dw2, ks_cpu, ks)
            else:
                dw2.zero_()
                dg.k_grouped_bf16_gemm_tn_contiguous(yt, a, dw2, ks_cpu, ks, dw2)
        dgrad(yt, w2t, da, layout)
        dgu, dp = activation_backward(da, pre, prob, args.clamp)
        gu_ready.record(main)
        with torch.cuda.stream(worker):
            worker.wait_event(gu_ready)
            if mixed:
                mixed_wgrad(dgu, saved_x, dw1, ks_cpu, ks)
            else:
                dw1.zero_()
                dg.k_grouped_bf16_gemm_tn_contiguous(dgu, expert_x, dw1, ks_cpu, ks, dw1)
            # dgu is produced on main and can go out of Python scope while the
            # worker reads it. main joins the worker before issuing the next run.
            w_time.record(worker)
            join.record(worker)
        dgrad(dgu, w1t, dx, layout)
        dist.all_to_all_single(returned, dx.index_select(0, recv_rows), send_counts, recv_counts)
        dist.all_to_all_single(returned_p, dp.index_select(0, recv_rows), send_counts, recv_counts)
        outputs = combine(returned, returned_p, inverse)
        dx_time.record(main)
        main.wait_event(join)
        return outputs

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples = []
    for _ in range(args.repeats):
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values = torch.tensor([start.elapsed_time(ev) for ev in (end, dx_time, w_time)], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        samples.append(values.tolist())
    if args.check:
        # Independent per-expert reference, formed in original token/slot order.
        all_dy = gather(dy)
        ref_dx = torch.zeros((world * t * k, h), dtype=torch.bfloat16, device=device)
        ref_dp = torch.zeros_like(all_p)
        for j in range(local):
            q = (all_ids == rank * local + j).nonzero().flatten()
            xv, yv = all_x[q // k].float(), all_dy[q // k].float()
            if not len(q):
                assert not torch.count_nonzero(dw1[j]) and not torch.count_nonzero(dw2[j])
                continue
            forward = ref_fp8(xv) @ ref_fp4(w1[j]).T if mixed else xv @ w1[j].float().T
            gg, uu = forward.bfloat16().float().chunk(2, -1)
            gc, uc = gg.clamp(max=args.clamp), uu.clamp(-args.clamp, args.clamp)
            sig = gc.sigmoid()
            s = gc * sig * uc
            act = (s * all_p[q, None]).bfloat16().float()
            grad = (ref_fp8(yv) @ ref_fp4(w2[j].T).T if mixed else yv @ w2[j].float()).bfloat16().float()
            ds = grad * all_p[q, None]
            gu = torch.cat((ds * uc * (sig + gc * sig * (1 - sig)) * (gg < args.clamp), ds * gc * sig * (uu.abs() < args.clamp)), -1).bfloat16().float()
            ref_dx[q] = (ref_fp8(gu) @ ref_fp4(w1[j].T).T if mixed else gu @ w1[j].float()).bfloat16()
            if mixed:
                gu, xv, yv, act = (ref_fp8(z, token_axis=True) for z in (gu, xv, yv, act))
            ref_dp[q] = (grad * s).sum(-1)
            check(dw1[j], gu.T @ xv)
            check(dw2[j], yv.T @ act)
        dist.all_reduce(ref_dx)
        dist.all_reduce(ref_dp)
        check(outputs[0], ref_dx.view(world, t, k, h)[rank].float().sum(1).bfloat16())
        check(outputs[1], ref_dp.view(world, t, k)[rank])
    if rank == 0:
        ms = [statistics.median(column) for column in zip(*samples)]
        print(
            json.dumps(
                dict(
                    baseline=("DeepGEMM W4A8 dgrad + FP8 wgrad" if mixed else "DeepGEMM BF16") + " + NCCL all-to-all",
                    model=args.model,
                    tokens_per_rank=t,
                    hidden=h,
                    intermediate=i,
                    experts=e,
                    top_k=k,
                    repeats=args.repeats,
                    backward_ms=ms[0],
                    dx_ms=ms[1],
                    wgrad_ms=ms[2],
                    samples=samples,
                    checked=args.check,
                    padded_rows_rank0=extent,
                    active_rows_rank0=len(slots),
                )
            ),
            flush=True,
        )
    del graph
    torch.cuda.synchronize()


def check(actual, reference):
    relative = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-8)
    assert relative.item() < 0.01, relative.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=V4_PRESETS, help="Use the named DeepSeek-V4 routed expert configuration and synthetic score-based router")
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precision", choices=("bf16", "w4a8"), default="bf16")
    parser.add_argument("--check", action="store_true", help="Small-shape numerical check with distinct random expert weights")
    args = parser.parse_args()
    configure_model(args)
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    for t in args.tokens:
        measure(args, t, dist.get_rank(), device)
    dist.destroy_process_group()
