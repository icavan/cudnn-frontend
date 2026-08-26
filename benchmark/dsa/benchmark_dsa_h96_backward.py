#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused SM100 H96/D576 DSA backward benchmark.

The synthetic forward tensors avoid materializing a dense ``S_q x S_kv``
random matrix or running the slow PyTorch reference forward.  Inputs preserve
the production dtypes, shapes, fixed top-k contract, and numerically benign
softmax range; both candidate and baseline kernels receive identical tensors.
"""

import argparse
import math

import torch

from cudnn import DSA


def make_inputs(
    seqlen_q: int,
    seqlen_kv: int,
    topk: int,
    heads: int,
    seed: int,
    pair_overlap: float | None,
):
    torch.manual_seed(seed)
    device = "cuda"
    head_dim, head_dim_v = 576, 512
    dtype = torch.bfloat16

    q = torch.randn((seqlen_q, heads, head_dim), device=device, dtype=dtype) * 0.1
    kv = torch.randn((seqlen_kv, head_dim), device=device, dtype=dtype) * 0.1
    out = torch.randn((seqlen_q, heads, head_dim_v), device=device, dtype=dtype) * 0.1
    dout = torch.randn_like(out) * 0.1
    lse = torch.full((seqlen_q, heads), math.log(topk), device=device, dtype=torch.float32)
    attn_sink = torch.zeros((heads,), device=device, dtype=torch.float32)

    # Unique cyclic indices per query, generated in O(S_q * topk) memory.
    columns = torch.arange(topk, device=device, dtype=torch.int32)
    if pair_overlap is None:
        offsets = (torch.arange(seqlen_q, device=device, dtype=torch.int32) * 997) % seqlen_kv
    else:
        if not 0.0 <= pair_overlap <= 1.0:
            raise ValueError("pair_overlap must be in [0, 1]")
        query_ids = torch.arange(seqlen_q, device=device, dtype=torch.int32)
        pair_ids = query_ids // 2
        within_pair = query_ids % 2
        shift = int(round(topk * (1.0 - pair_overlap)))
        offsets = (pair_ids * 4099 + within_pair * shift) % seqlen_kv
    topk_idxs = (columns.unsqueeze(0) + offsets.unsqueeze(1)) % seqlen_kv
    topk_length = torch.full((seqlen_q,), topk, device=device, dtype=torch.int32)

    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    return q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, dq, dkv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen-q", type=int, default=16384)
    parser.add_argument("--seqlen-kv", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--pair-overlap", type=float, default=None)
    parser.add_argument("--check-pair", action="store_true")
    parser.add_argument("--pair-h96", action="store_true")
    parser.add_argument("--with-topk-length", action="store_true")
    args = parser.parse_args()

    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, dq, dkv = make_inputs(
        args.seqlen_q,
        args.seqlen_kv,
        args.topk,
        args.heads,
        args.seed,
        args.pair_overlap,
    )
    softmax_scale = 576**-0.5

    if args.check_pair:
        outputs = []
        for enable_pair in (False, True):
            dq_check = torch.empty_like(q)
            dkv_check = torch.empty_like(kv)
            pair_kwargs = {"pair_queries": True} if enable_pair else {}
            dq_out, dkv_out, dsink_out = DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length if args.with_topk_length else None,
                dq=dq_check,
                dkv=dkv_check,
                **pair_kwargs,
            )
            torch.cuda.synchronize()
            outputs.append((dq_out.clone(), dkv_out.clone(), dsink_out.clone()))

        names = ("dQ", "dKV", "dSink")
        for name, baseline, paired in zip(names, outputs[0], outputs[1]):
            baseline_f = baseline.float()
            paired_f = paired.float()
            abs_diff = (baseline_f - paired_f).abs()
            bitwise = (baseline.view(torch.uint8) == paired.view(torch.uint8)).float().mean().item()
            print(
                f"{name}: bitwise_bytes={bitwise:.6f} max_abs={abs_diff.max().item():.6e} "
                f"mean_abs={abs_diff.mean().item():.6e} ref_max={baseline_f.abs().max().item():.6e}"
            )
        return

    def run():
        pair_kwargs = {"pair_queries": True} if args.pair_h96 else {}
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length if args.with_topk_length else None,
            dq=dq,
            dkv=dkv,
            **pair_kwargs,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("dsa_h96_bwd_16k"):
            run()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        run()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / args.repeat
    flops = 2 * args.seqlen_q * args.heads * args.topk * (3 * 576 + 2 * 512)
    tflops = flops / (elapsed_ms * 1.0e-3) / 1.0e12
    print(
        f"H{args.heads} D576/Dv512 BF16: Sq={args.seqlen_q} Skv={args.seqlen_kv} "
        f"topk={args.topk} pair_overlap={args.pair_overlap} "
        f"time_ms={elapsed_ms:.4f} tflops={tflops:.2f}"
    )


if __name__ == "__main__":
    main()
