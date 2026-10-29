#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare non-deterministic and deterministic CSA training GPU latency.

This benchmark drives :class:`cudnn.CompressedSparseAttention`, including key
LayerNorm, compressor, indexer, Top-K, sparse attention, score recompute, and
all first-order backward paths. CUDA events measure device latency; compilation,
warmup, graph capture, correctness checks, and host enqueue are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import warnings

import torch
from cudnn import CompressedSparseAttention


def _inputs(args):
    torch.manual_seed(args.seed)
    blocks = args.tokens // args.ratio
    index_q = torch.randn(1, args.sq, 64, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    index_k = torch.randn(args.tokens, args.coff * 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    compressor_score = torch.randn_like(index_k, requires_grad=True)
    ape = torch.randn(args.ratio, args.coff * 128, dtype=torch.float32, device="cuda", requires_grad=True)
    weights = torch.randn(1, args.sq, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_q = torch.randn(args.sq, args.heads, args.dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_kv = torch.randn(args.tokens, args.dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_sink = torch.randn(args.heads, dtype=torch.float32, device="cuda", requires_grad=True)
    return {
        "index_q": index_q,
        "index_k": index_k,
        "weights": weights,
        "attention_q": attention_q,
        "attention_kv": attention_kv,
        "seq_lens": torch.full((args.sq,), blocks, dtype=torch.int32, device="cuda"),
        "attention_sink": attention_sink,
        "compressor_score": compressor_score,
        "ape": ape,
        "cu_seqlens": torch.tensor([0, args.tokens], dtype=torch.int32, device="cuda"),
        "cu_seqlens_comp": torch.tensor([0, blocks], dtype=torch.int32, device="cuda"),
        "q_causal_offsets": torch.tensor([args.tokens], dtype=torch.int32, device="cuda"),
    }


class Case:
    """One fixed-address public-module training step."""

    def __init__(self, args, *, deterministic, planes):
        self.data = _inputs(args)
        self.module = CompressedSparseAttention(
            compress_ratio=args.ratio,
            coff=args.coff,
            top_k=args.top_k,
            k_norm_placement=args.k_norm_placement,
            deterministic=deterministic,
            deterministic_dk_planes=planes,
            indexer_backend=args.backend,
            device="cuda",
        )
        self.grad_inputs = [
            self.data[name]
            for name in (
                "index_q",
                "index_k",
                "weights",
                "attention_q",
                "attention_kv",
                "attention_sink",
                "compressor_score",
                "ape",
            )
        ]
        self.grad_inputs.extend((self.module.k_norm.weight, self.module.k_norm.bias))

    def run(self):
        result = self.module(
            self.data["index_q"],
            self.data["index_k"],
            self.data["weights"],
            self.data["attention_q"],
            self.data["attention_kv"],
            self.data["seq_lens"],
            self.data["attention_sink"],
            compressor_score=self.data["compressor_score"],
            ape=self.data["ape"],
            cu_seqlens=self.data["cu_seqlens"],
            cu_seqlens_comp=self.data["cu_seqlens_comp"],
            q_causal_offsets=self.data["q_causal_offsets"],
        )
        loss = result["out"].float().square().mean() + result["indexer_loss"]
        grads = torch.autograd.grad(loss, self.grad_inputs)
        return result, grads


def _event_median_ms(call, samples):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        outputs = call()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        del outputs
    return statistics.median(timings)


def _eager_peak_mib(call):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del outputs
    return (peak - baseline) / (1024 * 1024)


def _capture(case, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            case.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The AccumulateGrad node's stream does not match the stream of the node that produced the incoming gradient.*",
                category=UserWarning,
            )
            with torch.cuda.graph(graph, stream=stream):
                captured = case.run()
        graph.replay()
        stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, captured


def _measure(args, *, deterministic, planes):
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    case = Case(args, deterministic=deterministic, planes=planes)
    for _ in range(args.warmup):
        case.run()
    torch.cuda.synchronize()

    eager_peak_mib = _eager_peak_mib(case.run)
    eager_ms = _event_median_ms(case.run, args.samples)
    graph, captured = _capture(case, args.warmup)
    try:
        graph_ms = _event_median_ms(graph.replay, args.samples)
    finally:
        graph.reset()
        del captured, graph, case
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {
        "policy": "deterministic" if deterministic else "non-deterministic",
        "planes": planes if deterministic else None,
        "indexer_dk_scratch_mib": (planes if deterministic else 1) * (args.tokens // args.ratio) * 128 * 4 / (1024 * 1024),
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "eager_peak_mib": eager_peak_mib,
    }


def _validate_args(args):
    if args.tokens <= 0 or args.tokens % args.ratio:
        raise ValueError("tokens must be positive and divisible by ratio")
    if args.sq <= 0:
        raise ValueError("sq must be positive")
    if args.top_k <= 0 or args.top_k > 2048 or args.top_k % (128 * args.ratio):
        raise ValueError("top-k must be a multiple of 128 * ratio and at most 2048")
    if args.tokens // args.ratio < args.top_k // args.ratio:
        raise ValueError("compressed K length must be at least top-k / ratio")


def _print_table(rows):
    baseline = rows[0]
    print("| policy | dK planes | dK scratch (MiB) | eager GPU (ms) | eager slowdown | graph GPU (ms) | graph slowdown | eager peak (MiB) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        planes = "-" if row["planes"] is None else str(row["planes"])
        print(
            f"| {row['policy']} | {planes} | {row['indexer_dk_scratch_mib']:.1f} | {row['eager_ms']:.3f} | "
            f"{row['eager_ms'] / baseline['eager_ms']:.2f}x | {row['graph_ms']:.3f} | "
            f"{row['graph_ms'] / baseline['graph_ms']:.2f}x | {row['eager_peak_mib']:.1f} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384, help="Local KV tokens; 16384 models 1M / CP64.")
    parser.add_argument("--sq", type=int, default=8192)
    parser.add_argument("--heads", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--dim", type=int, choices=(512, 576), default=512)
    parser.add_argument("--ratio", type=int, choices=(2, 4), default=4)
    parser.add_argument("--coff", type=int, choices=(1, 2), default=1)
    parser.add_argument("--top-k", type=int, default=2048, help="Selected token count, not compressed blocks.")
    parser.add_argument("--k-norm-placement", choices=("before_pool", "after_pool"), default="after_pool")
    parser.add_argument("--backend", choices=("default", "sm100_v2"), default="default")
    parser.add_argument("--planes", type=int, nargs="+", choices=(96, 128, 148), default=(96, 128, 148))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--construction-order", choices=("non-deterministic-first", "deterministic-first"), default="non-deterministic-first")
    args = parser.parse_args()
    _validate_args(args)

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill_uninitialized = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        requests = [(False, 96), *((True, planes) for planes in args.planes)]
        if args.construction_order == "deterministic-first":
            requests.reverse()
        rows = [_measure(args, deterministic=deterministic, planes=planes) for deterministic, planes in requests]
        rows.sort(key=lambda row: (row["policy"] == "deterministic", row["planes"] or 0))
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill_uninitialized
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)

    print(
        f"shape: S_q={args.sq}, local_KV={args.tokens}, H={args.heads}, D={args.dim}, " f"ratio={args.ratio}, token_top_k={args.top_k}, backend={args.backend}"
    )
    _print_table(rows)
    print(json.dumps({"shape": vars(args), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
