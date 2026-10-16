#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile the deterministic CSA/DSA training pipeline end to end.

The eager measurement intentionally drives the public allocating wrappers. The
CUDA Graph measurement captures that same composition after warmup. CUDA-event
device latency excludes compilation, correctness checks, and profiler collection.
"""

import argparse
import contextlib
import json
import statistics

import torch
from cudnn import CSA, DSA


class Pipeline:
    """One fixed-shape CSA compressor -> DSA training step."""

    def __init__(self, args):
        if args.tokens % args.ratio:
            raise ValueError("tokens must be divisible by ratio")
        self.ratio = args.ratio
        self.coff = args.coff
        self.tokens = args.tokens
        self.sq = args.sq
        self.topk = args.topk
        self.num_blocks = args.tokens // args.ratio
        if self.num_blocks < args.topk:
            raise ValueError("tokens / ratio must be at least topk")

        torch.manual_seed(args.seed)
        device = torch.device("cuda")
        width = args.coff * 128
        self.compressor_kv = torch.randn(args.tokens, width, dtype=torch.bfloat16, device=device)
        self.compressor_score = torch.randn_like(self.compressor_kv)
        self.ape = torch.randn(args.ratio, width, dtype=torch.float32, device=device)
        self.cu_seqlens = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
        self.cu_seqlens_comp = torch.tensor([0, self.num_blocks], dtype=torch.int32, device=device)
        self.index_q = torch.randn(1, args.sq, 64, 128, dtype=torch.bfloat16, device=device)
        self.weights = torch.randn(1, args.sq, 64, dtype=torch.bfloat16, device=device)
        self.attn_q = torch.randn(args.sq, 64, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
        self.attn_kv = torch.randn(args.tokens, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
        self.attn_sink = torch.randn(64, dtype=torch.float32, device=device, requires_grad=True)
        self.dout = torch.randn_like(self.attn_q)
        self.q_causal_offsets = torch.tensor([args.tokens - args.sq], dtype=torch.int32, device=device)
        self.seq_lens = torch.full((args.sq,), self.num_blocks, dtype=torch.int32, device=device)
        self.token_offsets = torch.arange(args.ratio, dtype=torch.int32, device=device).view(1, 1, args.ratio)
        self.grad_loss = torch.ones((), dtype=torch.float32, device=device)
        self.backend = args.backend
        self.planes = args.planes
        self.deterministic = args.deterministic

    @staticmethod
    def _range(name, enabled):
        return torch.profiler.record_function(name) if enabled else contextlib.nullcontext()

    def run(self, *, profile_ranges=False):
        """Enqueue one complete training step and return live outputs."""
        with self._range("pipeline::compressor_forward", profile_ranges):
            index_k = CSA.csa_compressor_forward_wrapper(
                self.compressor_kv,
                self.compressor_score,
                self.ape,
                self.cu_seqlens,
                self.cu_seqlens_comp,
                ratio=self.ratio,
                head_dim=128,
                coff=self.coff,
                total_comp=self.num_blocks,
            )["out"].unsqueeze(0)

        with self._range("pipeline::indexer_forward", profile_ranges):
            scores = DSA.indexer_forward_wrapper(
                self.index_q,
                index_k.unsqueeze(2),
                self.weights,
                ratio=self.ratio,
                sm_scale=128**-0.5,
                q_causal_offsets=self.q_causal_offsets,
            )["scores"]

        with self._range("pipeline::topk", profile_ranges):
            topk = DSA.indexer_top_k_wrapper(
                scores.reshape(-1, self.num_blocks),
                self.seq_lens,
                top_k=self.topk,
                return_val=True,
                deterministic=self.deterministic,
            )
            block_indices = topk["indices"].view(1, self.sq, self.topk)

        with self._range("pipeline::expand_block_indices", profile_ranges):
            token_indices = (block_indices.squeeze(0).unsqueeze(-1) * self.ratio + self.token_offsets).reshape(self.sq, -1)

        with self._range("pipeline::sparse_attention_forward_backward", profile_ranges):
            attention = DSA.sparse_attention(
                self.attn_q,
                self.attn_kv,
                token_indices,
                self.attn_sink,
                deterministic=self.deterministic,
                trusted_compact_metadata=True,
            )
            dq, dkv, d_sink = torch.autograd.grad(
                attention["out"],
                (self.attn_q, self.attn_kv, self.attn_sink),
                self.dout,
            )

        with self._range("pipeline::score_recompute", profile_ranges):
            target = (
                DSA.sparse_attention_score_recompute(
                    self.attn_q.detach(),
                    self.attn_kv.detach(),
                    attention["lse"],
                    token_indices,
                )["target"]
                .view(self.sq, self.topk, self.ratio)
                .sum(-1)
                .unsqueeze(0)
            )
            predict = torch.softmax(topk["values"].view(1, self.sq, self.topk), dim=-1)

        with self._range("pipeline::indexer_backward", profile_ranges):
            indexer_grads = DSA.indexer_backward_wrapper(
                self.index_q,
                self.weights,
                index_k,
                target.contiguous(),
                predict.contiguous(),
                block_indices,
                self.grad_loss,
                sm_scale=128**-0.5,
                loss_coeff=float(self.sq),
                backend=self.backend,
                deterministic=self.deterministic,
                deterministic_dk_planes=self.planes,
            )

        with self._range("pipeline::compressor_backward", profile_ranges):
            compressor_grads = CSA.csa_compressor_backward_wrapper(
                self.compressor_kv,
                self.compressor_score,
                self.ape,
                self.cu_seqlens,
                self.cu_seqlens_comp,
                indexer_grads["d_index_k"].squeeze(0),
                ratio=self.ratio,
                head_dim=128,
                coff=self.coff,
                deterministic=self.deterministic,
            )

        return {
            "topk_indices": topk["indices"],
            "attention_out": attention["out"],
            "dq": dq,
            "dkv": dkv,
            "d_sink": d_sink,
            "d_index_q": indexer_grads["d_index_q"],
            "d_index_k": indexer_grads["d_index_k"],
            "d_ape": compressor_grads["grad_ape"],
        }


def _device_us(call, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _peak_memory_mib(call):
    """Incremental allocated-memory high-water mark for one synchronized call."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del outputs
    return (peak - baseline) / (1024 * 1024)


def _profile(pipeline, iterations, trace):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=True,
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        for _ in range(iterations):
            pipeline.run(profile_ranges=True)
        torch.cuda.synchronize()
    if trace:
        profiler.export_chrome_trace(trace)
    return profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30), profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384, help="Local uncompressed tokens; 16384 models 1M / CP64.")
    parser.add_argument("--sq", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512, help="Compressed blocks; sparse attention consumes topk * ratio tokens.")
    parser.add_argument("--ratio", type=int, choices=(2, 4), default=4)
    parser.add_argument("--coff", type=int, choices=(1, 2), default=1)
    parser.add_argument("--backend", choices=("default", "sm100_v2"), default="default")
    parser.add_argument("--planes", type=int, choices=(96, 128, 148), default=96)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--trace", help="Optional Chrome trace output path.")
    parser.add_argument("--seed", type=int, default=2301)
    args = parser.parse_args()

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill_uninitialized = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    # Measure the selected algorithms, not PyTorch's debug fill kernels for
    # wrapper-owned torch.empty outputs and workspaces.
    torch.utils.deterministic.fill_uninitialized_memory = False
    graph = None
    capture_stream = None
    try:
        pipeline = Pipeline(args)
        for _ in range(args.warmup):
            pipeline.run()
        torch.cuda.synchronize()

        eager_device = _device_us(pipeline.run, args.iterations)
        eager_peak_memory = _peak_memory_mib(pipeline.run)
        cpu_table, cuda_table = _profile(pipeline, args.profile_iterations, args.trace)

        # Independent leaves and one explicit stream keep AccumulateGrad from
        # inheriting the eager stream before graph capture.
        graph_pipeline = Pipeline(args)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(args.warmup):
                graph_pipeline.run()
            capture_stream.synchronize()
            torch.cuda.empty_cache()
            graph_baseline_allocated = torch.cuda.memory_allocated()
            graph_baseline_reserved = torch.cuda.memory_reserved()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = graph_pipeline.run()
            graph.replay()
            capture_stream.synchronize()
            graph_retained_allocated = (torch.cuda.memory_allocated() - graph_baseline_allocated) / (1024 * 1024)
            graph_retained_reserved = (torch.cuda.memory_reserved() - graph_baseline_reserved) / (1024 * 1024)
            graph_device = _device_us(graph.replay, args.iterations)

            with torch.no_grad():
                graph_pipeline.compressor_kv.mul_(0.75)
                graph_pipeline.index_q.mul_(0.75)
                graph_pipeline.attn_q.mul_(1.25)
                graph_pipeline.attn_kv.mul_(0.8)
            graph.replay()
            replayed = {name: tensor.detach().clone() for name, tensor in captured.items()}
            expected = graph_pipeline.run()
            for name in replayed:
                if name == "topk_indices" and not args.deterministic:
                    actual_tensor = replayed[name].sort(dim=-1).values
                    expected_tensor = expected[name].sort(dim=-1).values
                    atol = rtol = 0
                else:
                    actual_tensor, expected_tensor = replayed[name], expected[name]
                    atol = rtol = 0 if args.deterministic else 2e-2
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    atol=atol,
                    rtol=rtol,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
        torch.cuda.current_stream().wait_stream(capture_stream)

        result = {
            "shape": {
                "tokens": args.tokens,
                "sq": args.sq,
                "topk_blocks": args.topk,
                "topk_tokens": args.topk * args.ratio,
                "ratio": args.ratio,
                "coff": args.coff,
                "backend": args.backend,
                "planes": args.planes,
                "deterministic": args.deterministic,
            },
            "eager_device_us": eager_device,
            "graph_device_us": graph_device,
            "eager_peak_memory_mib": eager_peak_memory,
            "graph_capture_retained_allocated_mib": graph_retained_allocated,
            "graph_capture_retained_reserved_mib": graph_retained_reserved,
            "device_speedup": eager_device / graph_device,
        }
        print(json.dumps(result, indent=2))
        print("\nCPU profile (warm eager public-wrapper path):\n" + cpu_table)
        print("\nCUDA profile (warm eager public-wrapper path):\n" + cuda_table)
    finally:
        if graph is not None:
            graph.reset()
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill_uninitialized
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


if __name__ == "__main__":
    main()
