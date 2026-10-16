# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect an NVTX-annotated CSA training iteration for Nsight Systems."""

from __future__ import annotations

import argparse
import contextlib
import gc

import torch

from cudnn import DSA
import cudnn.csa.training as training


@contextlib.contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _wrap(name, fn):
    def wrapped(*args, **kwargs):
        with nvtx(name):
            return fn(*args, **kwargs)

    return wrapped


class _TaggedLayerNorm(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        with nvtx("CSA::k_norm_forward"):
            return self.inner(value)


def _install_stage_ranges():
    DSA.indexer_forward_wrapper = _wrap("CSA::indexer_forward", DSA.indexer_forward_wrapper)
    DSA.indexer_top_k_wrapper = _wrap("CSA::topk", DSA.indexer_top_k_wrapper)
    DSA.sparse_attention = _wrap("CSA::sparse_attention_forward", DSA.sparse_attention)
    DSA.sparse_attention_score_recompute = _wrap("CSA::score_recompute", DSA.sparse_attention_score_recompute)
    DSA.indexer_backward_wrapper = _wrap("CSA::indexer_backward", DSA.indexer_backward_wrapper)

    import cudnn.csa.compressor as compressor
    import cudnn.deepseek_sparse_attention.sparse_attention_backward as attention_backward

    compressor.csa_compressor_forward_wrapper = _wrap("CSA::compressor_forward", compressor.csa_compressor_forward_wrapper)
    compressor.csa_compressor_backward_wrapper = _wrap("CSA::compressor_backward", compressor.csa_compressor_backward_wrapper)
    attention_backward.sparse_attention_backward_wrapper = _wrap(
        "CSA::sparse_attention_backward",
        attention_backward.sparse_attention_backward_wrapper,
    )


def _make_inputs(sq, tokens, indexer_heads, attention_heads, dim, ratio):
    torch.manual_seed(1729)
    blocks = tokens // ratio
    index_q = torch.randn(
        1,
        sq,
        indexer_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    if ratio == 1:
        index_k = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        compressor_score = ape = cu_seqlens = cu_seqlens_comp = None
    else:
        index_k = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        compressor_score = torch.randn_like(index_k, requires_grad=True)
        ape = torch.randn(ratio, 128, device="cuda", dtype=torch.float32, requires_grad=True)
        cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        cu_seqlens_comp = torch.tensor([0, blocks], device="cuda", dtype=torch.int32)
    weights = torch.randn(
        1,
        sq,
        indexer_heads,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    attention_q = torch.randn(
        sq,
        attention_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    attention_kv = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attention_sink = torch.randn(attention_heads, device="cuda", dtype=torch.float32, requires_grad=True)
    seq_lens = torch.full((sq,), blocks, device="cuda", dtype=torch.int32)
    q_causal_offsets = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    return {
        "index_q": index_q,
        "index_k": index_k,
        "weights": weights,
        "attention_q": attention_q,
        "attention_kv": attention_kv,
        "seq_lens": seq_lens,
        "attention_sink": attention_sink,
        "compressor_score": compressor_score,
        "ape": ape,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_comp": cu_seqlens_comp,
        "q_causal_offsets": q_causal_offsets,
    }


def _grad_inputs(module, data):
    tensors = [
        data["index_q"],
        data["index_k"],
        data["weights"],
        data["attention_q"],
        data["attention_kv"],
        data["attention_sink"],
    ]
    if data["compressor_score"] is not None:
        tensors.extend((data["compressor_score"], data["ape"]))
    tensors.extend((module.k_norm.inner.weight, module.k_norm.inner.bias))
    return tensors


def _run(module, data, label, *, synchronize=False):
    with nvtx(label):
        result = module(
            data["index_q"],
            data["index_k"],
            data["weights"],
            data["attention_q"],
            data["attention_kv"],
            data["seq_lens"],
            data["attention_sink"],
            compressor_score=data["compressor_score"],
            ape=data["ape"],
            cu_seqlens=data["cu_seqlens"],
            cu_seqlens_comp=data["cu_seqlens_comp"],
            q_causal_offsets=data["q_causal_offsets"],
        )
        loss = result["out"].float().square().mean() + result["indexer_loss"]
        grads = torch.autograd.grad(loss, _grad_inputs(module, data))
        if synchronize:
            torch.cuda.synchronize()
    return result, grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--indexer-heads", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--token-topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=5)
    args = parser.parse_args()
    if args.token_topk % args.ratio:
        raise ValueError("token-topk must be divisible by ratio")
    block_topk = args.token_topk // args.ratio

    _install_stage_ranges()
    data = _make_inputs(
        args.sq,
        args.tokens,
        args.indexer_heads,
        args.attention_heads,
        args.dim,
        args.ratio,
    )
    common = dict(
        compress_ratio=args.ratio,
        top_k=args.token_topk,
        deterministic=True,
        deterministic_dk_planes=96,
        device="cuda",
    )
    module = training.CompressedSparseAttention(**common)
    module.k_norm = _TaggedLayerNorm(module.k_norm)

    for _ in range(args.warmup):
        _run(module, data, "warmup_nope")
    torch.cuda.synchronize()
    gc.collect()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.profile_iters):
        _run(module, data, "CSA::iteration_nope", synchronize=True)
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"profiled sq={args.sq} tokens={args.tokens} indexer_h={args.indexer_heads} "
        f"attention_h={args.attention_heads} dim={args.dim} ratio={args.ratio} "
        f"token_topk={args.token_topk} block_topk={block_topk} position_encoding=nope"
    )


if __name__ == "__main__":
    main()
