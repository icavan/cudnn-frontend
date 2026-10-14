# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end deterministic CSA/DSA training composition coverage."""

import pytest
import torch

from cudnn import CSA, DSA

pytestmark = pytest.mark.L1


@pytest.fixture(autouse=True)
def supported():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("deterministic CSA/DSA pipeline requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("deterministic CSA/DSA pipeline test")
    if error:
        pytest.skip(error)


def _inputs():
    torch.manual_seed(2301)
    ratio, total_tokens, sq, heads = 4, 1024, 4, 64
    num_blocks = total_tokens // ratio
    compressor_kv = torch.randn(total_tokens, 128, dtype=torch.bfloat16, device="cuda")
    compressor_score = torch.randn_like(compressor_kv)
    ape = torch.randn(ratio, 128, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")
    cu_seqlens_comp = torch.tensor([0, num_blocks], dtype=torch.int32, device="cuda")
    index_q = torch.randn(1, sq, heads, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, sq, heads, dtype=torch.bfloat16, device="cuda")
    attn_q = torch.randn(sq, heads, 512, dtype=torch.bfloat16, device="cuda")
    attn_kv = torch.randn(total_tokens, 512, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(heads, dtype=torch.float32, device="cuda")
    dout = torch.randn_like(attn_q)
    q_causal_offsets = torch.tensor([total_tokens - sq], dtype=torch.int32, device="cuda")
    return {
        "ratio": ratio,
        "num_blocks": num_blocks,
        "compressor_kv": compressor_kv,
        "compressor_score": compressor_score,
        "ape": ape,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_comp": cu_seqlens_comp,
        "index_q": index_q,
        "weights": weights,
        "attn_q": attn_q,
        "attn_kv": attn_kv,
        "attn_sink": attn_sink,
        "dout": dout,
        "q_causal_offsets": q_causal_offsets,
    }


def _run_pipeline(data):
    ratio = data["ratio"]
    num_blocks = data["num_blocks"]
    index_k = CSA.csa_compressor_forward_wrapper(
        data["compressor_kv"],
        data["compressor_score"],
        data["ape"],
        data["cu_seqlens"],
        data["cu_seqlens_comp"],
        ratio=ratio,
        head_dim=128,
        coff=1,
        total_comp=num_blocks,
    )["out"].unsqueeze(0)
    scores = DSA.indexer_forward_wrapper(
        data["index_q"],
        index_k.unsqueeze(2),
        data["weights"],
        ratio=ratio,
        sm_scale=128**-0.5,
        q_causal_offsets=data["q_causal_offsets"],
    )["scores"]

    seq_lens = torch.full((scores.shape[1],), num_blocks, dtype=torch.int32, device="cuda")
    topk = DSA.indexer_top_k_wrapper(
        scores.reshape(-1, num_blocks),
        seq_lens,
        top_k=128,
        return_val=True,
        deterministic=True,
    )
    block_indices = topk["indices"].view(1, scores.shape[1], 128)
    assert torch.all(block_indices >= 0)

    # A same-shape side probe places 160 equal maxima across the K=128 cutoff.
    # Deterministic Top-K must choose and order the smallest source indices,
    # independent of scheduling.
    tied_scores = torch.zeros_like(scores)
    tied_scores[..., :160] = 1.0
    tied_topk = DSA.indexer_top_k_wrapper(
        tied_scores.reshape(-1, num_blocks),
        seq_lens,
        top_k=128,
        return_val=True,
        deterministic=True,
    )
    tied_indices = tied_topk["indices"].view_as(block_indices)
    expected_indices = torch.arange(128, dtype=torch.int32, device="cuda").view(1, 1, 128).expand_as(tied_indices)
    assert torch.equal(tied_indices, expected_indices)

    token_indices = (block_indices.squeeze(0).unsqueeze(-1) * ratio + torch.arange(ratio, dtype=torch.int32, device="cuda").view(1, 1, ratio)).reshape(
        scores.shape[1], -1
    )
    attn_q = data["attn_q"].detach().clone().requires_grad_()
    attn_kv = data["attn_kv"].detach().clone().requires_grad_()
    attn_sink = data["attn_sink"].detach().clone().requires_grad_()
    attention = DSA.sparse_attention(
        attn_q,
        attn_kv,
        token_indices,
        attn_sink,
        deterministic=True,
        trusted_compact_metadata=True,
    )
    attention["out"].backward(data["dout"])

    target = (
        DSA.sparse_attention_score_recompute(
            attn_q.detach(),
            attn_kv.detach(),
            attention["lse"],
            token_indices,
        )["target"]
        .view(scores.shape[1], 128, ratio)
        .sum(-1)
        .unsqueeze(0)
    )
    selected_logits = torch.gather(scores, -1, block_indices.long())
    predict = torch.softmax(selected_logits, dim=-1)
    indexer_grads = DSA.indexer_backward_wrapper(
        data["index_q"],
        data["weights"],
        index_k,
        target.contiguous(),
        predict.contiguous(),
        block_indices,
        torch.ones((), dtype=torch.float32, device="cuda"),
        sm_scale=128**-0.5,
        loss_coeff=float(scores.shape[1]),
        deterministic=True,
        deterministic_dk_planes=96,
    )
    compressor_grads = CSA.csa_compressor_backward_wrapper(
        data["compressor_kv"],
        data["compressor_score"],
        data["ape"],
        data["cu_seqlens"],
        data["cu_seqlens_comp"],
        indexer_grads["d_index_k"].squeeze(0),
        ratio=ratio,
        head_dim=128,
        coff=1,
        deterministic=True,
    )
    torch.cuda.synchronize()
    return tuple(
        tensor.detach().clone()
        for tensor in (
            index_k,
            scores,
            topk["indices"],
            topk["values"],
            tied_topk["indices"],
            tied_topk["values"],
            attention["out"],
            attention["lse"],
            attn_q.grad,
            attn_kv.grad,
            attn_sink.grad,
            indexer_grads["d_index_q"],
            indexer_grads["d_weights"],
            indexer_grads["d_index_k"],
            compressor_grads["grad_kv"],
            compressor_grads["grad_score"],
            compressor_grads["grad_ape"],
        )
    )


def test_csa_dsa_deterministic_training_pipeline_repeat_100():
    """Every floating output and gradient is bitwise stable for 100 executions."""
    data = _inputs()
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        reference = _run_pipeline(data)
        for repeat in range(1, 100):
            actual = _run_pipeline(data)
            for name, got, expected in zip(
                (
                    "index_k",
                    "index_scores",
                    "topk_indices",
                    "topk_values",
                    "tied_topk_indices",
                    "tied_topk_values",
                    "attention_out",
                    "attention_lse",
                    "dq",
                    "dkv",
                    "d_sink",
                    "d_index_q",
                    "d_weights",
                    "d_index_k",
                    "d_compressor_kv",
                    "d_compressor_score",
                    "d_ape",
                ),
                actual,
                reference,
            ):
                assert torch.equal(got, expected), f"{name} changed at repeat {repeat}"
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
