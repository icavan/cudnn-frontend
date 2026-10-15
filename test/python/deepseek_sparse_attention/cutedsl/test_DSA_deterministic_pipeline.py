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
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _inputs(*, ratio=4, coff=1, total_tokens=1024, sq=4, with_sink=True, short_topk_rows=False, seed=2301):
    torch.manual_seed(seed)
    heads = 64
    num_blocks = total_tokens // ratio
    compressor_kv = torch.randn(total_tokens, coff * 128, dtype=torch.bfloat16, device="cuda")
    compressor_score = torch.randn_like(compressor_kv)
    ape = torch.randn(ratio, coff * 128, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")
    cu_seqlens_comp = torch.tensor([0, num_blocks], dtype=torch.int32, device="cuda")
    index_q = torch.randn(1, sq, heads, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, sq, heads, dtype=torch.bfloat16, device="cuda")
    attn_q = torch.randn(sq, heads, 512, dtype=torch.bfloat16, device="cuda")
    attn_kv = torch.randn(total_tokens, 512, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(heads, dtype=torch.float32, device="cuda") if with_sink else None
    dout = torch.randn_like(attn_q)
    q_causal_offsets = torch.tensor([total_tokens - sq], dtype=torch.int32, device="cuda")
    if short_topk_rows:
        seq_lens = torch.linspace(32, 128, steps=sq, dtype=torch.float32, device="cuda").to(torch.int32)
    else:
        seq_lens = torch.full((sq,), num_blocks, dtype=torch.int32, device="cuda")
    return {
        "ratio": ratio,
        "coff": coff,
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
        "seq_lens": seq_lens,
    }


def _run_pipeline(data, *, deterministic_dk_planes=96, backend="default", check_topk=False):
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
        coff=data["coff"],
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

    topk = DSA.indexer_top_k_wrapper(
        scores.reshape(-1, num_blocks),
        data["seq_lens"],
        top_k=128,
        return_val=True,
        deterministic=True,
    )
    block_indices = topk["indices"].view(1, scores.shape[1], 128)
    tied_topk = None
    if check_topk:
        assert torch.all(block_indices >= 0)
        gathered_values = torch.gather(scores, -1, block_indices.long())
        assert torch.equal(topk["values"].view_as(gathered_values), gathered_values)

        # A same-shape side probe places 160 equal maxima across the K=128 cutoff.
        # Deterministic Top-K must choose and order the smallest source indices,
        # independent of scheduling.
        tied_scores = torch.zeros_like(scores)
        tied_scores[..., :160] = 1.0
        tied_topk = DSA.indexer_top_k_wrapper(
            tied_scores.reshape(-1, num_blocks),
            data["seq_lens"],
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
    attn_sink = data["attn_sink"].detach().clone().requires_grad_() if data["attn_sink"] is not None else None
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
    # Top-K already returns the selected logits in exactly the same slot order
    # as its indices. Re-gathering the dense scores would add an allocation and
    # a GPU kernel to every training step.
    predict = torch.softmax(topk["values"].view_as(block_indices), dim=-1)
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
        backend=backend,
        deterministic=True,
        deterministic_dk_planes=deterministic_dk_planes,
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
        coff=data["coff"],
        deterministic=True,
    )
    outputs = {
        "index_k": index_k,
        "index_scores": scores,
        "topk_indices": topk["indices"],
        "topk_values": topk["values"],
        "attention_out": attention["out"],
        "attention_lse": attention["lse"],
        "dq": attn_q.grad,
        "dkv": attn_kv.grad,
        "d_index_q": indexer_grads["d_index_q"],
        "d_weights": indexer_grads["d_weights"],
        "d_index_k": indexer_grads["d_index_k"],
        "d_compressor_kv": compressor_grads["grad_kv"],
        "d_compressor_score": compressor_grads["grad_score"],
        "d_ape": compressor_grads["grad_ape"],
    }
    if attn_sink is not None:
        outputs["d_sink"] = attn_sink.grad
    if tied_topk is not None:
        outputs["tied_topk_indices"] = tied_topk["indices"]
        outputs["tied_topk_values"] = tied_topk["values"]
    return outputs


def _clone_outputs(outputs):
    torch.cuda.synchronize()
    return {name: tensor.detach().clone() for name, tensor in outputs.items()}


def _assert_bitwise_equal(actual, expected, context=""):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.equal(actual[name], expected[name]), f"{name} changed{context}"


def test_csa_dsa_deterministic_training_pipeline_repeat_100():
    """Every floating output and gradient is bitwise stable for 100 executions."""
    data = _inputs()
    reference = _clone_outputs(_run_pipeline(data, check_topk=True))
    for repeat in range(1, 100):
        actual = _clone_outputs(_run_pipeline(data, check_topk=True))
        _assert_bitwise_equal(actual, reference, f" at repeat {repeat}")


@pytest.mark.parametrize(
    "ratio,coff,with_sink,backend,planes,short_topk_rows",
    [
        (2, 1, False, "default", 96, True),
        (4, 2, True, "default", 128, False),
        (4, 1, True, "sm100_v2", 148, False),
    ],
)
def test_csa_dsa_deterministic_training_pipeline_configuration_matrix(ratio, coff, with_sink, backend, planes, short_topk_rows):
    """Cross compressor modes, indexer backends/planes, no sink, and short Top-K rows."""
    data = _inputs(
        ratio=ratio,
        coff=coff,
        with_sink=with_sink,
        short_topk_rows=short_topk_rows,
        seed=2400 + ratio + coff + planes,
    )
    reference = _clone_outputs(_run_pipeline(data, deterministic_dk_planes=planes, backend=backend))
    if short_topk_rows:
        indices = reference["topk_indices"].view(data["seq_lens"].numel(), -1)
        values = reference["topk_values"].view_as(indices)
        positions = torch.arange(indices.shape[1], dtype=torch.int32, device="cuda").unsqueeze(0)
        active = positions < data["seq_lens"].clamp(max=indices.shape[1]).unsqueeze(1)
        assert torch.all((indices >= 0) & (indices < data["seq_lens"].unsqueeze(1)) | ~active)
        assert torch.all(indices[~active] == -1)
        assert torch.all(torch.isneginf(values[~active]))
    for repeat in range(1, 3):
        actual = _clone_outputs(_run_pipeline(data, deterministic_dk_planes=planes, backend=backend))
        _assert_bitwise_equal(actual, reference, f" at repeat {repeat}")


def test_csa_dsa_deterministic_training_pipeline_cuda_graph_changed_inputs():
    """The complete allocating composition remains replayable and reads changed source tensors."""
    data = _inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run_pipeline(data)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                captured = _run_pipeline(data)
            with torch.no_grad():
                data["compressor_kv"].mul_(0.75)
                data["compressor_score"].mul_(1.25)
                data["index_q"].mul_(0.5)
                data["attn_q"].mul_(1.5)
                data["attn_kv"].mul_(0.8)
            graph.replay()
            actual = {name: tensor.detach().clone() for name, tensor in captured.items()}
            expected = _run_pipeline(data)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0, msg=lambda message, name=name: f"{name}: {message}")
        finally:
            graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
