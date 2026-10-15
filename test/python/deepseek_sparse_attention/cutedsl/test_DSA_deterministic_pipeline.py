# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end deterministic CSA/DSA training composition coverage."""

import pytest
import torch

from cudnn import CSA, DSA
from test_utils import bitwise_bits


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


def _inputs(
    *,
    ratio=4,
    coff=1,
    total_tokens=1024,
    sq=4,
    topk=128,
    attention_heads=64,
    with_sink=True,
    seq_lens=None,
    seed=2301,
):
    torch.manual_seed(seed)
    indexer_heads = 64
    num_blocks = total_tokens // ratio
    if num_blocks < topk:
        raise ValueError(f"test setup requires num_blocks >= topk, got {num_blocks} < {topk}")
    compressor_kv = torch.randn(total_tokens, coff * 128, dtype=torch.bfloat16, device="cuda")
    compressor_score = torch.randn_like(compressor_kv)
    ape = torch.randn(ratio, coff * 128, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")
    cu_seqlens_comp = torch.tensor([0, num_blocks], dtype=torch.int32, device="cuda")
    index_q = torch.randn(1, sq, indexer_heads, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, sq, indexer_heads, dtype=torch.bfloat16, device="cuda")
    attn_q = torch.randn(sq, attention_heads, 512, dtype=torch.bfloat16, device="cuda")
    attn_kv = torch.randn(total_tokens, 512, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(attention_heads, dtype=torch.float32, device="cuda") if with_sink else None
    dout = torch.randn_like(attn_q)
    q_causal_offsets = torch.tensor([total_tokens - sq], dtype=torch.int32, device="cuda")
    seq_lens = torch.full((sq,), num_blocks, dtype=torch.int32, device="cuda") if seq_lens is None else torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    if seq_lens.shape != (sq,):
        raise ValueError(f"test setup requires one logical length per query row, got {tuple(seq_lens.shape)} for sq={sq}")
    return {
        "ratio": ratio,
        "coff": coff,
        "num_blocks": num_blocks,
        "topk": topk,
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
    topk_size = data["topk"]
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
        top_k=topk_size,
        return_val=True,
        deterministic=True,
    )
    block_indices = topk["indices"].view(1, scores.shape[1], topk_size)
    tied_topk = None
    if check_topk:
        assert torch.all(block_indices >= 0)
        gathered_values = torch.gather(scores, -1, block_indices.long())
        assert torch.equal(topk["values"].view_as(gathered_values), gathered_values)

        tie_count = topk_size + 32
        assert num_blocks >= tie_count, "tie probe needs equal maxima on both sides of the cutoff"
        # A same-shape side probe places equal maxima across the Top-K cutoff.
        # Deterministic Top-K must choose and order the smallest source indices,
        # independent of scheduling.
        tied_scores = torch.zeros_like(scores)
        tied_scores[..., :tie_count] = 1.0
        tied_topk = DSA.indexer_top_k_wrapper(
            tied_scores.reshape(-1, num_blocks),
            data["seq_lens"],
            top_k=topk_size,
            return_val=True,
            deterministic=True,
        )
        tied_indices = tied_topk["indices"].view_as(block_indices)
        expected_indices = torch.arange(topk_size, dtype=torch.int32, device="cuda").view(1, 1, topk_size).expand_as(tied_indices)
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
        .view(scores.shape[1], topk_size, ratio)
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
        assert torch.equal(bitwise_bits(actual[name]), bitwise_bits(expected[name])), f"{name} changed{context}"


def _assert_no_unexpected_nonfinite(outputs):
    for name, tensor in outputs.items():
        if name in ("index_scores", "topk_values", "tied_topk_values"):
            assert torch.all(torch.isfinite(tensor) | torch.isneginf(tensor)), f"{name} contains NaN or positive infinity"
        else:
            assert torch.isfinite(tensor).all(), f"{name} contains a non-finite value"


@pytest.mark.L0
def test_csa_dsa_deterministic_training_pipeline_smoke():
    """Default CI executes one complete strict-deterministic training step twice."""
    data = _inputs()
    reference = _clone_outputs(_run_pipeline(data))
    _assert_no_unexpected_nonfinite(reference)
    actual = _clone_outputs(_run_pipeline(data))
    _assert_bitwise_equal(actual, reference)


@pytest.mark.L1
def test_csa_dsa_deterministic_training_pipeline_repeat_100():
    """Every floating output and gradient is bitwise stable for 100 executions."""
    data = _inputs()
    reference = _clone_outputs(_run_pipeline(data, check_topk=True))
    _assert_no_unexpected_nonfinite(reference)
    for repeat in range(1, 100):
        actual = _clone_outputs(_run_pipeline(data, check_topk=True))
        _assert_bitwise_equal(actual, reference, f" at repeat {repeat}")


@pytest.mark.L1
@pytest.mark.parametrize(
    "ratio,coff,total_tokens,sq,attention_heads,with_sink,backend,planes,seq_lens",
    [
        pytest.param(2, 1, 1023, 1, 16, False, "default", 96, (1,), id="singleton-h16-r2-tail"),
        pytest.param(4, 2, 1024, 4, 32, True, "default", 128, (1, 2, 127, 128), id="length-boundaries-h32"),
        pytest.param(4, 1, 1025, 129, 128, True, "sm100_v2", 148, None, id="q-wave-plus-one-h128-tail-v2"),
    ],
)
def test_csa_dsa_deterministic_training_pipeline_configuration_matrix(ratio, coff, total_tokens, sq, attention_heads, with_sink, backend, planes, seq_lens):
    """Orthogonally cover compressor modes, attention heads, lengths, backends, and planes."""
    data = _inputs(
        ratio=ratio,
        coff=coff,
        total_tokens=total_tokens,
        sq=sq,
        attention_heads=attention_heads,
        with_sink=with_sink,
        seq_lens=seq_lens,
        seed=2400 + ratio + coff + planes + sq + attention_heads,
    )
    reference = _clone_outputs(_run_pipeline(data, deterministic_dk_planes=planes, backend=backend))
    _assert_no_unexpected_nonfinite(reference)
    if seq_lens is not None:
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


@pytest.mark.L1
@pytest.mark.parametrize(
    "total_tokens,sq,topk,attention_heads,backend",
    [
        pytest.param(8194, 2, 2048, 32, "default", id="short-k2048-default"),
        pytest.param(4096, 193, 896, 64, "sm100_v2", id="persistent-k896-v2"),
    ],
)
def test_csa_dsa_deterministic_training_pipeline_large_topk(total_tokens, sq, topk, attention_heads, backend):
    """Large Top-K composes through attention and repeated indexer producer handoffs."""
    data = _inputs(total_tokens=total_tokens, sq=sq, topk=topk, attention_heads=attention_heads, seed=2600 + sq + topk)
    reference = _clone_outputs(_run_pipeline(data, backend=backend))
    _assert_no_unexpected_nonfinite(reference)
    for repeat in range(1, 3):
        actual = _clone_outputs(_run_pipeline(data, backend=backend))
        _assert_bitwise_equal(actual, reference, f" at repeat {repeat}")


@pytest.mark.L1
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
            graph.replay()
            reference = _clone_outputs(captured)
            _assert_no_unexpected_nonfinite(reference)
            for repeat in range(1, 8):
                graph.replay()
                actual = _clone_outputs(captured)
                _assert_bitwise_equal(actual, reference, f" at graph replay {repeat}")
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
