# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrated compressed/direct sparse-attention training coverage."""

import warnings

import pytest
import torch
from cudnn import CSA, CompressedSparseAttention
from test_utils import bitwise_bits

pytestmark = pytest.mark.L1


@pytest.fixture(autouse=True)
def supported():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("integrated sparse-attention training requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("integrated sparse-attention training test")
    if error:
        pytest.skip(error)


def _inputs(*, ratio=1, coff=1, sq=4, tokens=512, indexer_heads=64, attention_heads=64, top_k=None, seed=9021):
    """Create one single-sequence module invocation."""

    torch.manual_seed(seed)
    top_k = 128 * ratio if top_k is None else top_k
    blocks = tokens if ratio == 1 else tokens // ratio
    index_q = torch.randn(1, sq, indexer_heads, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    if ratio == 1:
        index_k = torch.randn(1, blocks, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        compressor_score = ape = cu_seqlens = cu_seqlens_comp = None
    else:
        index_k = torch.randn(tokens, coff * 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        compressor_score = torch.randn_like(index_k, requires_grad=True)
        ape = torch.randn(ratio, coff * 128, dtype=torch.float32, device="cuda", requires_grad=True)
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        cu_seqlens_comp = torch.tensor([0, blocks], dtype=torch.int32, device="cuda")
    weights = torch.randn(1, sq, indexer_heads, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_q = torch.randn(sq, attention_heads, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_kv = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    attention_sink = torch.randn(attention_heads, dtype=torch.float32, device="cuda", requires_grad=True)
    seq_lens = torch.full((sq,), blocks, dtype=torch.int32, device="cuda")
    q_causal_offsets = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    return {
        "index_q": index_q,
        "index_k": index_k,
        "weights": weights,
        "attention_q": attention_q,
        "attention_kv": attention_kv,
        "seq_lens": seq_lens,
        "q_causal_offsets": q_causal_offsets,
        "attention_sink": attention_sink,
        "compressor_score": compressor_score,
        "ape": ape,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_comp": cu_seqlens_comp,
        "top_k": top_k,
        "block_top_k": top_k // ratio,
    }


def _call(module, data, **kwargs):
    """Invoke the public module from a shared input dictionary."""

    return module(
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
        **kwargs,
    )


def _grad_inputs(module, data):
    tensors = [data[name] for name in ("index_q", "index_k", "weights", "attention_q", "attention_kv")]
    if data["attention_sink"] is not None:
        tensors.append(data["attention_sink"])
    if data["compressor_score"] is not None:
        tensors.extend((data["compressor_score"], data["ape"]))
    tensors.extend((module.k_norm.weight, module.k_norm.bias))
    return tensors


def _assert_bitwise_equal(actual, expected, context=""):
    assert torch.equal(bitwise_bits(actual.reshape(-1)), bitwise_bits(expected.reshape(-1))), context


def test_public_exports_are_the_same_class():
    """Top-level and namespace imports share one lazily loaded class."""

    assert CompressedSparseAttention is CSA.CompressedSparseAttention


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"compress_ratio": 128, "top_k": 2048}, "compress_ratio"),
        ({"compress_ratio": True}, "compress_ratio must be an int"),
        ({"compress_ratio": 1, "coff": 2}, "coff must be 1"),
        ({"top_k": 2048.0}, "top_k must be an int"),
        ({"compress_ratio": 4, "top_k": 256}, "128 \\* compress_ratio"),
        ({"k_norm_placement": "middle"}, "k_norm_placement"),
        ({"deterministic": False, "deterministic_dk_planes": 128}, "only when deterministic"),
        ({"deterministic_dk_planes": 96.0}, "deterministic_dk_planes must be an int"),
        ({"dtype": torch.float32}, "k_norm parameters must use torch.bfloat16"),
    ],
)
def test_configuration_rejects_before_layernorm_allocation(monkeypatch, kwargs, message):
    """Invalid static policy never reaches parameter allocation."""

    def forbidden(*args, **kw):
        raise AssertionError("invalid configuration allocated LayerNorm")

    monkeypatch.setattr(torch.nn, "LayerNorm", forbidden)
    with pytest.raises((TypeError, ValueError), match=message):
        CompressedSparseAttention(**kwargs)


def test_direct_nope_bypasses_compressor_and_index_expansion(monkeypatch):
    """Ratio one invokes no compressor and reuses Top-K storage as token ids."""

    import cudnn.csa.training as training

    data = _inputs(ratio=1)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("direct path reached the compressor")

    monkeypatch.setattr(training._Compressor, "apply", forbidden)
    result = _call(module, data)
    assert result["token_indices"].untyped_storage().data_ptr() == result["block_indices"].untyped_storage().data_ptr()
    assert torch.isfinite(result["out"]).all()
    assert torch.isfinite(result["indexer_loss"])
    loss = result["out"].float().square().mean() + result["indexer_loss"]
    grads = torch.autograd.grad(loss, _grad_inputs(module, data))
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_deterministic_h64_contract_rejects_before_pipeline(monkeypatch):
    """Unsupported deterministic indexer head counts fail before any kernel."""

    import cudnn.csa.training as training

    data = _inputs(ratio=1, indexer_heads=32)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid deterministic call reached the pipeline")

    monkeypatch.setattr(training._SparseAttentionPipeline, "apply", forbidden)
    with pytest.raises(ValueError, match="require indexer H=64"):
        _call(module, data)


@pytest.mark.parametrize(
    "input_update,message",
    [
        ({"sq": 0}, "S_q must be positive"),
        ({"indexer_heads": 16}, "requires H=32 or H=64"),
        ({"sq": 513}, "must be <= compressed K length"),
    ],
)
def test_invalid_indexer_geometry_rejects_before_pipeline(monkeypatch, input_update, message):
    """Unsupported training geometry fails before normalization or kernels."""

    import cudnn.csa.training as training

    data = _inputs(ratio=1, **input_update)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid indexer geometry reached the pipeline")

    monkeypatch.setattr(training._SparseAttentionPipeline, "apply", forbidden)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        _call(module, data)


def test_unsupported_architecture_rejects_before_pipeline(monkeypatch):
    """The integrated SM100 contract is checked before optional-kernel work."""

    import cudnn.csa.training as training

    data = _inputs(ratio=1)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported architecture reached the pipeline")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (9, 0))
    monkeypatch.setattr(training._SparseAttentionPipeline, "apply", forbidden)
    with pytest.raises(NotImplementedError, match="requires SM100"):
        _call(module, data)


@pytest.mark.parametrize("placement", ["before_pool", "after_pool"])
def test_k_norm_placement_selects_the_declared_order(monkeypatch, placement):
    """LayerNorm is applied to raw projections or pooled keys, never both."""

    import cudnn.csa.training as training

    data = _inputs(ratio=4, coff=1)
    module = CompressedSparseAttention(
        compress_ratio=4,
        coff=1,
        top_k=data["top_k"],
        k_norm_placement=placement,
        device="cuda",
    )
    captured = {}

    def pool(kv, _score, _ape, _cu, _cuc, ratio, head_dim, coff, total_comp, _deterministic):
        assert (ratio, head_dim, coff, total_comp) == (4, 128, 1, 128)
        return kv.view(total_comp, ratio, head_dim).mean(1).to(torch.bfloat16)

    def capture(index_q, index_k, *args):
        captured["k"] = index_k
        out = torch.zeros_like(data["attention_q"])
        stats = torch.zeros(data["attention_q"].shape[:2], dtype=torch.float32, device="cuda")
        block = torch.zeros(1, data["index_q"].shape[1], data["block_top_k"], dtype=torch.int32, device="cuda")
        return out, stats, stats.clone(), block, block.squeeze(0), index_k.float().sum() * 0

    monkeypatch.setattr(training._Compressor, "apply", pool)
    monkeypatch.setattr(training._SparseAttentionPipeline, "apply", capture)
    _call(module, data)

    raw = data["index_k"].view(128, 4, 128)
    if placement == "before_pool":
        expected = module.k_norm(raw).mean(1).to(torch.bfloat16).unsqueeze(0)
    else:
        expected = module.k_norm(raw.mean(1).to(torch.bfloat16)).unsqueeze(0)
    torch.testing.assert_close(captured["k"], expected, atol=0, rtol=0)


def test_no_sink_has_no_sink_gradient():
    """The internal all-negative-infinity sink remains an implementation detail."""

    data = _inputs(ratio=1)
    data["attention_sink"] = None
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")
    result = _call(module, data)
    grads = torch.autograd.grad(result["out"].float().square().mean() + result["indexer_loss"], _grad_inputs(module, data))
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_attention_head_count_is_independent_from_indexer_heads():
    """H32 attention composes with the deterministic H64 indexer end to end."""

    data = _inputs(ratio=1, indexer_heads=64, attention_heads=32)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")
    result = _call(module, data)
    assert result["out"].shape == data["attention_q"].shape
    assert result["max_logits"].shape == data["attention_q"].shape[:2]
    loss = result["out"].float().square().mean() + result["indexer_loss"]
    grads = torch.autograd.grad(loss, _grad_inputs(module, data))
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_short_valid_prefixes_remain_finite_and_in_range():
    """Per-query K prefixes can differ without selecting masked score columns."""

    data = _inputs(ratio=1, sq=4)
    data["seq_lens"].copy_(torch.tensor([128, 192, 320, 511], dtype=torch.int32, device="cuda"))
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")
    result = _call(module, data)
    row_limits = data["seq_lens"].view(-1, 1)
    assert torch.all(result["block_indices"].squeeze(0) >= 0)
    assert torch.all(result["block_indices"].squeeze(0) < row_limits)
    assert torch.isfinite(result["out"]).all()
    assert torch.isfinite(result["indexer_loss"])
    grads = torch.autograd.grad(result["out"].float().square().mean() + result["indexer_loss"], _grad_inputs(module, data))
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("ratio", [1, 4])
def test_deterministic_topk_uses_smaller_indices_for_ties(ratio):
    """Direct and compressed routes preserve Top-K's deterministic tie rule."""

    data = _inputs(ratio=ratio, sq=2)
    data["index_q"] = torch.zeros_like(data["index_q"], requires_grad=True)
    module = CompressedSparseAttention(compress_ratio=ratio, top_k=data["top_k"], deterministic=True, device="cuda")
    result = _call(module, data)
    expected = torch.arange(data["block_top_k"], dtype=torch.int32, device="cuda").expand(data["index_q"].shape[1], -1)
    assert torch.equal(result["block_indices"].squeeze(0), expected)


def test_indexer_loss_backward_rejects_retained_graph_reuse():
    """The native in-place target scratch cannot silently corrupt a second backward."""

    data = _inputs(ratio=1)
    module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], deterministic=True, device="cuda")
    result = _call(module, data)
    targets = [data["index_q"], data["index_k"], data["weights"]]
    torch.autograd.grad(result["indexer_loss"], targets, retain_graph=True)
    with pytest.raises(RuntimeError, match="indexer_loss backward is single-use"):
        torch.autograd.grad(result["indexer_loss"], targets)


@pytest.mark.parametrize(
    "ratio,placement",
    [
        pytest.param(2, "before_pool", id="r2-before"),
        pytest.param(2, "after_pool", id="r2-after"),
        pytest.param(4, "before_pool", id="r4-before"),
        pytest.param(4, "after_pool", marks=pytest.mark.L0, id="r4-after-smoke"),
    ],
)
def test_compressed_deterministic_autograd_is_bitwise_stable(ratio, placement):
    """Both compression ratios and normalization orders cover deterministic backward."""

    data = _inputs(ratio=ratio, coff=1)
    module = CompressedSparseAttention(
        compress_ratio=ratio,
        coff=1,
        top_k=data["top_k"],
        k_norm_placement=placement,
        deterministic=True,
        deterministic_dk_planes=96,
        device="cuda",
    )
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        reference = None
        for _ in range(2):
            result = _call(module, data)
            loss = result["out"].float().square().mean() + result["indexer_loss"]
            grads = torch.autograd.grad(loss, _grad_inputs(module, data))
            current = [result["out"].detach().clone(), result["indexer_loss"].detach().clone(), result["block_indices"].detach().clone()]
            current.extend(grad.detach().clone() for grad in grads)
            if reference is None:
                reference = current
            else:
                for actual, expected in zip(current, reference):
                    _assert_bitwise_equal(actual, expected)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def test_public_module_deterministic_autograd_repeat_100():
    """The public compressed module keeps every result and gradient bitwise stable."""

    data = _inputs(ratio=4, coff=1)
    module = CompressedSparseAttention(
        compress_ratio=4,
        coff=1,
        top_k=data["top_k"],
        deterministic=True,
        deterministic_dk_planes=96,
        device="cuda",
    )
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        reference = None
        for repeat in range(100):
            result = _call(module, data)
            loss = result["out"].float().square().mean() + result["indexer_loss"]
            grads = torch.autograd.grad(loss, _grad_inputs(module, data))
            current = [
                result["out"].detach().clone(),
                result["indexer_loss"].detach().clone(),
                result["block_indices"].detach().clone(),
                result["token_indices"].detach().clone(),
            ]
            current.extend(grad.detach().clone() for grad in grads)
            if reference is None:
                reference = current
                assert all(torch.isfinite(tensor).all() for tensor in current if tensor.is_floating_point())
            else:
                for name, actual, expected in zip(
                    ("out", "indexer_loss", "block_indices", "token_indices", *[f"grad_{i}" for i in range(len(grads))]),
                    current,
                    reference,
                ):
                    _assert_bitwise_equal(actual, expected, f"{name} changed at repeat {repeat}")
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def test_non_deterministic_path_is_rejected_by_strict_torch_mode():
    """A negative control proves strict mode distinguishes the atomic backward."""

    data = _inputs(ratio=4)
    module = CompressedSparseAttention(compress_ratio=4, top_k=data["top_k"], deterministic=False, device="cuda")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = _call(module, data)
        with pytest.raises(RuntimeError, match="deterministic"):
            torch.autograd.grad(result["out"].float().square().mean() + result["indexer_loss"], _grad_inputs(module, data))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def test_indexer_loss_coefficient_scales_value_and_gradients_once():
    """The custom backward matches the public mean-KL coefficient contract."""

    data = _inputs(ratio=1)
    reference_module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], indexer_loss_coeff=1.0, deterministic=True, device="cuda")
    scaled_module = CompressedSparseAttention(compress_ratio=1, top_k=data["top_k"], indexer_loss_coeff=0.25, deterministic=True, device="cuda")
    scaled_module.load_state_dict(reference_module.state_dict())

    reference = _call(reference_module, data)
    reference_inputs = [data["index_q"], data["index_k"], data["weights"], reference_module.k_norm.weight, reference_module.k_norm.bias]
    reference_grads = torch.autograd.grad(reference["indexer_loss"], reference_inputs)
    scaled = _call(scaled_module, data)
    scaled_inputs = [data["index_q"], data["index_k"], data["weights"], scaled_module.k_norm.weight, scaled_module.k_norm.bias]
    scaled_grads = torch.autograd.grad(scaled["indexer_loss"], scaled_inputs)

    assert torch.equal(scaled["indexer_loss"], reference["indexer_loss"] * 0.25)
    for actual, expected in zip(scaled_grads, reference_grads):
        assert torch.equal(actual, expected * 0.25)


@pytest.mark.parametrize("ratio", [1, 4])
def test_cuda_graph_replay_reads_changed_inputs(ratio):
    """Direct and compressed paths capture after warmup and rebind live values."""

    data = _inputs(ratio=ratio, tokens=1024)
    module = CompressedSparseAttention(compress_ratio=ratio, top_k=data["top_k"], deterministic=True, device="cuda")
    grad_inputs = _grad_inputs(module, data)

    def run():
        result = _call(module, data)
        loss = result["out"].float().square().mean() + result["indexer_loss"]
        grads = torch.autograd.grad(loss, grad_inputs)
        return result, grads

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            warm_result, warm_grads = run()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    del warm_result, warm_grads
    graph = torch.cuda.CUDAGraph()
    try:
        # The retained leaf AccumulateGrad nodes intentionally span warmup and
        # capture on this same side stream. PyTorch 2.11 still warns for this
        # valid pattern; replay equality below is the stronger safety check.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The AccumulateGrad node's stream does not match the stream of the node that produced the incoming gradient.*",
                category=UserWarning,
            )
            with torch.cuda.graph(graph, stream=capture_stream):
                captured_result, captured_grads = run()
        capture_stream.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        reference = [captured_result["out"].detach().clone(), captured_result["indexer_loss"].detach().clone()]
        reference.extend(grad.detach().clone() for grad in captured_grads)
        for repeat in range(1, 8):
            graph.replay()
            torch.cuda.synchronize()
            current = [captured_result["out"].detach().clone(), captured_result["indexer_loss"].detach().clone()]
            current.extend(grad.detach().clone() for grad in captured_grads)
            for name, actual, expected in zip(("out", "indexer_loss", *[f"grad_{i}" for i in range(len(captured_grads))]), current, reference):
                _assert_bitwise_equal(actual, expected, f"{name} changed at graph replay {repeat}")
        with torch.no_grad():
            data["index_q"].add_(0.25)
            data["index_k"].mul_(0.75)
            if data["compressor_score"] is not None:
                data["compressor_score"].add_(0.0625)
                data["ape"].mul_(0.875)
            data["attention_q"].mul_(0.5)
            data["attention_kv"].add_(0.125)
            data["attention_sink"].add_(0.03125)
            data["seq_lens"].sub_(1)
            data["q_causal_offsets"].sub_(ratio + 1)
        graph.replay()
        torch.cuda.synchronize()
        replay = [captured_result["out"].detach().clone(), captured_result["indexer_loss"].detach().clone()]
        replay.extend(grad.detach().clone() for grad in captured_grads)
        with torch.cuda.stream(capture_stream):
            expected_result, expected_grads = run()
        capture_stream.synchronize()
        expected = [expected_result["out"].detach(), expected_result["indexer_loss"].detach(), *expected_grads]
        for actual, reference in zip(replay, expected):
            assert torch.equal(actual, reference)
    finally:
        graph.reset()
