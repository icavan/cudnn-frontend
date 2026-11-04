# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and training-contract coverage for native DSA composition."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from cudnn import DSA

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def supported():
    capability = torch.cuda.get_device_capability()
    if capability not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("native DSA training requires SM90, SM100, or SM103")
    if capability == (9, 0):
        from cudnn.deepseek_sparse_attention.sparse_attention_forward._support_sm90 import sm90_dsa_cutedsl_requirement_error

        error = sm90_dsa_cutedsl_requirement_error()
        if error:
            pytest.skip(error)
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("DSA training test")
    if error:
        pytest.skip(error)


def inputs(heads=16, dim=512, sink=True, lengths=True):
    torch.manual_seed(125)
    q = (torch.randn(4, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    kv = (torch.randn(19, dim, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    a = torch.randn(heads, device="cuda", requires_grad=True) if sink else None
    idx = torch.tensor([[1, -9, 4, 22, 1, 8, 3], [5, 6, 2, 1, 8, 3, 4], [-1] * 7, [2, 1, 0, 9, 8, 10, 12]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([5, 2, -1, 90], dtype=torch.int32, device="cuda") if lengths else None
    return q, kv, idx, a, lens


def reference(q, kv, idx, sink, lengths):
    valid = (idx >= 0) & (idx < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(idx.shape[1], device=idx.device)[None] < lengths.clamp(0, idx.shape[1])[:, None]
    selected = kv[idx.clamp(0, kv.shape[0] - 1).long()].float()
    scores = torch.einsum("shd,skd->shk", q.float(), selected) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~valid[:, None], -float("inf"))
    sink_logits = q.new_full(q.shape[:2], -float("inf"), dtype=torch.float32) if sink is None else sink[None].expand(q.shape[:2])
    if sink is None:
        sink_logits = torch.where(valid.any(-1)[:, None], sink_logits, 0.0)
    probs = torch.softmax(torch.cat([scores, sink_logits[..., None]], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("shk,skd->shd", probs, selected[..., :512])
    lse = torch.logsumexp(scores, -1)
    lse = torch.where(valid.any(-1)[:, None], lse, float("inf"))
    target = torch.exp(scores - lse[..., None]).sum(1)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    return out, lse, target, valid


def assert_numerics(actual, expected, tol=0.035):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=tol * max(1.0, expected.detach().abs().max().item()), rtol=tol)


@pytest.mark.parametrize(
    "heads,dim,sink,lengths",
    [
        (16, 512, True, True),
        (16, 576, True, True),
        (32, 576, False, True),
        (32, 512, False, True),
        (64, 512, True, False),
        (64, 576, True, True),
        (96, 512, True, True),
        (96, 576, True, True),
        (128, 512, True, True),
        (128, 576, True, True),
    ],
)
def test_native_training_and_original_score_slots(heads, dim, sink, lengths):
    if torch.cuda.get_device_capability() != (9, 0) and (heads == 96 or (heads, dim) == (128, 576)):
        pytest.skip("this H/D training pair is native only on SM90")
    q, kv, idx, a, lens = inputs(heads, dim, sink, lengths)
    qr, kr = q.detach().float().requires_grad_(), kv.detach().float().requires_grad_()
    ar = a.detach().clone().requires_grad_() if a is not None else None
    out_ref, lse_ref, target_ref, valid = reference(qr, kr, idx, ar, lens)
    result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
    assert not result["lse"].requires_grad and not result["max_logits"].requires_grad
    assert_numerics(result["out"], out_ref, 0.01)
    torch.testing.assert_close(result["lse"], lse_ref, atol=3e-4, rtol=3e-4)
    grad = torch.randn_like(result["out"])
    result["out"].backward(grad)
    out_ref.backward(grad.float())
    for actual, expected in ((q.grad, qr.grad), (kv.grad, kr.grad)):
        assert_numerics(actual, expected)
    if a is not None:
        assert_numerics(a.grad, ar.grad)
    score = DSA.sparse_attention_score_recompute(q.detach(), kv.detach(), result["lse"], idx, topk_length=lens)
    torch.testing.assert_close(score["target"], target_ref.detach(), atol=2e-3, rtol=2e-3)
    assert torch.equal(score["indices"], torch.where(valid, idx, -1))
    assert torch.equal(score["target"][~valid], torch.zeros_like(score["target"][~valid]))


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_and_no_device_sync(reentrant):
    q, kv, idx, a, lens = inputs()

    def run(q, kv, a):
        return DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]

    # Prime JIT before testing the warmed execution path.
    run(q, kv, a).sum().backward()
    expected = [t.grad.clone() for t in (q, kv, a)]
    for t in (q, kv, a):
        t.grad = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        checkpoint(run, q, kv, a, use_reentrant=reentrant).sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for tensor, ref in zip((q, kv, a), expected):
        assert_numerics(tensor.grad, ref, 0.01)


def test_inactive_nan_suffix_and_trusted_metadata():
    q, kv, idx, a, lens = inputs()
    with torch.no_grad():
        kv[18].fill_(float("nan"))
    idx[:, -1] = 18
    lens = torch.full((4,), 5, device="cuda", dtype=torch.int32)
    result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
    assert torch.isfinite(result["out"]).all()
    from cudnn.deepseek_sparse_attention.training import _normalize_cudnn_sparse_metadata

    safe, count = _normalize_cudnn_sparse_metadata(idx, lens, kv.shape[0])
    fast = DSA.sparse_attention(q, kv, safe, a, topk_length=count, trusted_compact_metadata=True)
    torch.testing.assert_close(fast["out"], result["out"], atol=0, rtol=0)


def test_stream_and_changed_input_graph_replay():
    q, kv, idx, a, lens = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
        q.mul_(2)
        kv.mul_(0.5)
        graph.replay()
        expected = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
        torch.testing.assert_close(captured, expected)
    torch.cuda.current_stream().wait_stream(stream)


def test_unsupported_shape_and_device_scale_fail_before_launch():
    capability = torch.cuda.get_device_capability()
    q, kv, idx, a, lens = inputs(8 if capability == (9, 0) else 128, 512 if capability == (9, 0) else 576)
    with pytest.raises(NotImplementedError, match="native training supports"):
        DSA.sparse_attention(q, kv, idx, a)
    q, kv, idx, a, lens = inputs()
    with pytest.raises(TypeError, match="host real"):
        DSA.sparse_attention(q, kv, idx, a, softmax_scale=torch.tensor(1.0, device="cuda"))
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        DSA.sparse_attention(q, kv, idx, a, deterministic=1)
    with pytest.raises(ValueError, match="requires deterministic=True"):
        DSA.sparse_attention(q, kv, idx, a, deterministic_relay_cluster_size=16)
    with pytest.raises(ValueError, match="one of"):
        DSA.sparse_attention(q, kv, idx, a, deterministic=True, deterministic_relay_cluster_size=3)


def test_old_dsl_declines_before_any_kernel_launch(monkeypatch):
    from cudnn.frost import buffers

    q, kv, idx, a, lens = inputs()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    required = "4.8.0" if torch.cuda.get_device_capability() == (9, 0) else "4.7.0"
    with pytest.raises(RuntimeError, match=rf"requires nvidia-cutlass-dsl >= {required}; found 4.6.2"):
        DSA.sparse_attention(q, kv, idx, a)


def test_training_graph_replay_uses_changed_inputs_and_metadata():
    q, kv, idx, a, lens = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def step():
            for tensor in (q, kv, a):
                if tensor.grad is not None:
                    tensor.grad.zero_()
            output = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
            output.sum().backward()
            return output

        for _ in range(3):
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = step()
        with torch.no_grad():
            q.mul_(1.5)
            kv.mul_(0.7)
            idx[0, 0] = 3
            lens[1] = 1
        graph.replay()
        actual = [tensor.grad.clone() for tensor in (q, kv, a)]
        actual_output = captured.clone()
        expected_output = step()
        assert_numerics(actual_output, expected_output, 0.01)
        for grad, tensor in zip(actual, (q, kv, a)):
            assert_numerics(grad, tensor.grad, 0.01)
    torch.cuda.current_stream().wait_stream(stream)


def test_sm90_training_keeps_native_heads_and_unaligned_topk(monkeypatch):
    """SM90 forward/backward/score recompute consume the logical shape directly."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("native H/K no-padding composition targets SM90")

    from cudnn.deepseek_sparse_attention.score_recompute import api as score_api
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import api as bwd_api
    from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as fwd_api

    seen = {"fwd": [], "bwd": [], "score": []}
    real_fwd = fwd_api.sparse_attention_forward_wrapper
    real_bwd = bwd_api.sparse_attention_backward_wrapper
    real_score = score_api.sparse_attn_score_recompute_wrapper

    def record_fwd(q, kv, indices, **kwargs):
        seen["fwd"].append((q.shape[1], indices.shape[1]))
        return real_fwd(q, kv, indices, **kwargs)

    def record_bwd(q, kv, out, dout, lse, sink, indices, **kwargs):
        seen["bwd"].append((q.shape[1], indices.shape[1]))
        return real_bwd(q, kv, out, dout, lse, sink, indices, **kwargs)

    def record_score(q, kv, lse, indices, scale, **kwargs):
        seen["score"].append((q.shape[2], indices.shape[2]))
        return real_score(q, kv, lse, indices, scale, **kwargs)

    def fail_on_pad(*args, **kwargs):
        pytest.fail("SM90 E2E composition must not call torch.nn.functional.pad")

    monkeypatch.setattr(fwd_api, "sparse_attention_forward_wrapper", record_fwd)
    monkeypatch.setattr(bwd_api, "sparse_attention_backward_wrapper", record_bwd)
    monkeypatch.setattr(score_api, "sparse_attn_score_recompute_wrapper", record_score)
    monkeypatch.setattr(torch.nn.functional, "pad", fail_on_pad)

    q, kv, idx, a, _ = inputs(heads=32, dim=512, sink=True, lengths=False)
    assert idx.shape[1] == 7
    result = DSA.sparse_attention(q, kv, idx, a, trusted_compact_metadata=True)
    result["out"].sum().backward()
    target = DSA.sparse_attention_score_recompute(q.detach(), kv.detach(), result["lse"], idx)

    assert seen == {"fwd": [(32, 7)], "bwd": [(32, 7)], "score": [(32, 7)]}
    assert target["target"].shape == idx.shape
    assert target["target"].is_contiguous()


@pytest.mark.parametrize("relay_cluster_size", [4, 8, 16, 32, 64])
def test_sm90_training_deterministic_relay_is_bitwise_stable(relay_cluster_size):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("deterministic E2E relay coverage targets SM90")

    q, kv, idx, a, _ = inputs(heads=64, dim=512, sink=True, lengths=False)
    grad = torch.randn(q.shape[0], q.shape[1], 512, dtype=q.dtype, device=q.device)
    reference = None
    for _ in range(3):
        result = DSA.sparse_attention(
            q,
            kv,
            idx,
            a,
            trusted_compact_metadata=True,
            deterministic=True,
            deterministic_relay_cluster_size=relay_cluster_size,
        )
        actual = torch.autograd.grad(result["out"], (q, kv, a), grad)
        if reference is None:
            reference = tuple(t.clone() for t in actual)
        else:
            assert all(torch.equal(got, want) for got, want in zip(actual, reference))


@pytest.mark.L1
def test_sm90_training_deterministic_relay_cuda_graph_replay():
    """The warmed E2E deterministic plan captures without allocation or host work."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("deterministic E2E relay coverage targets SM90")

    q, kv, idx, a, _ = inputs(heads=64, dim=512, sink=True, lengths=False)
    grad = torch.randn(q.shape[0], q.shape[1], 512, dtype=q.dtype, device=q.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def step():
            for tensor in (q, kv, a):
                if tensor.grad is not None:
                    tensor.grad.zero_()
            output = DSA.sparse_attention(
                q,
                kv,
                idx,
                a,
                trusted_compact_metadata=True,
                deterministic=True,
                deterministic_relay_cluster_size=16,
            )["out"]
            output.backward(grad)
            return output

        for _ in range(3):
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = step()

        graph.replay()
        first = (captured.clone(), *(tensor.grad.clone() for tensor in (q, kv, a)))
        graph.replay()
        second = (captured.clone(), *(tensor.grad.clone() for tensor in (q, kv, a)))
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second))

        with torch.no_grad():
            q.mul_(1.25)
            kv.mul_(0.75)
        graph.replay()
        replay = (captured.clone(), *(tensor.grad.clone() for tensor in (q, kv, a)))
        eager_output = step()
        eager = (eager_output.clone(), *(tensor.grad.clone() for tensor in (q, kv, a)))
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(replay, eager))
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("order", [((10, 0), (10, 3)), ((10, 3), (10, 0))], ids=["sm100_then_sm103", "sm103_then_sm100"])
def test_compiled_dependencies_follow_the_device_target(monkeypatch, order):
    """One process may drive an SM100 and an SM103 GPU: the training composition's compiled dependencies (the index compaction and
    the attention score recompute) take the device's CuTe target per call and key their artifacts by it, so neither part reuses or
    compiles the other's sm_100a / sm_103a code. Host-modeled: the capability is patched and cute.compile only records its target."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the target-switching probe models SM100/SM103")

    import re

    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.indexer_top_k import compactify as compact_mod
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm100 as score_mod
    from cudnn.deepseek_sparse_attention.utils.compiler import _ARCH_MAP, gpu_arch_flag

    compiled = []

    def record(kernel, *args, options=""):
        compiled.append((type(kernel).__name__, re.search(r"--gpu-arch (\S+)", options).group(1)))
        return lambda *a, **k: None

    monkeypatch.setattr(cute, "compile", record)
    monkeypatch.setattr(compact_mod, "_compile_cache", {})
    monkeypatch.setattr(score_mod._sparse_attn_score_recompute, "compile_cache", {})
    idx = torch.tensor([[1, -1, 3, 0], [2, 2, -1, -1]], dtype=torch.int32, device="cuda")
    q = torch.zeros(1, 4, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 19, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(1, 4, 16, device="cuda")
    top = torch.zeros(1, 4, 128, device="cuda", dtype=torch.int32)
    out = torch.empty(1, 4, 128, device="cuda")
    for cc in order + order:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, cc=cc: cc)
        assert gpu_arch_flag() == gpu_arch_flag(idx.device) == _ARCH_MAP[cc]
        compact_mod.compactify(idx)
        score_mod.sparse_attn_score_recompute(q, k, lse, top, 512**-0.5, qhead_per_kv_head=16, out=out, topk_indices_global=False)
    want = [_ARCH_MAP[cc] for cc in order]  # one compile per part; the repeat of each part hits its own artifact
    assert [arch for name, arch in compiled if name == "CompactifyKernel"] == want, compiled
    assert [arch for name, arch in compiled if name == "SparseScoreRecomputeSm100"] == want, compiled
    assert sorted(key[-1] for key in compact_mod._compile_cache) == sorted(want)
