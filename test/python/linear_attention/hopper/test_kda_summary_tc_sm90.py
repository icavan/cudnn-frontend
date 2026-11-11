# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct-M mixed-precision SM90 summaries; the FP32 suite remains unchanged.

This path deliberately rounds MMA operands to BF16. Its independent numerical
checks use a 1% normwise budget and a separate maximum-error cap; they do not
reuse or relax the FP32 engine's tolerance. This is a summary API, so backward
outputs are the boundary-state gradient and transition, not dQ/dK/dV/dg/dbeta.
"""

import math
from pathlib import Path

import pytest
import torch
import cudnn

import test_kda_summary_sm90 as fp32
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

ENGINE = "kda_summary_hopper_tc"
pytestmark = fp32.pytestmark


@pytest.fixture(autouse=True)
def select_tensor_core(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(fp32, "ENGINE", ENGINE)
    original = fp32.make_graph

    def make_graph(x, backward=False, transition=True):
        tokens, heads, _ = x["k"].shape
        seqs = x["cu"].numel() - 1
        size = (tokens // 16 + seqs) * heads * (15456 if backward else 10400) if tokens and seqs else 0
        return original(x, backward, transition, expected_workspace=size)

    monkeypatch.setattr(fp32, "make_graph", make_graph)


def assert_accuracy(actual, expected):
    for got, ref in zip(actual, expected):
        assert got.dtype == torch.float32
        assert torch.isfinite(got).all() and torch.isfinite(ref).all()
        error = got.double() - ref.double()
        assert error.norm() <= 0.01 * ref.norm() + 1e-5 * math.sqrt(ref.numel())
        assert error.abs().max() <= 0.02 * ref.abs().max() + 1e-4


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("seeded", [False, True], ids=["zero", "seed"])
@pytest.mark.parametrize("gate", [-0.001, -0.1, -5.0, -1e8], ids=["weak", "moderate", "strong", "extreme"])
def test_summary_fp64(backward, seeded, gate):
    x = fp32.make_case((1, 15, 16, 17, 0, 33, 65), heads=2, gate=gate)
    assert_accuracy(fp32.run(x, backward, seeded), fp32.reference(x, backward, seeded))


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_transition_independence(backward):
    x = fp32.make_case((256,), heads=2)
    _, expected = fp32.run(x, backward)
    expected = expected.clone()
    for name in ("q", "v", "dO", "state"):
        x[name].mul_(32)
    _, got = fp32.run(x, backward, seeded=True)
    assert_bitwise({"M": got}, {"M": expected}, "direct M must be value/seed independent")


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_logical_cp64(backward):
    x = fp32.make_case((17,) * 64, heads=1)
    h, m = fp32.run(x, backward)
    merged = x["state"][0].double()
    for n in range(63, -1, -1) if backward else range(64):
        merged = merged @ m[n].double() + h[n].double()
    whole = dict(x, cu=torch.tensor([0, 17 * 64], device="cuda", dtype=torch.int32), state=x["state"][:1])
    expected, _ = fp32.reference(whole, backward, seeded=True)
    assert_accuracy((merged.float(),), (expected[0],))


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_replay_and_batch_invariance(backward):
    target = fp32.make_case((33,), heads=2)
    before = {k: v.clone() for k, v in target.items()}
    expected = assert_replays(lambda: fp32.outputs(fp32.run({k: v.clone() for k, v in target.items()}, backward, seeded=True, batch_invariant=True)), repeats=4)
    assert_accuracy(expected.values(), fp32.reference(target, backward, seeded=True))
    cases = [target, fp32.make_case((17,), seed=121), fp32.make_case((0,), seed=717)]
    for order in ((0, 1, 2, 0), (2, 0, 1, 0), tuple([1] * 20 + [0, 2, 0])):
        packed = {key: torch.cat([cases[i][key] for i in order]) for key in target if key != "cu"}
        lengths = [cases[i]["k"].shape[0] for i in order]
        packed["cu"] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        got = snapshot(fp32.outputs(fp32.run(packed, backward, seeded=True, batch_invariant=True)))
        for n, i in enumerate(order):
            if i == 0:
                assert_bitwise({name: value[n : n + 1] for name, value in got.items()}, expected, "THD batch invariance")
    assert_bitwise(target, before, "input mutation")
    assert_golden(f"kda_summary_tc_sm90_{'bwd' if backward else 'fwd'}_bf16", target, expected)


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64], ids=["cu32", "cu64"])
def test_strided_changed_input_capture(backward, cu_dtype, cudnn_handle, monkeypatch):
    x = fp32.make_case((17, 0, 65), heads=2)
    for name, value in list(x.items()):
        if name == "cu":
            x[name] = value.to(cu_dtype)
        else:
            shape = list(value.shape)
            shape[1] *= 2
            storage = torch.full(shape, float("nan"), device="cuda", dtype=value.dtype)
            x[name] = storage[:, : value.shape[1]]
            x[name].copy_(value)
    graph, _, ht, mt, bindings, h, m = fp32.make_graph(x, backward)
    padding = []
    for tensor, out in ((ht, h), (mt, m)):
        shape = list(out.shape)
        shape[1] *= 2
        storage = torch.full(shape, float("nan"), device="cuda")
        bindings[tensor] = storage[:, : out.shape[1]]
        padding.append(storage[:, out.shape[1] :])
    h, m = bindings[ht], bindings[mt]
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    from cudnn.linear_attention.cake import compiler

    def no_compile(*args, **kwargs):
        raise AssertionError("execute tried to compile")

    monkeypatch.setattr(compiler, "compile_cubin", no_compile)
    saved_stream = cudnn.get_stream(cudnn_handle)
    side = torch.cuda.Stream()
    capture = torch.cuda.CUDAGraph()
    try:
        cudnn.set_stream(cudnn_handle, side.cuda_stream)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph.execute(bindings, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(capture, stream=side):
            graph.execute(bindings, workspace, handle=cudnn_handle)
        for gate in (-0.001, -5.0, -0.001):
            x["g"].fill_(gate)
            x["v"].mul_(0.5)
            x["dO"].mul_(0.5)
            x["cu"].copy_(torch.tensor([0, 15, 48, 82], dtype=cu_dtype, device="cuda"))
            h.fill_(float("nan"))
            m.fill_(float("nan"))
            workspace.fill_(255)
            capture.replay()
            assert_accuracy((h, m), fp32.reference(x, backward, seeded=True))
            expected = snapshot(dict(h=h, m=m))
            capture.replay()
            assert_bitwise(dict(h=h, m=m), expected, "graph replay")
            assert all(torch.isnan(v).all() for v in padding)
    finally:
        capture.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("bad_binding", ["short", "alias", "overlap", "inner_stride", "wrong_device"])
def test_reject_before_launch(bad_binding, cudnn_handle, monkeypatch):
    fp32.test_invalid_binding_rejects_before_launch(bad_binding, cudnn_handle, monkeypatch)


def test_compile_count(monkeypatch):
    from cudnn.linear_attention.cake import compiler
    from cudnn.linear_attention.hopper import summary_engine

    original = compiler.KernelLibrary
    libraries = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        libraries.append(result)
        return result

    monkeypatch.setattr(summary_engine, "_TC_LIBRARIES", {})
    monkeypatch.setattr(compiler, "KernelLibrary", counted)
    for lengths, backward, cu_dtype in (((7,), False, torch.int32), ((3, 8), True, torch.int64), ((1,) * 65, False, torch.int32)):
        x = fp32.make_case(lengths, heads=3)
        x["cu"] = x["cu"].to(cu_dtype)
        fp32.make_graph(x, backward)
    assert len(libraries) == 1


def test_default_still_fp32(monkeypatch):
    monkeypatch.setattr(fp32, "ENGINE", "kda_summary_hopper_cuda")
    fp32.test_default_summary_uses_native_sm90_plan()


@pytest.mark.parametrize("scale", [0.0, -0.125, 0.5])
def test_backward_scale(scale):
    x = fp32.make_case((17, 0, 33), heads=1)
    assert_accuracy(fp32.run(x, backward=True, seeded=True, scale=scale), fp32.reference(x, backward=True, seeded=True, scale=scale))


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_empty_optional_and_partial_range(backward, cudnn_handle):
    for lengths in ((), (0, 0), (1, 0, 16), (17, 33)):
        x = fp32.make_case(lengths, heads=1)
        # Nonzero initial/final offsets exercise unused capacity and empty gaps.
        if lengths == (17, 33):
            x["cu"].copy_(torch.tensor([3, 17, 39], device="cuda", dtype=torch.int32))
        graph, _, _, _, bindings, h, _ = fp32.make_graph(x, backward, transition=False)
        workspace = torch.full((graph.get_workspace_size() + 32,), 255, dtype=torch.uint8, device="cuda")
        saved = cudnn.get_stream(cudnn_handle)
        try:
            cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
            graph.execute(bindings, workspace, handle=cudnn_handle)
            if lengths:
                assert_accuracy((h,), fp32.reference(x, backward, seeded=True)[:1])
            else:
                assert h.numel() == 0
            assert (workspace[graph.get_workspace_size() :] == 255).all()
        finally:
            cudnn.set_stream(cudnn_handle, saved)


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_mixed_gates_and_large_seed(backward):
    x = fp32.make_case((65, 17), heads=2)
    rng = torch.Generator().manual_seed(932)
    x["g"].copy_(-torch.rand(x["g"].shape, generator=rng).square())
    x["g"][16:32].fill_(-5)
    x["g"][32:48].fill_(-0.001)
    x["beta"].copy_(torch.rand(x["beta"].shape, generator=rng))
    x["state"].mul_(100)
    assert_accuracy(fp32.run(x, backward, seeded=True), fp32.reference(x, backward, seeded=True))


@pytest.mark.parametrize("kind", ["short", "zero", "misaligned", "input_alias", "output_alias", "cpu", "other_device", "strided"])
def test_workspace_rejects_before_launch(kind, cudnn_handle, monkeypatch):
    x = fp32.make_case((17,), heads=1)
    graph, tensors, ht, _, bindings, h, m = fp32.make_graph(x)
    size = graph.get_workspace_size()
    workspace = torch.empty(max(size, x["state"].numel() * 4) + 32, device="cuda", dtype=torch.uint8)
    if kind == "short":
        workspace = workspace[: size - 1]
    elif kind == "zero":
        workspace = workspace[:0]
    elif kind == "misaligned":
        workspace = workspace[1 : size + 1]
    elif kind in ("input_alias", "output_alias"):
        tensor = tensors["state"] if kind == "input_alias" else ht
        # The aliased operand fits inside the workspace allocation.
        bindings[tensor] = workspace[: x["state"].numel() * 4].view(torch.float32).view_as(x["state"])
    elif kind == "cpu":
        workspace = workspace.cpu()
    elif kind == "other_device":
        if torch.cuda.device_count() < 2:
            pytest.skip("requires a second device for workspace rejection")
        workspace = workspace.to("cuda:1")
    else:
        workspace = torch.empty(size * 2, device="cuda", dtype=torch.uint8)[::2]
    h.fill_(123)
    m.fill_(456)
    from cudnn.linear_attention.cake import compiler

    def no_launch(*args, **kwargs):
        raise AssertionError("invalid workspace reached CUDA")

    monkeypatch.setattr(compiler, "launch", no_launch)
    with pytest.raises((ValueError, RuntimeError), match="workspace|device|CUDA|cuda|contiguous"):
        graph.execute(bindings, workspace, handle=cudnn_handle)
    assert (h == 123).all() and (m == 456).all()


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_closed_form_reference_matches_recurrence(backward, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmark/linear_attention"))
    from benchmark_sm90_summary_tc import correlated_reference

    x = fp32.make_case((0, 17, 33), heads=2, gate=-1e-6)
    x["k"].copy_(x["k"][:1].clone().expand_as(x["k"]))
    expected = fp32.reference(x, backward)
    actual = correlated_reference(dict(x, cu_seqlens=x["cu"]), backward)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=1e-11, rtol=1e-11)
