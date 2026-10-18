# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA interval-decay range regressions against the FP64 token recurrence.

Gate inputs stay finite and nonpositive after activation. A large decay before
a write must not erase that write's interactions with subsequent tokens.
"""

import importlib
import math

import pytest
import torch

import cudnn
from cudnn.frost.buffers import cutedsl_requirement_error

from .reference_kda import kda_reference

requirement_error = cutedsl_requirement_error("KDA interval-decay tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

import cutlass

pytestmark = [pytest.mark.L1, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.fixture(scope="module", autouse=True)
def _available_frost_plan():
    """Gate the suite through one ordinary public call, not a hardware probe.

    Only this known-supported mild configuration may waive the module. A
    decline for any regression configuration below must remain a failure.
    """
    module = importlib.import_module("cudnn.linear_attention.ops.kda")
    actual, _, _ = _case("bf16_mild", 64, 16)
    q, k, v, g, beta = (t.detach()[0] for t in actual[:5])
    try:
        with torch.no_grad():
            module.kimi_delta_attention(q, k, v, g, beta, torch.tensor([0, 16], device="cuda", dtype=torch.int32), plan_name="kda_frost")
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"KDA FROST declined: {exc}")


def _case(pattern, dim, length=32, value_dim=None):
    """Construct rounded operands and matching differentiable FP64 leaves."""
    dtype = torch.float16 if pattern == "fp16_safe" else torch.bfloat16
    value_dim = value_dim or dim
    rng = torch.Generator().manual_seed(1644)
    q = (torch.randn(1, length, 2, dim, generator=rng) * 0.1).to(dtype).cuda()
    k = torch.nn.functional.normalize(torch.randn(1, length, 2, dim, generator=rng), dim=-1).to(dtype).cuda()
    v = (torch.randn(1, length, 2, value_dim, generator=rng) * 0.2).to(dtype).cuda()
    g = torch.full((1, length, 2, dim), 0.0 if pattern == "fp16_safe" else -0.1, device="cuda", dtype=torch.float32)
    if pattern == "bf16_spike":
        g[:, 5, :, 0] = -200.0
    beta = torch.full((1, length, 2), 0.25, device="cuda", dtype=dtype)
    state = (torch.randn(1, 2, value_dim, dim, generator=rng) * 0.01).cuda()
    a = torch.tensor([-0.1, 0.1], device="cuda")
    bias = torch.full((2, dim), 0.125, device="cuda")
    tensors = (q, k, v, g, beta, state, a, bias)
    actual = tuple(t.detach().clone().requires_grad_() for t in tensors)
    reference = tuple(t.detach().double().requires_grad_() for t in tensors)
    kw = dict(scale=0.375, safe_gate=pattern == "fp16_safe")
    return actual, reference, kw


@pytest.mark.parametrize(
    "pattern,dim,normalize,checkpoint",
    [
        pytest.param(pattern, dim, normalize, checkpoint, marks=(pytest.mark.L0,) if dim == 64 and normalize and checkpoint == 16 else ())
        for checkpoint in (0, 16, 32)
        for pattern in ("fp16_safe", "bf16_spike")
        for normalize in (False, True)
        for dim in (64, 128)
    ],
)
def test_kda_steep_decay_forward_backward(pattern, dim, normalize, checkpoint):
    """Check O, final state and every gradient, including safe-gate parameters."""
    _check_forward_backward(pattern, dim, normalize, checkpoint)


def _check_forward_backward(
    pattern,
    dim,
    normalize,
    checkpoint,
    *,
    length=32,
    batch_invariant=True,
    use_initial=True,
    gate_io_dtype=False,
    value_dim=None,
    operands=None,
    grad_scale=1.0,
    rms_tolerance=None,
):
    """Compare the public autograd path with a differentiable token recurrence."""
    module = importlib.import_module("cudnn.linear_attention.ops.kda")
    actual, reference, kw = _case(pattern, dim, length, value_dim) if operands is None else operands
    q, k, v, g, beta, state, a, bias = actual
    if gate_io_dtype:
        g = g.to(q.dtype).detach().requires_grad_()
        actual = (q, k, v, g, beta, state, a, bias)
        reference = tuple(t.detach().double().requires_grad_() for t in actual)
    qr, kr, vr, gr, br, sr, ar, dr = reference
    cu = torch.tensor([0, q.shape[1]], device="cuda", dtype=torch.int32)
    got = module.kimi_delta_attention(
        q[0],
        k[0],
        v[0],
        g[0],
        beta[0],
        cu,
        initial_state=state if use_initial else None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=normalize,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint,
        a_log=a if kw["safe_gate"] else None,
        dt_bias=bias if kw["safe_gate"] else None,
        plan_name="kda_frost",
        **kw,
    )
    if normalize:
        qr = qr * (qr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        kr = kr * (kr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    want = kda_reference(
        qr,
        kr,
        vr,
        gr,
        br,
        initial_state=sr if use_initial else None,
        a_log=ar if kw["safe_gate"] else None,
        dt_bias=dr if kw["safe_gate"] else None,
        **kw,
    )
    got = got[:2]
    weights = (torch.full_like(got[0], 0.125 * grad_scale), torch.full_like(got[1], 0.0625 * grad_scale))
    grads = torch.autograd.grad(got, actual, grad_outputs=weights, allow_unused=True)
    refs = torch.autograd.grad(want, reference, grad_outputs=(weights[0][None].double(), weights[1].double()), allow_unused=True)
    pairs = [("O", got[0], want[0][0]), ("final_state", got[1], want[1])]
    pairs += [(name, x, y) for name, x, y in zip(("dQ", "dK", "dV", "dG", "dBeta", "dState", "dA", "dBias"), grads, refs) if y is not None]
    failures = []
    for name, x, y in pairs:
        assert x is not None, name
        assert torch.isfinite(y).all(), f"nonfinite FP64 reference: {name}"
        finite = bool(torch.isfinite(x).all())
        rms = ((x.double() - y).square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-10)).item()
        maximum = (x.double() - y).abs().max().item()
        print(f"{pattern} D{dim} norm={normalize} {name}: finite={finite}, rms={rms:.6g}, max_abs={maximum:.6g}", flush=True)
        tolerance = (0.02 if name.startswith("d") else 0.01) if q.dtype == torch.bfloat16 else 0.002
        if rms_tolerance is not None:
            tolerance = rms_tolerance
        if not finite or rms > tolerance:
            failures.append(f"{name}: finite={finite}, rms={rms:.6g}")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_spike"])
@pytest.mark.parametrize("route", ["prep", "chain"])
def test_kda_steep_decay_schedules(pattern, route, monkeypatch):
    """Force the production split schedules on small ragged, ring-wrapping inputs."""
    engine = importlib.import_module("cudnn.linear_attention.frost.common.piece_chain")
    module = importlib.import_module("cudnn.linear_attention.ops.kda")
    monkeypatch.setattr(module, "fprop_cache", {})
    monkeypatch.setattr(module, "bprop_cache", {})
    if route == "chain":
        monkeypatch.setattr(engine, "choose_pieces", lambda **kw: (3, 1))
    else:
        monkeypatch.setattr(engine, "choose_pieces", lambda **kw: (0, 1))
        monkeypatch.setattr(engine, "is_dv_split", lambda **kw: True)
    _check_forward_backward(pattern, 128, True, 0, length=97, batch_invariant=False)


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_spike"])
def test_kda_direct_decay_without_initial_state(pattern):
    """The direct path also supports absent initial state and 16-bit gates."""
    _check_forward_backward(pattern, 64, True, 16, length=97, use_initial=False, gate_io_dtype=True)


def test_kda_direct_decay_cached_plan_rebinds_gate_values():
    """One cached direct plan must consume current gate values on every call."""
    for pattern in ("bf16_spike", "bf16_mild", "bf16_spike"):
        _check_forward_backward(pattern, 64, True, 16, length=97, use_initial=False, gate_io_dtype=True)


def test_kda_steep_decay_static_bounds():
    """Only a proven normalized BF16 safe-gate bound keeps factorization."""
    import cutlass
    from cudnn.linear_attention.frost.common.kda_decay import STEEP_DISABLED, STEEP_INLINE, default_steep_mode

    assert default_steep_mode(cutlass.BFloat16, True, True, -5 * 1.4426950408889634, 16) == STEEP_DISABLED
    assert default_steep_mode(cutlass.BFloat16, True, True, -6 * 1.4426950408889634, 16) == STEEP_INLINE
    assert default_steep_mode(cutlass.BFloat16, False, True, -5 * 1.4426950408889634, 16) == STEEP_INLINE
    assert default_steep_mode(cutlass.Float16, True, True, -5 * 1.4426950408889634, 16) == STEEP_INLINE
    assert default_steep_mode(cutlass.Float16, True, True, -5 * 1.4426950408889634, 16, state_alias=True) == STEEP_INLINE


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_mild"])
def test_kda_steep_decay_large_unnormalized_keys(pattern):
    """Direct intervals stay finite when factorized inverse keys would overflow."""
    actual, _, _ = _case(pattern, 64, 16)
    q, k, v, g, beta, state, a, bias = actual
    fp16 = q.dtype == torch.float16
    key = 8.0 if fp16 else 2.0**20
    with torch.no_grad():
        k.fill_(key)
        q.fill_(1.0 / key)
        g.fill_(-0.6 if fp16 else -4.7)
        # A power-of-two beta keeps the rank-one update well conditioned and
        # exactly representable while the inverse key itself exceeds range.
        beta.fill_(0.25 / (64 * key * key))
        state.zero_()
    operands = (actual, tuple(t.detach().double().requires_grad_() for t in actual), dict(scale=0.375, safe_gate=False))
    # Loss scaling keeps this FP16 fixture's small gradients out of underflow.
    # Large raw keys amplify the half-precision operand rounding. This range
    # regression has its own 0.3% FP16 budget; ordinary cases retain 0.2%.
    _check_forward_backward(pattern, 64, False, 16, length=16, operands=operands, grad_scale=128.0 if fp16 else 1.0, rms_tolerance=0.003 if fp16 else None)


def test_kda_non_safe_gate_combines_prefix_and_inverse_hazards():
    """One call repairs a steep prefix and a large-key inverse overflow."""
    actual, _, _ = _case("bf16_mild", 64, 16)
    q, k, v, g, beta, state, _, _ = actual
    key = 2.0**20
    with torch.no_grad():
        # Channel 0 crosses the prefix threshold with an ordinary key. The
        # remaining channels stay above that threshold, but their large keys
        # still push K * exp2(-prefix) beyond the BF16 operand range.
        q.fill_(1.0 / key)
        q[..., 0] = 1.0
        k.fill_(key)
        k[..., 0] = 1.0
        g.fill_(-4.7)
        g[..., 0] = -0.1
        g[:, 5, :, 0] = -200.0
        beta.fill_(0.25 / (1.0 + 63.0 * key * key))
        state.zero_()

    prefix_hazard = (-200.0 - 15.0 * 0.1) * 1.4426950408889634
    inverse_prefix = (16.0 * -4.7) * 1.4426950408889634
    assert prefix_hazard < -120.0
    assert inverse_prefix > -120.0
    assert math.log2(key) - inverse_prefix > math.log2(torch.finfo(torch.bfloat16).max)

    reference = tuple(t.detach().double().requires_grad_() for t in actual)
    operands = (actual, reference, dict(scale=0.375, safe_gate=False))
    _check_forward_backward("bf16_combined", 64, False, 16, length=16, operands=operands)


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_spike"])
@pytest.mark.parametrize("dim,value_dim", [(64, 128), (128, 64)])
def test_kda_steep_decay_rectangular(pattern, dim, value_dim):
    """Both native rectangular state layouts use the same direct intervals."""
    _check_forward_backward(pattern, dim, True, 0, value_dim=value_dim)


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_spike"])
def test_kda_steep_decay_overwrite_state(pattern):
    """The direct launch must read the original state before overwriting it."""
    module = importlib.import_module("cudnn.linear_attention.ops.kda")
    actual, reference, kw = _case(pattern, 64, 97)
    q, k, v, g, beta, state, a, bias = (t.detach() for t in actual)
    qr, kr, vr, gr, br, sr, ar, dr = reference
    qr = qr * (qr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    kr = kr * (kr.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    want = kda_reference(qr, kr, vr, gr, br, initial_state=sr, a_log=ar if kw["safe_gate"] else None, dt_bias=dr if kw["safe_gate"] else None, **kw)
    got = module.kimi_delta_attention(
        q[0],
        k[0],
        v[0],
        g[0],
        beta[0],
        torch.tensor([0, 97], device="cuda", dtype=torch.int32),
        initial_state=state,
        overwrite_initial_state=True,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        batch_invariant=True,
        plan_name="kda_frost",
        a_log=a if kw["safe_gate"] else None,
        dt_bias=bias if kw["safe_gate"] else None,
        **kw,
    )
    assert got[1].data_ptr() == state.data_ptr()
    for x, y in zip(got, (want[0][0], want[1])):
        assert torch.isfinite(x).all()
        error = (x.double() - y).square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-10)
        assert error < (0.01 if q.dtype == torch.bfloat16 else 0.002)


@pytest.mark.parametrize("operation", ["forward", "backward", "summary", "summary_transition", "summary_backward", "summary_backward_transition"])
def test_kda_steep_decay_native_state_rebinding(operation):
    """One warmed native host must support both aliased and disjoint state bindings."""
    from cudnn.frost.workspace import Workspace
    from cudnn.linear_attention.frost.kda_engine import build_kda, build_kda_summary

    actual, _, _ = _case("bf16_spike", 64, 97)
    q, k, v, g, beta, state, _, _ = (t.detach() for t in actual)
    values = dict(q=q[0], k=k[0], v=v[0], g=g[0], beta=beta[0], initial_state=state)
    values["cu_seqlens"] = torch.tensor([0, 97], device="cuda", dtype=torch.int32)
    summary = operation.startswith("summary")
    backward = "backward" in operation
    if backward:
        values.update(dO=torch.full_like(v[0], 0.125), d_final_state=torch.full_like(state, 0.0625))
    if summary:
        if backward:
            del values["v"], values["initial_state"]
        else:
            del values["q"]
    types = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float32: cudnn.data_type.FLOAT, torch.int32: cudnn.data_type.INT32}
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(t.shape), data_type=types[t.dtype], name=name) for name, t in values.items()}
    attrs = dict(scale=0.375, use_qk_l2norm=True, batch_invariant=True)
    if summary:
        if backward:
            graph.kda_summary_bwd(**ports, output_transition=True, **attrs)
            sources = dict(d_initial_state="d_final_state", transition="d_final_state")
        else:
            attrs.pop("scale")
            graph.kda_summary(**ports, output_transition=True, **attrs)
            sources = dict(final_state="initial_state", transition="initial_state")
    elif backward:
        graph.kda_bwd(**ports, **attrs)
        sources = dict(dQ="q", dK="k", dV="v", dG="g", dBeta="beta", d_initial_state="initial_state")
    else:
        graph.kda(**ports, output_final_state=True, **attrs)
        sources = dict(O="v", final_state="initial_state")
    outputs = {}
    for name, port in graph.nodes[0].outputs.items():
        dtype = values[sources[name]].dtype
        port.set_output(True).set_data_type(types[dtype])
        outputs[name] = torch.empty(tuple(port.dim), dtype=dtype, device="cuda")
    graph.validate()
    plan = (build_kda_summary if summary else build_kda)(graph)
    plan.bind(tuple(values) + tuple(outputs))
    scratch = torch.empty(plan.workspace_bytes(), dtype=torch.uint8, device="cuda")
    workspace = Workspace(scratch, plan.workspace_bytes(), "KDA alias rebinding")
    stream = torch.cuda.current_stream().cuda_stream
    plan.run(tuple(values.values()) + tuple(outputs.values()), workspace, stream)
    expected = {name: tensor.clone() for name, tensor in outputs.items()}
    input_name, output_name = ("d_final_state", "d_initial_state") if backward else ("initial_state", "final_state")
    if operation.endswith("transition"):
        output_name = "transition"
    original = values[input_name].clone()
    # Rebind the same plan to overlap, then back to independent buffers.
    for overlap in (True, False, True):
        values[input_name].copy_(original)
        bound = dict(outputs)
        if overlap:
            bound[output_name] = values[input_name]
        plan.run(tuple(values.values()) + tuple(bound.values()), workspace, stream)
        for name, tensor in bound.items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("pattern", ["fp16_safe", "bf16_spike"])
def test_kda_direct_decay_cuda_graph_rebind(pattern):
    """Captured forward/backward must consume rebound direct-path gates."""
    module = importlib.import_module("cudnn.linear_attention.ops.kda")
    actual, _, kw = _case(pattern, 64, 97)
    q, k, v, g, beta, _, a, bias = (t.detach() for t in actual)
    q, k, v, g, beta = q[0], k[0], v[0], g[0], beta[0]
    severe = g.clone()
    mild = torch.full_like(g, -30.0 if kw["safe_gate"] else -0.1)
    cu = torch.tensor([0, 97], device="cuda", dtype=torch.int32)
    do = torch.full_like(v, 0.125)
    dfs = torch.full((1, 2, 64, 64), 0.0625, device="cuda")
    attrs = dict(
        **kw,
        use_qk_l2norm_in_kernel=True,
        batch_invariant=True,
        checkpoint_every_n_tokens=16,
        a_log=a if kw["safe_gate"] else None,
        dt_bias=bias if kw["safe_gate"] else None,
        plan_name="kda_frost",
    )

    def step():
        out = module.kda_fwd(q, k, v, g, beta, cu, output_final_state=True, **attrs)
        grads = module.kda_bwd(do, q, k, v, g, beta, cu, d_final_state=dfs, state_checkpoints=out[2], **attrs)
        return out + grads

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for gate in (severe, mild, severe):
        g.copy_(gate)
        expected = tuple(t.clone() for t in step())
        graph.replay()
        for got, want in zip(captured, expected):
            assert torch.isfinite(want).all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
