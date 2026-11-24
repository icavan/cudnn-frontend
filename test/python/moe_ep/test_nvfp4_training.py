# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM 933366ab65d4e75085dd8d1c11b1898cdbadf286 forward and STE backward oracles."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cudnn.moe_ep import (
    BlockScaledTensor,
    MoeEp,
    MoeEpConfig,
    MoeEpDataPathConfig,
    MoeEpModelConfig,
    MoeEpParallelConfig,
    MoeFormat,
    MoeEpNvfp4ForwardWeights,
    MoeEpNvfp4BackwardWeights,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingBackwardOutputs,
)
from cudnn.moe_ep._nvfp4 import pack

pytestmark = pytest.mark.L0


def reference_pack(x, norm):
    # Independent nearest-neighbor table, with the even code considered first on ties.
    levels = torch.tensor([0, 1, 2, 4, 0.5, 1.5, 3, 6], device=x.device)
    codes = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device, dtype=torch.uint8)
    blocks = x.float().unflatten(-1, (-1, 16))
    sf = (norm * blocks.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn)
    scaled = blocks * torch.where(sf.float() > 0, norm / sf.float(), 0)[..., None]
    nearest = (scaled.abs()[..., None] - levels).abs().argmin(-1)
    code = (codes[nearest] | (torch.signbit(scaled).byte() << 3)).flatten(-2)
    dense = torch.copysign(levels[nearest], scaled) * (sf.float() / norm)[..., None]
    return code[..., ::2] | (code[..., 1::2] << 4), sf, dense.flatten(-2)


def physical_scales(sf):
    """Map [rows,K/16] into the SM100 128x4 scale atom in memory order."""
    rows, cols = sf.shape
    padded = F.pad(sf.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    return padded.reshape(-1, 4, 32, padded.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).contiguous().unsqueeze(0).view(torch.float8_e4m3fn)


def test_packing_boundaries():
    x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.0, -0.75, 0, 0, 0, 0, 0, 0], [0] * 16, [9000] * 16])
    data, sf = pack(x, 1.0)
    assert data[0, :4].tolist() == [0x20, 0x42, 0x64, 0x76]
    assert data[0, 4] == 0xA8
    assert sf.float().flatten().tolist() == [1, 0, 448]
    expected, expected_sf, dense = reference_pack(x, 1.0)
    torch.testing.assert_close(data, expected)
    torch.testing.assert_close(sf.float(), expected_sf.float())
    torch.testing.assert_close(BlockScaledTensor(data, sf, MoeFormat.NVFP4, x.shape).dequantize(), dense)


def test_swiglu_clamp_boundaries():
    from cudnn.moe_ep._nvfp4_pack import SwiGLUBackward

    device = torch.device("cuda", torch.cuda.current_device())
    kernel = SwiGLUBackward(32, 1.0, device)
    gate = torch.tensor([1.0, 0.0], device=device).repeat(256, 16)
    up = torch.tensor([-1.0, 1.0], device=device).repeat(256, 16)
    p = torch.ones(256, device=device)
    p[1] = 0
    da = torch.ones_like(gate)
    dp = torch.empty_like(p)
    dgu = kernel.run(torch.cat((gate, up), -1), da, p, dp)
    dg, du = dgu.chunk(2, -1)
    assert torch.count_nonzero(dg[:, ::2]) == 0
    assert torch.count_nonzero(du) == 0
    assert torch.count_nonzero(dg[1]) == 0
    torch.testing.assert_close(dp, (F.silu(gate) * up).sum(-1))


def oracle(x, dy, ids, p, master1, master2, dx_norm, a_norm, w1_norm, w2_norm, path):
    e, _, h = master1.shape
    i = master2.shape[-1]
    route_y = torch.zeros((*p.shape, h), device=x.device)
    route_x = torch.zeros_like(route_y)
    dp = torch.zeros_like(p)
    dw1, dw2 = torch.zeros_like(master1), torch.zeros_like(master2)
    saved = []

    def q(v, n):
        return reference_pack(v, n)[2]

    for expert in range(e):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        for begin in range(0, len(tokens), 256):
            t, k = tokens[begin : begin + 256], slots[begin : begin + 256]

            def pad(value):
                return F.pad(value, (0, 0, 0, 256 - len(t))).float()

            xp = reference_pack(pad(x[t]), dx_norm)
            pre = xp[2] @ q(master1[expert], w1_norm).T
            g, u = pre.chunk(2, -1)
            gc, uc = g.clamp(max=path.gate_up_clamp), u.clamp(-path.gate_up_clamp, path.gate_up_clamp)
            s = F.silu(gc) * uc
            prob = pad(p[t, k, None])
            ap = reference_pack(prob * s, a_norm[expert])
            y = ap[2] @ q(master2[expert], w2_norm).T
            route_y[t, k] = y[: len(t)].bfloat16().float()
            yt = pad(dy[t])
            da = q(yt, path.dy_norm) @ q(master2[expert].T, 2.0).T
            dp[t, k] = (da * s).sum(1)[: len(t)]
            # PyTorch autodiff for the activation, applying the specified open-boundary masks.
            with torch.enable_grad():
                gv, uv = gc.detach().requires_grad_(), uc.detach().requires_grad_()
                dg, du = torch.autograd.grad(F.silu(gv) * uv, (gv, uv), grad_outputs=prob * da)
            dgu = torch.cat((dg * (g < path.gate_up_clamp), du * (u.abs() < path.gate_up_clamp)), 1)
            route_x[t, k] = (q(dgu, path.dgu_norm) @ q(master1[expert].T, 4.0).T)[: len(t)]
            dw1[expert] += q(dgu.T, path.dgu_norm) @ q(xp[2].T, path.x_wgrad_norm).T
            dw2[expert] += q(yt.T, path.dy_norm) @ q(ap[2].T, path.a_wgrad_norm).T
            saved.append((t * ids.shape[1] + k, xp, ap, y))
    return route_y.sum(1).bfloat16(), route_x.sum(1).bfloat16(), dp, dw1, dw2, saved


def buffers(req, cls, device):
    return cls(**{f.name: torch.empty(req[f.name][0], device=device, dtype=req[f.name][2]) for f in fields(cls)})


def fixture(e, h, i, device):
    gen = torch.Generator(device=device).manual_seed(123)
    w1, w2 = [torch.randn(shape, generator=gen, device=device) / 8 for shape in ((e, 2 * i, h), (e, h, i))]

    def weight(value, norm):
        data, sf, _ = reference_pack(value, norm)
        return BlockScaledTensor(data, sf, MoeFormat.NVFP4, tuple(value.shape))

    norms = torch.arange(1, e + 1, device=device, dtype=torch.float32) * 2
    one = torch.ones(e, device=device)
    fw = MoeEpNvfp4ForwardWeights(weight(w1, 8), weight(w2, 4), one.new_tensor(4.0), norms, one / 32, 1 / (4 * norms))
    bw = MoeEpNvfp4BackwardWeights.from_master(w1, w2, w1_norm=4.0, w2_norm=2.0)
    return fw, bw, w1, w2


def test_trtllm_weight_view():
    fw, *_ = fixture(2, 64, 64, "cpu")

    def native(matrix, interleave):
        def reorder(x):
            return x.reshape(2, 2, -1, 16, x.shape[-1]).transpose(1, 2).reshape_as(x) if interleave else x

        return reorder(matrix.data), torch.stack([physical_scales(x).view(torch.uint8).flatten() for x in reorder(matrix.scale.view(torch.uint8))])

    data1, sf1 = native(fw.fc1, True)
    data2, sf2 = native(fw.fc2, False)
    view = SimpleNamespace(
        fc1_weight=data1,
        fc1_weight_sf=sf1,
        fc2_weight=data2,
        fc2_weight_sf=sf2,
        fc31_alpha=fw.fc31_alpha,
        fc2_alpha=fw.fc2_alpha,
        fc1_norm_const=fw.fc1_norm_const,
    )
    actual = MoeEpNvfp4ForwardWeights.from_trtllm(view, fw.input_norm, input_channel_scale=None)
    for a, b in ((actual.fc1, fw.fc1), (actual.fc2, fw.fc2)):
        assert torch.equal(a.data, b.data) and torch.equal(a.scale.view(torch.uint8), b.scale.view(torch.uint8))
    with pytest.raises(ValueError, match="channel scale"):
        MoeEpNvfp4ForwardWeights.from_trtllm(view, fw.input_norm, input_channel_scale=torch.ones(64))


@pytest.mark.parametrize("count,h,i", [(n, 128, 128) for n in (15, 16, 17, 255, 256, 257)] + [(17, 160, 288)])
def test_training(count, h, i, record_property):
    e = 3
    device = torch.device("cuda", torch.cuda.current_device())
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0, dy_norm=4.0, dgu_norm=8.0, x_wgrad_norm=2.0, a_wgrad_norm=4.0)
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=e, top_k=2, hidden_size=h, intermediate_size=i),
            parallel=MoeEpParallelConfig(max_tokens_per_rank=count),
            data_path=path,
        )
    )
    req = op.prepare_training(device=device)
    rec = buffers(req, MoeEpTrainingForwardOutputs, device)
    out = buffers(req, MoeEpTrainingBackwardOutputs, device)
    fw, bw, w1, w2 = fixture(e, h, i, device)
    gen = torch.Generator(device=device).manual_seed(456)
    x, dy = [torch.randn((count, h), device=device, dtype=torch.bfloat16, generator=gen) for _ in range(2)]
    ids = torch.tensor([0, 1], device=device, dtype=torch.int32).expand(count, -1).contiguous()
    ids[0] = 0
    ids[-1] = 1
    p = torch.rand((count, 2), device=device, generator=gen)
    p[0] = torch.tensor([0.0, 1.0], device=device)
    y = op.training_forward(x, ids, p, weights=fw, out=rec)
    if count == 257:
        other = buffers(req, MoeEpTrainingForwardOutputs, device)
        op.training_forward(-x, (ids + 1) % e, p * 0.5, weights=fw, out=other)
    dx, dp, operands = op.training_backward(dy, ids, p, weights=bw, forward_record=rec, out=out)
    dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
    op.training_wgrad(operands, dw1=dw1, dw2=dw2)
    expected = oracle(x, dy, ids, p, w1, w2, 4.0, fw.fc1_norm_const, 8.0, 4.0, path)
    for a, b in zip((y, dx, dp, dw1, dw2), expected[:5]):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    for tile, (idx, xp, ap, _) in enumerate(expected[-1]):
        row = slice(tile * 256, tile * 256 + len(idx))
        assert torch.equal(rec.route_slots[row], idx)
        for buf, val in ((rec.fc1_a, xp[0]), (rec.fc1_sfa, xp[1]), (rec.saved_a, ap[0]), (rec.saved_sfa, ap[1])):
            assert torch.equal(buf[row].view(torch.uint8), val[: len(idx)].view(torch.uint8))
    if (count, h) == (17, 128):
        # A contiguous view may start one byte into its allocation. Both this
        # bank and a strided bank must remain readable on every captured replay.
        left, right = bw.w1_transpose.data, bw.w2_transpose.data
        unaligned = left.new_empty(left.numel() + 1)[1:].view_as(left)
        strided = right.new_empty((*right.shape, 2))[..., 0]
        unaligned.copy_(left)
        strided.copy_(right)
        views = replace(bw, w1_transpose=replace(bw.w1_transpose, data=unaligned), w2_transpose=replace(bw.w2_transpose, data=strided))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.training_backward(dy, ids, p, weights=views, forward_record=rec, out=out, dw1=dw1, dw2=dw2)
        graph.replay()
        for result, reference in zip((dx, dp, dw1, dw2), expected[1:5]):
            torch.testing.assert_close(result, reference, rtol=3e-5, atol=3e-5)
        unaligned.zero_()
        graph.replay()
        assert torch.count_nonzero(dx) == 0
        del graph
    if count == 257:
        # BF16 matmuls, FP32 gate and routing math; compare the quantization loss
        # separately from the implementation-vs-W4A4 oracle checks above.
        xb = x.detach().requires_grad_()
        mb1, mb2 = [w.bfloat16().requires_grad_() for w in (w1, w2)]
        probs = p.detach().requires_grad_()
        y_ref = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(e):
            tokens, slots = (ids == expert).nonzero(as_tuple=True)
            g, u = F.linear(xb[tokens], mb1[expert]).float().chunk(2, -1)
            s = F.silu(g.clamp(max=2.0)) * u.clamp(-2.0, 2.0)
            z = F.linear((s * probs[tokens, slots, None]).bfloat16(), mb2[expert])
            y_ref.index_add_(0, tokens, z.float())
        grads = torch.autograd.grad(y_ref, (xb, probs, mb1, mb2), dy.float())
        for name, value, baseline in zip(("dx", "dp", "dw1", "dw2"), (dx, dp, dw1, dw2), grads):
            relative = ((value.float() - baseline.float()).norm() / baseline.float().norm()).item()
            record_property(f"bf16_relative_l2_{name}", relative)
    op.close()


@pytest.mark.parametrize("slots,overlap", [(1, False), (2, False), (2, True)])
def test_fc1_ring_wrap_and_replay(slots, overlap, monkeypatch):
    """Change expert order, wrap count and empty experts without moving any graph input."""
    device = torch.device("cuda", torch.cuda.current_device())
    t, h, i, e, k = 513, 160, 128, 4, 2
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0)
    config = MoeEpConfig(
        model=MoeEpModelConfig(num_experts=e, top_k=k, hidden_size=h, intermediate_size=i),
        parallel=MoeEpParallelConfig(max_tokens_per_rank=t),
        data_path=path,
        validation_mode="trusted",
    )
    full = MoeEp(config)
    ring = MoeEp(replace(config, data_path=replace(path, fc1_ring_slots=slots, fc1_ring_overlap=overlap)))
    full_req = full.prepare_training(device=device)
    req = ring.prepare_training(device=device)
    assert req["fc1_a"][0] == (t, h // 2) and req["fc1_sfa"][0] == (t, h // 16)
    fw, bw, w1, w2 = fixture(e, h, i, device)
    x, dy = [torch.randn(t, h, dtype=torch.bfloat16, device=device) for _ in range(2)]
    ids = torch.zeros(t, k, dtype=torch.int32, device=device)
    p = torch.rand(t, k, device=device)
    ref, rec = [buffers(spec, MoeEpTrainingForwardOutputs, device) for spec in (full_req, req)]
    grads = [buffers(spec, MoeEpTrainingBackwardOutputs, device) for spec in (full_req, req)]

    def observe_fc2(op, spec):
        routes_out = torch.empty((spec["route_slots"][0][0], h), dtype=torch.bfloat16, device=device)
        combine = op._training_state.combine

        def save(routes, *args, **kwargs):
            routes_out.copy_(routes)
            return combine(routes, *args, **kwargs)

        monkeypatch.setattr(op._training_state, "combine", save)
        return routes_out

    ref_fc2 = observe_fc2(full, full_req)
    ring_fc2 = observe_fc2(ring, req)
    graph = torch.cuda.CUDAGraph()
    ring.training_forward(x, ids, p, weights=fw, out=rec)
    from cutlass import cute

    def unexpected_compile(*args, **kwargs):
        pytest.fail("FC1 ring compiled a new kernel after preparation")

    monkeypatch.setattr(cute, "compile", unexpected_compile)
    captured_windows = []
    if overlap:
        # Delay each producer and the final FC1 feature reader before
        # retirement. Capture the delays only in the stressed graph.
        def delayed(fn):
            def run(*args, **kwargs):
                torch.cuda._sleep(100_000)
                result = fn(*args, **kwargs)
                torch.cuda._sleep(100_000)
                return result

            return run

        plan = ring._training_state.fc1_ring
        window = plan.window

        def observe_window(state, n):
            captured_windows.append(n)
            return window(state, n)

        monkeypatch.setattr(plan, "window", observe_window)
        monkeypatch.setattr(plan.source, "load", delayed(plan.source.load))
        monkeypatch.setattr(plan.gemm1.op, "execute", delayed(plan.gemm1.op.execute))
    with torch.cuda.graph(graph):
        ring.training_forward(x, ids, p, weights=fw, out=rec)
    if overlap:
        # A packet must wrap inside the conditional body to exercise per-slot retirement.
        assert 2 * slots in captured_windows
    row = torch.arange(t, device=device)
    for offset in (0, 1, 3):
        ids[:, 0] = (row // (t if offset == 0 else 129) + offset) % e
        ids[:, 1] = (ids[:, 0] + offset) % e
        x.neg_()
        full.training_forward(x, ids, p, weights=fw, out=ref)
        ring._training_state.fc1_ring.payload.fill_(255)
        ring._training_state.fc1_ring.sf.view(torch.uint8).fill_(127)
        graph.replay()
        valid = ref.route_slots >= 0
        src = ref.route_slots[valid] // k
        assert torch.equal(ref.route_slots, rec.route_slots)
        for name in ("fc1_a", "fc1_sfa"):
            assert torch.equal(getattr(ref, name).view(torch.uint8)[valid], getattr(rec, name).view(torch.uint8)[src])
        for name in ("fc1_preact", "saved_a", "saved_sfa"):
            lhs, rhs = (getattr(v, name) for v in (ref, rec))
            assert torch.equal(lhs.view(torch.uint8)[valid], rhs.view(torch.uint8)[valid])
        assert torch.equal(ref_fc2[valid], ring_fc2[valid])
        assert torch.equal(ref.output, rec.output)
    # Another microbatch wraps every slot before deferred backward consumes rec.
    other = buffers(req, MoeEpTrainingForwardOutputs, device)
    ring.training_forward(-x, (ids + 1) % e, p, weights=fw, out=other)
    results = []
    for op, record, grad in zip((full, ring), (ref, rec), grads):
        dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
        dx, dp, operands = op.training_backward(dy, ids, p, weights=bw, forward_record=record, out=grad)
        op.training_wgrad(operands, dw1=dw1, dw2=dw2)
        results.append((dx, dp, dw1, dw2))
    assert all(torch.equal(a, b) for a, b in zip(*results))
    del graph
    full.close()
    ring.close()


def test_megamoe_training_tape_replay_and_deferred_backward(monkeypatch):
    from cutlass import cute
    from cudnn.moe_ep import MoeEpNvfp4FusedForwardWeights

    device = torch.device("cuda", torch.cuda.current_device())
    t, h, i, e, k = 33, 256, 256, 2, 2
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0)
    config = MoeEpConfig(
        model=MoeEpModelConfig(num_experts=e, top_k=k, hidden_size=h, intermediate_size=i),
        parallel=MoeEpParallelConfig(max_tokens_per_rank=t),
        data_path=path,
        validation_mode="trusted",
    )
    reference = MoeEp(config)
    fused = MoeEp(replace(config, data_path=replace(path, nvfp4_fused_forward=True)))
    req_ref = reference.prepare_training(device=device)
    req = fused.prepare_training(device=device)
    logical, bw, w1, w2 = fixture(e, h, i, device)
    weights = MoeEpNvfp4FusedForwardWeights.from_logical(logical)
    x, dy = [torch.randn(t, h, device=device, dtype=torch.bfloat16) for _ in range(2)]
    ids = torch.zeros(t, k, device=device, dtype=torch.int32)
    prob = torch.rand(t, k, device=device)
    prob[0].zero_()
    rec, other = [buffers(req, MoeEpTrainingForwardOutputs, device) for _ in range(2)]
    ref = buffers(req_ref, MoeEpTrainingForwardOutputs, device)
    monkeypatch.setattr(cute, "compile", lambda *a, **kw: pytest.fail("recompiled prepared fused forward"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.training_forward(x, ids, prob, weights=weights, out=rec)
    for mode in (0, 1, 2):
        ids[:, 0].fill_(mode % 2)
        ids[:, 1].copy_((torch.arange(t, device=device) + mode) % e)
        x.neg_()
        rec.fc1_preact.fill_(float("nan"))
        rec.saved_sfa.view(torch.uint8).fill_(127)
        graph.replay()
        reference.training_forward(x, ids, prob, weights=logical, out=ref)
        mask = ref.route_slots >= 0
        assert torch.equal(ref.route_slots, rec.route_slots)
        for name in ("fc1_preact", "saved_a", "saved_sfa"):
            a, b = (getattr(r, name).view(torch.uint8) for r in (rec, ref))
            assert torch.equal(a[mask], b[mask]), name
        torch.testing.assert_close(rec.output, ref.output, rtol=0.01, atol=0.005)
    fused.training_forward(-x, 1 - ids, prob, weights=weights, out=other)
    grads = []
    for op, rec_i, spec in ((reference, ref, req_ref), (fused, rec, req)):
        out = buffers(spec, MoeEpTrainingBackwardOutputs, device)
        dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
        dx, dp, operand = op.training_backward(dy, ids, prob, weights=bw, forward_record=rec_i, out=out)
        op.training_wgrad(operand, dw1=dw1, dw2=dw2)
        grads.append((dx, dp, dw1, dw2))
    assert all(torch.equal(a, b) for a, b in zip(*grads))
    del graph
    fused.close()
    reference.close()
