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


def reference_pack(x, norm, *, raw=False):
    # Independent nearest-neighbor table, with the even code considered first on ties.
    levels = torch.tensor([0, 1, 2, 4, 0.5, 1.5, 3, 6], device=x.device)
    codes = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device, dtype=torch.uint8)
    blocks = x.float().unflatten(-1, (-1, 16))
    norm = torch.as_tensor(norm, dtype=torch.float32, device=x.device)
    # Round each FP32 division independently of the production CUDA path.
    sf = ((norm * blocks.abs().amax(-1)).double() / 6).float().clamp(max=448).to(torch.float8_e4m3fn)
    quotient = norm.double() / torch.where(sf.float() > 0, sf.double(), 1)
    # Round the reciprocal significand to FP32 without overflowing its exponent.
    mantissa, exponent = torch.frexp(quotient)
    inverse = torch.ldexp(mantissa.float().double(), exponent)
    scaled = (blocks.double() * torch.where(sf.float() > 0, inverse, 0)[..., None]).float()
    nearest = (scaled.abs().clamp_max(6)[..., None] - levels).abs().argmin(-1)
    code = (codes[nearest] | (torch.signbit(scaled).byte() << 3)).flatten(-2)
    dense = torch.copysign(levels[nearest], scaled) * sf.float()[..., None]
    if not raw:
        dense = (dense.double() / norm.double()).float()
    return code[..., ::2] | (code[..., 1::2] << 4), sf, dense.flatten(-2)


def physical_scales(sf):
    """Map [rows,K/16] into the SM100 128x4 scale atom in memory order."""
    rows, cols = sf.shape
    padded = F.pad(sf.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    return padded.reshape(-1, 4, 32, padded.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).contiguous().unsqueeze(0).view(torch.float8_e4m3fn)


def test_packing_boundaries():
    x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.0, -0.75, 0, 0, 0, 0, 0, 0], [0] * 16, [9000] * 16, [2.0**40, -(2.0**40)] * 8])
    data, sf = pack(x, 1.0)
    assert data[0, :4].tolist() == [0x20, 0x42, 0x64, 0x76]
    assert data[0, 4] == 0xA8
    assert data[3].tolist() == [0xF7] * 8
    assert sf.float().flatten().tolist() == [1, 0, 448, 448]
    expected, expected_sf, dense = reference_pack(x, 1.0)
    torch.testing.assert_close(data, expected)
    torch.testing.assert_close(sf.float(), expected_sf.float())
    torch.testing.assert_close(BlockScaledTensor(data, sf, MoeFormat.NVFP4, x.shape).dequantize(), dense)


@pytest.mark.parametrize("hidden", [160, 1056, 2080])
def test_peer_publish_divides_before_reduction(hidden):
    from cuda.bindings import driver as cuda
    from cudnn.moe_ep._nvfp4_peer import PeerKernels, TileExchange

    device = torch.device("cuda", torch.cuda.current_device())
    state = SimpleNamespace(max_tokens=2, h=hidden, k=2, topology=SimpleNamespace(ep_size=1), device=device, rows=256)
    exchange = TileExchange(state)
    # Any stream from the torch pool can become the parent of a conditional graph.
    parent_handles = {torch.cuda.Stream(device=device).cuda_stream for _ in range(64)}
    assert parent_handles.isdisjoint((exchange.get_stream.cuda_stream, exchange.put_stream.cuda_stream))
    _, publish = exchange.compile_transfer(PeerKernels(2, hidden, 2), 256, scale_dx=True)
    slots = torch.full((256,), -1, dtype=torch.int64, device=device)
    slots[:4] = torch.tensor([2, 1, 3, 0], device=device)
    dx = torch.randn((256, hidden), generator=torch.Generator(device=device).manual_seed(47), device=device)
    dp = torch.arange(256, device=device, dtype=torch.float32)
    denom = torch.tensor([3.7], device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exchange.flags.zero_()
        if exchange.drain is not None:
            exchange.done.zero_()
        publish(exchange.ptrs, slots, dx, dp, exchange.gen, denom, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        exchange.combine_ready(torch.cuda.current_stream())
    for norm in (3.7, 1.0):
        denom.fill_(norm)
        exchange.dx.fill_(float("nan"))
        exchange.combined.fill_(float("nan"))
        graph.replay()
        expected = dx[:4] / denom
        assert torch.equal(exchange.dx[slots[:4]].view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(exchange.dp.view(-1)[slots[:4]], dp[:4])
        combined = torch.stack((expected[3] + expected[1], expected[0] + expected[2])).bfloat16()
        assert torch.equal(exchange.combined.view(torch.uint8), combined.view(torch.uint8))


def test_peer_dgrad_store_replay():
    from cuda.bindings import driver as cuda
    from cudnn.moe_ep._nvfp4 import TileGemm, mma_operand
    from cudnn.moe_ep._nvfp4_peer import PeerDgradStore

    device = torch.device("cuda", torch.cuda.current_device())
    tiles, rows, hidden, inner = 2, 512, 1024, 512
    rng = torch.Generator(device=device).manual_seed(179)
    op = TileGemm(256, hidden, inner, device, expert_weights=True)
    op.prepare_dgrad(2, hidden, inner, device)
    local = op.compile_dgrad(2, hidden, inner, tiles, live=True)
    remote = op.compile_dgrad(2, hidden, inner, tiles, live=True, peer_store=PeerDgradStore(2, rows // 2, hidden))
    values = mma_operand(torch.randint(256, (rows, inner // 2), device=device, dtype=torch.uint8, generator=rng))
    bank = torch.randint(256, (2, hidden, inner // 2), device=device, dtype=torch.uint8, generator=rng).view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    sf = torch.ones((1, rows // 128, *op.blocked[0].shape[2:]), device=device).to(torch.float8_e4m3fn)
    op.expert_sf.fill_(1.0)
    layout = torch.empty((1, rows, hidden), dtype=torch.float32, device=device).permute(1, 2, 0)
    reference = torch.empty_like(layout)
    planes = torch.empty((2, rows // 2, hidden), dtype=torch.float32, device=device)
    peers = torch.tensor([plane.data_ptr() for plane in planes], device=device)
    owners = torch.tensor([1, 0], device=device)
    bounds = torch.tensor([0, 2], device=device)
    slots = torch.empty(rows, device=device, dtype=torch.int64)
    divisor = torch.ones(1, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        local(values, bank, sf, op.expert_sf, reference, op.amax, stream, owners, bounds, 0)
        remote(values, bank, sf, op.expert_sf, layout, op.amax, stream, owners, bounds, 0, (peers, slots, divisor))
    for live, scale in ((2, 1.0), (1, 3.7), (0, 2.0), (2, 4.3)):
        slots.copy_(torch.randperm(rows, device=device, generator=rng))
        slots[::7] = -1
        bounds[1].fill_(live)
        divisor.fill_(scale)
        planes.fill_(12345.0)
        layout.fill_(-17.0)
        graph.replay()
        expected = torch.full_like(planes.view(rows, hidden), 12345.0)
        active_slots = slots[: live * 256]
        valid = active_slots >= 0
        expected[active_slots[valid]] = reference[: live * 256, :, 0][valid] / divisor
        assert torch.equal(planes.view(torch.uint8).reshape(rows, -1), expected.view(torch.uint8))
        assert (layout == -17.0).all()


def test_peer_drain_crosses_int32_element_offset():
    from cudnn.moe_ep._nvfp4_peer import PeerKernels, TileExchange

    h, k = 8192, 8
    tokens = 2**31 // (h * k) + 1
    device = torch.device("cuda", torch.cuda.current_device())
    kernel = PeerKernels(tokens, h, k)
    if torch.cuda.mem_get_info(device)[0] < kernel.nbytes + 2**30:
        pytest.skip("the FP32 route plane crosses the 8 GiB boundary")
    state = SimpleNamespace(max_tokens=tokens, h=h, k=k, topology=SimpleNamespace(ep_size=1), device=device, rows=256)
    exchange = TileExchange(state)
    exchange.flags[-2 * kernel.chunks :].fill_(k)
    last = exchange.dx[-2 * k :].view(2, k, h)
    slots = torch.arange(2 * k, dtype=torch.float32, device=device).view(2, k, 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exchange.done.zero_()
        exchange.combine_ready(torch.cuda.current_stream())
    for bias in (1, 17):
        last.copy_(slots + bias)
        exchange.combined[-2:].fill_(float("nan"))
        graph.replay()
        assert torch.equal(exchange.combined[-2:], last.sum(1).bfloat16())


@pytest.mark.parametrize("limit", [None, 2.0])
def test_fused_swiglu_packs_and_dp(limit):
    from cudnn.moe_ep._nvfp4_pack import DualPack, SwiGLUBackward

    # Nine chunks exercise the ordered reduction across multiple warp groups.
    rows, width = 256, 9 * 256
    device = torch.device("cuda", torch.cuda.current_device())
    rng = torch.Generator(device=device).manual_seed(17)
    pre = torch.randn((2 * rows, 2 * width), device=device, generator=rng)
    pre[rows:, :2] = 2.0
    pre[rows:, width : width + 2] = -2.0
    pre_ptr = torch.tensor([pre.data_ptr()], device=device)
    da = torch.randn((rows, width), device=device, generator=rng)
    prob = torch.rand(rows, device=device, generator=rng)
    prob[::3] = 0
    divisor = torch.tensor([3.7], device=device)
    norm = torch.tensor([2.3], device=device)
    base = torch.zeros(1, dtype=torch.int64, device=device)
    args = (2 * width, rows, torch.float32, device, SimpleNamespace(names=("data", "sf"), count=2), "data", "sf")
    split = DualPack(*args)
    fused = DualPack(*args, activation=(limit, 0, 1))
    outputs = []
    for kernel in (split, fused):
        # The first tile is a canary for pointer-table base/offset addressing.
        data = torch.full((2, 2 * width, rows // 2), 0x5A, dtype=torch.uint8, device=device)
        sf = torch.full((2, 2 * width, rows // 16), 0xA5, dtype=torch.uint8, device=device)
        ptr = torch.tensor([data.data_ptr(), sf.data_ptr()], device=device)
        dp = torch.empty_like(prob)
        if kernel is split:
            gate = SwiGLUBackward(width, limit, device, rows, record_field=0, fields=1)
            gu = gate.run(pre_ptr, da, prob, dp, base, 1, divisor)
            kernel.run(gu, norm, ptr, base, 1)
        else:
            kernel.run_swiglu(pre_ptr, da, prob, divisor, norm, ptr, base, 1, dp)
        outputs.append((kernel.payload, kernel.sf, data, sf, dp))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_partial_window_replay():
    from cuda.bindings import driver as cuda
    from cudnn.moe_ep._nvfp4_schedule import WindowRoutes

    device = torch.device("cuda", torch.cuda.current_device())
    routes = WindowRoutes(SimpleNamespace(names=("unused", "route_slots"), count=2))
    saved = torch.arange(3 * 256, device=device)
    table = torch.tensor([0, saved.data_ptr()], device=device)
    owners = torch.tensor([7, 5, 3], device=device)
    bounds = torch.tensor([1, 3], device=device)
    slots = torch.empty(4 * 256, device=device, dtype=torch.int64)
    experts = torch.empty(4, device=device, dtype=torch.int64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routes.fn(table, owners, bounds, slots, experts, 4, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for begin, end in ((0, 3), (2, 3), (3, 3), (0, 1)):
        bounds.copy_(bounds.new_tensor([begin, end]))
        graph.replay()
        live = end - begin
        assert torch.equal(slots[: live * 256], saved[begin * 256 : end * 256])
        assert torch.equal(experts[:live], owners[begin:end])
        assert (slots[live * 256 :] == -1).all()
        assert (experts[live:] == 0).all()


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


def oracle(x, dy, ids, p, master1, master2, dx_norm, a_norm, w1_norm, w2_norm, path, *, check_operands=None):
    e, _, h = master1.shape
    i = master2.shape[-1]
    route_y = torch.zeros((*p.shape, h), device=x.device)
    route_x = torch.zeros_like(route_y)
    dp = torch.zeros_like(p)
    dw1, dw2 = torch.zeros_like(master1), torch.zeros_like(master2)
    saved = []

    def q(v, n):
        return reference_pack(v, n)[2]

    def backward_gemm(a, b, divisor):
        # The GEMM consumes raw block-scaled operands. Restore both global
        # multipliers after the FP32 accumulation, as the training contract specifies.
        return ((a @ b.T).double() / x.new_tensor(divisor, dtype=torch.float32).double()).float()

    if check_operands is not None:
        # The production SwiGLU uses Blackwell's approximate sigmoid. A single
        # FP32 ulp at an E2M1 midpoint can change an entire subsequent dX row.
        # Reproduce just the scalar PTX operations independently of FE kernels.
        from torch.cuda.jiterator import _create_jit_fn

        sigmoid_ref = _create_jit_fn("""template <typename T> T moe_sigmoid(T g) {
                float e, s;
                asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(g * -1.4426950408889634f));
                asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(1.0f + e));
                return s;
            }""")

    for expert in range(e):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        for begin in range(0, len(tokens), 256):
            t, k = tokens[begin : begin + 256], slots[begin : begin + 256]

            def pad(value):
                return F.pad(value, (0, 0, 0, 256 - len(t))).float()

            xp = reference_pack(pad(x[t]), dx_norm)
            pre = xp[2] @ q(master1[expert], w1_norm).T
            g, u = pre.chunk(2, -1)
            limit = path.gate_up_clamp
            gc, uc = (g, u) if limit is None else (g.clamp(max=limit), u.clamp(-limit, limit))
            if check_operands is None:
                s = F.silu(gc) * uc
            else:
                sig = sigmoid_ref(gc)
                s = (uc * gc) * sig
            prob = pad(p[t, k, None])
            ap = reference_pack(prob * s, a_norm[expert])
            y = ap[2] @ q(master2[expert], w2_norm).T
            route_y[t, k] = y[: len(t)].bfloat16().float()
            yt = pad(dy[t])
            da = backward_gemm(reference_pack(yt, path.dy_norm, raw=True)[2], reference_pack(master2[expert].T, 2.0, raw=True)[2], path.dy_norm * 2.0)
            dp[t, k] = (da * s).sum(1)[: len(t)]
            if check_operands is None:
                with torch.enable_grad():
                    gv, uv = gc.detach().requires_grad_(), uc.detach().requires_grad_()
                    dg, du = torch.autograd.grad(F.silu(gv) * uv, (gv, uv), grad_outputs=prob * da)
            else:
                silu = gc * sig
                dg = ((prob * da) * uc) * torch.addcmul(sig, silu, 1.0 - sig)
                du = (prob * da) * silu
            if limit is not None:
                dg, du = torch.where(g < limit, dg, 0.0), torch.where(u.abs() < limit, du, 0.0)
            dgu = torch.cat((dg, du), 1)
            route_x[t, k] = backward_gemm(
                reference_pack(dgu, path.dgu_norm, raw=True)[2], reference_pack(master1[expert].T, 4.0, raw=True)[2], path.dgu_norm * 4.0
            )[: len(t)]
            gu_k, x_k, dy_k, a_k = (
                reference_pack(v.T, norm, raw=True)
                for v, norm in ((dgu, path.dgu_norm), (xp[2], path.x_wgrad_norm), (yt, path.dy_norm), (ap[2], path.a_wgrad_norm))
            )
            dw1[expert] += gu_k[2] @ x_k[2].T
            dw2[expert] += dy_k[2] @ a_k[2].T
            if check_operands is not None:
                check_operands(expert, begin, gu_k, x_k, dy_k, a_k)
            saved.append((t * ids.shape[1] + k, xp, ap, y))
    dw1 = (dw1.double() / x.new_tensor(path.dgu_norm * path.x_wgrad_norm, dtype=torch.float32).double()).float()
    dw2 = (dw2.double() / x.new_tensor(path.dy_norm * path.a_wgrad_norm, dtype=torch.float32).double()).float()
    # The source combines dX in slot order. A parallel torch.sum can put a
    # value on the other side of a BF16 midpoint for top-k greater than two.
    y = route_y[:, 0].clone()
    dx = torch.zeros_like(route_x[:, 0])
    for slot in range(ids.shape[1]):
        if slot:
            y.add_(route_y[:, slot])
        dx.add_(route_x[:, slot])
    return y.bfloat16(), dx.bfloat16(), dp, dw1, dw2, saved


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
    for malformed in (sf1.float(), sf1.reshape(4, -1), sf1[:, :-1]):
        with pytest.raises(ValueError, match="E4M3 scale carrier per expert"):
            MoeEpNvfp4ForwardWeights.from_trtllm(SimpleNamespace(**{**vars(view), "fc1_weight_sf": malformed}), fw.input_norm, input_channel_scale=None)
    actual = MoeEpNvfp4ForwardWeights.from_trtllm(view, fw.input_norm, input_channel_scale=None)
    for a, b in ((actual.fc1, fw.fc1), (actual.fc2, fw.fc2)):
        assert torch.equal(a.data, b.data) and torch.equal(a.scale.view(torch.uint8), b.scale.view(torch.uint8))
    with pytest.raises(ValueError, match="channel scale"):
        MoeEpNvfp4ForwardWeights.from_trtllm(view, fw.input_norm, input_channel_scale=torch.ones(64))


@pytest.mark.parametrize("count,h,i", [(n, 128, 128) for n in (15, 16, 17, 255, 256, 257)] + [(17, 32, 32), (17, 160, 288)])
def test_training(count, h, i, record_property):
    e = 3
    device = torch.device("cuda", torch.cuda.current_device())
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, gate_up_clamp=2.0, dy_norm=4.0, dgu_norm=8.0, x_wgrad_norm=2.0, a_wgrad_norm=4.0)
    if count == 16:
        path = replace(path, dy_norm=3.0, dgu_norm=5.0, x_wgrad_norm=1.25, a_wgrad_norm=7.0)
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
    if (count, h) == (17, 128):
        rec.activation_norms.fill_(2.0)
        for name in ("fc31_alpha", "fc2_alpha", "fc1_norm_const", "input_norm"):
            scale = rec.activation_norms[0, 0] if name == "input_norm" else rec.activation_norms[:, 0]
            with pytest.raises(ValueError, match=f"{name} must not alias"):
                op.training_forward(x, ids, p, weights=replace(fw, **{name: scale}), out=rec)
        assert torch.all(rec.activation_norms == 2.0)
        shared = torch.ones(e, device=device)
        op.training_forward(x, ids, p, weights=replace(fw, fc31_alpha=shared, fc2_alpha=shared, input_norm=shared[0]), out=rec)
        # Checkpoint scales can occupy a column of a larger tensor or be broadcast.
        norm = torch.stack((shared, fw.fc1_norm_const), dim=1)[:, 1]
        alpha2 = torch.stack((fw.fc2_alpha, shared), dim=1)[:, 0]
        fw = replace(fw, fc1_norm_const=norm, fc31_alpha=fw.fc31_alpha[:1].expand(e), fc2_alpha=alpha2)
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
        for name, weight in (("dw1", dw1), ("dw2", dw2)):
            shifted = weight.new_empty(weight.numel() + 1)[1:].view_as(weight)
            grads = {"dw1": dw1, "dw2": dw2, name: shifted}
            with pytest.raises(ValueError, match="16-byte aligned"):
                op.training_wgrad(operands, **grads)
            with pytest.raises(ValueError, match="16-byte aligned"):
                op.training_backward(dy, ids, p, weights=bw, forward_record=rec, out=out, **grads)
        compact = op.compact_training_record(rec)
        assert compact.route_slots.numel() == int(rec.expert_offsets[-1]) < rec.route_slots.numel()
        assert all(getattr(compact, field.name).data_ptr() != getattr(rec, field.name).data_ptr() for field in fields(rec))
        for field in fields(rec):
            getattr(rec, field.name).zero_()
        snapshot_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(snapshot_graph):
            op.training_backward(dy, ids, p, weights=bw, forward_record=compact, out=out, dw1=dw1, dw2=dw2)
        snapshot_graph.replay()
        for result, reference in zip((compact.output, dx, dp, dw1, dw2), expected[:5]):
            torch.testing.assert_close(result, reference, rtol=3e-5, atol=3e-5)
        del snapshot_graph
        op.training_forward(x, ids, p, weights=fw, out=rec)
        # A contiguous view may start one byte into its allocation. Both this
        # bank and a strided bank must remain readable on every captured replay.
        left, right = bw.w1_transpose.data, bw.w2_transpose.data
        unaligned = left.new_empty(left.numel() + 1)[1:].view_as(left)
        strided = right.new_empty((*right.shape, 2))[..., 0]
        unaligned.copy_(left)
        strided.copy_(right)
        left_sf, right_sf = bw.w1_transpose.scale, bw.w2_transpose.scale
        offset_sf = left_sf.new_empty(left_sf.numel() + 1)[1:].view_as(left_sf)
        strided_sf = right_sf.new_empty((*right_sf.shape, 2))[..., 1]
        offset_sf.copy_(left_sf)
        strided_sf.copy_(right_sf)
        views = replace(
            bw, w1_transpose=replace(bw.w1_transpose, data=unaligned, scale=offset_sf), w2_transpose=replace(bw.w2_transpose, data=strided, scale=strided_sf)
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.training_backward(dy, ids, p, weights=views, forward_record=rec, out=out, dw1=dw1, dw2=dw2)
        graph.replay()
        for result, reference in zip((dx, dp, dw1, dw2), expected[1:5]):
            torch.testing.assert_close(result, reference, rtol=3e-5, atol=3e-5)
        offset_sf.zero_()
        graph.replay()
        assert torch.count_nonzero(dx) == 0
        offset_sf.copy_(left_sf)
        unaligned.zero_()
        graph.replay()
        assert torch.count_nonzero(dx) == 0
        del graph
    if count == 17:
        # Cover both forward implementations. A padding row that borrows
        # slot zero's negative probability would encode negative zero nibbles.
        p[0, 0] = -1.0
        op.training_forward(x, ids, p, weights=fw, out=rec)
        end = int(rec.expert_offsets[-1])
        padding = rec.route_slots[:end] < 0
        assert padding.any()
        assert not torch.count_nonzero(rec.saved_a[:end][padding])
        assert not torch.count_nonzero(rec.saved_sfa[:end].view(torch.uint8)[padding])
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


def test_deferred_wgrad_survives_forward_reuse():
    device = torch.device("cuda", torch.cuda.current_device())
    e, h, i, count = 2, 128, 128, 17
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4)
    op = MoeEp(
        MoeEpConfig(
            model=MoeEpModelConfig(num_experts=e, hidden_size=h, intermediate_size=i, top_k=1),
            parallel=MoeEpParallelConfig(max_tokens_per_rank=count),
            data_path=path,
        )
    )
    req = op.prepare_training(device=device)
    rec = buffers(req, MoeEpTrainingForwardOutputs, device)
    out = buffers(req, MoeEpTrainingBackwardOutputs, device)
    fw, bw, w1, w2 = fixture(e, h, i, device)
    gen = torch.Generator(device=device).manual_seed(17)
    x, dy = [torch.randn((count, h), device=device, dtype=torch.bfloat16, generator=gen) for _ in range(2)]
    ids = torch.zeros((count, 1), dtype=torch.int32, device=device)
    p = torch.rand((count, 1), generator=gen, device=device)
    dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
    kwargs = dict(weights=bw, forward_record=rec)

    def forward(expert):
        ids.fill_(expert)
        op.training_forward(x, ids, p, weights=fw, out=rec)

    forward(0)
    for bad, message in (
        (replace(out, wgrad_expert_offsets=rec.expert_offsets), "must not alias"),
        (replace(out, wgrad_valid_route_counts=None), "supply both"),
        (replace(out, wgrad_expert_offsets=out.wgrad_expert_offsets.int()), "dtype"),
    ):
        with pytest.raises(ValueError, match=message):
            op.training_backward(dy, ids, p, out=bad, **kwargs)

    expected = []
    for expert in range(e):
        forward(expert)
        _, _, operands = op.training_backward(dy, ids, p, out=out, dw1=dw1, dw2=dw2, **kwargs)
        assert torch.count_nonzero(dw1[expert]) and torch.count_nonzero(dw2[expert])
        ref = oracle(x, dy, ids, p, w1, w2, 4.0, fw.fc1_norm_const, 8.0, 4.0, path)
        for value, reference in zip((out.grad_activation, out.dprob, dw1, dw2), ref[1:5]):
            torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-5)
        expected.append((dw1.clone(), dw2.clone()))
        forward(1 - expert)
        op.training_wgrad(operands, dw1=dw1, dw2=dw2)
        for result, reference in zip((dw1, dw2), expected[-1]):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    other = buffers(req, MoeEpTrainingBackwardOutputs, device)
    for dest in (out, replace(out, wgrad_expert_offsets=None, wgrad_valid_route_counts=None)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, _, operands = op.training_backward(dy, ids, p, out=dest, **kwargs)
        for expert in (0, 1, 0):
            forward(expert)
            graph.replay()
            forward(1 - expert)
            op.training_backward(dy, ids, p, out=other, **kwargs)
            assert operands.valid_route_counts.tolist() == ([count, 0] if expert == 0 else [0, count])
            op.training_wgrad(operands, dw1=dw1, dw2=dw2)
            for result, reference in zip((dw1, dw2), expected[expert]):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        del graph
    op.close()


@pytest.mark.parametrize("norm", [1.0, 3.0, 7.0])
def test_master_weight_encoding_matches_device_midpoints(norm):
    from cudnn.moe_ep._nvfp4_pack import PackTile

    device = torch.device("cuda", torch.cuda.current_device())
    sf = torch.arange(127, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float()
    mids = sf.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    w = torch.zeros((256, 64), device=device)
    blocks = w.reshape(-1, 16)
    for offset, s in enumerate((sf[1:], (sf[:-1] + sf[1:]) / 2)):
        centers = torch.cat((s[:, None] * mids, (s * 6)[:, None]), dim=1) / norm
        for direction, x in enumerate(
            (torch.nextafter(centers, centers.new_tensor(-float("inf"))), centers, torch.nextafter(centers, centers.new_tensor(float("inf"))))
        ):
            start = (3 * offset + direction) * len(s)
            blocks[start : start + len(s)] = torch.cat((x, -x), dim=1)
    # Both public master weights are strided views; the reduction axis changes.
    transpose2 = w.flatten()[: 32 * 256].reshape(32, 256)
    bw = MoeEpNvfp4BackwardWeights.from_master(w.T[None], transpose2.T[None], w1_norm=norm, w2_norm=norm)
    for value, matrix in ((w, bw.w1_transpose), (transpose2, bw.w2_transpose)):
        expected, scale, _ = reference_pack(value, norm)
        assert torch.equal(matrix.data[0], expected)
        assert torch.equal(matrix.scale[0].view(torch.uint8), scale.view(torch.uint8))
        padded = F.pad(value, (0, 0, 0, 256 - len(value)))
        data = torch.empty((256, value.shape[1] // 2), dtype=torch.uint8, device=device)
        sf = torch.empty((256, value.shape[1] // 16), dtype=torch.float8_e4m3fn, device=device)
        PackTile(value.shape[1], torch.float32, False, False).run(padded, w.new_tensor([norm]), data, sf)
        assert torch.equal(data[: len(value)], expected)
        assert torch.equal(sf[: len(value)].view(torch.uint8), scale.view(torch.uint8))


def test_singleton_expert_views_and_wgrad_division():
    device = torch.device("cuda", torch.cuda.current_device())
    path = MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, dgu_norm=3.0, dy_norm=3.0)
    model = MoeEpModelConfig(num_experts=1, hidden_size=128, intermediate_size=128, top_k=1)
    op = MoeEp(MoeEpConfig(model=model, parallel=MoeEpParallelConfig(max_tokens_per_rank=5), data_path=path))
    req = op.prepare_training(device=device)
    rec, out = (buffers(req, cls, device) for cls in (MoeEpTrainingForwardOutputs, MoeEpTrainingBackwardOutputs))
    fw, bw, w1, w2 = fixture(1, 128, 128, device)
    x = torch.randn(5, 128, device=device, dtype=torch.bfloat16)
    dy = torch.ones_like(x)
    ids = torch.zeros(5, 1, device=device, dtype=torch.int32)
    prob = torch.ones(5, 1, device=device)

    def run(fw, bw):
        y = op.training_forward(x, ids, prob, weights=fw, out=rec)
        dx, dp, operands = op.training_backward(dy, ids, prob, weights=bw, forward_record=rec, out=out, dw1=w1, dw2=w2)
        return (y, dx, dp, w1, w2), operands

    def odd_stride(matrix):
        def view(t):
            return t.as_strided(t.shape, (1, *t.stride()[1:]))

        return replace(matrix, data=view(matrix.data), scale=view(matrix.scale))

    expected = [v.clone() for v in run(fw, bw)[0]]
    fw = replace(fw, fc1=odd_stride(fw.fc1), fc2=odd_stride(fw.fc2))
    bw = replace(bw, w1_transpose=odd_stride(bw.w1_transpose), w2_transpose=odd_stride(bw.w2_transpose))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, operands = run(fw, bw)
    graph.replay()
    for value, reference in zip(values, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    del graph

    # Five products of exactly one distinguish RN(5/3) from 5 * RN(1/3).
    for data, sf in (
        (operands.fc1_a, operands.fc1_sfa),
        (operands.fc1_b, operands.fc1_sfb),
        (operands.fc2_a, operands.fc2_sfa),
        (operands.fc2_b, operands.fc2_sfb),
    ):
        data.zero_()
        data[..., :2] = 0x22
        data[..., 2] = 0x02
        sf.fill_(1.0)
    op.training_wgrad(operands, dw1=w1, dw2=w2)
    for w in (w1, w2):
        assert torch.equal(w, torch.full_like(w, 5.0 / 3.0))
    with pytest.raises(ValueError, match="FC1 dgrad divisor"):
        run(fw, replace(bw, w1_norm=3e38))
    op.close()


@pytest.mark.parametrize("invalid", [1e-60, 1e80])
def test_encoding_norm_must_be_representable_in_fp32(invalid):
    with pytest.raises(ValueError, match="FP32"):
        MoeEpDataPathConfig(compute_format=MoeFormat.NVFP4, dy_norm=invalid)
    with pytest.raises(ValueError, match="w1_norm"):
        MoeEpNvfp4BackwardWeights.from_master(torch.empty(1, 64, 32), torch.empty(1, 32, 32), w1_norm=invalid)


def test_large_encoding_norm_preserves_zero_and_subnormal_inputs():
    from cudnn.moe_ep._nvfp4_pack import DualPack, PackTile

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.zeros((256, 32), device=device)
    x[:, :8] = x.new_tensor([1e-39, -1e-39, 5e-40, -5e-40, 2.5e-40, -2.5e-40, 0.0, -0.0])
    # SF is 9/512, the magnitudes round to 6, 3, 1.5 and signed zero.
    expected = torch.zeros((256, 16), dtype=torch.uint8, device=device)
    expected[:, :4] = expected.new_tensor([0xF7, 0xD5, 0xB3, 0x80])
    expected_sf = x.new_tensor([9 / 512, 0]).expand(256, 2)
    master, master_sf = pack(x, 1e38)
    data, sf = torch.empty_like(master), torch.empty_like(master_sf)
    PackTile(32, torch.float32, False, False).run(x, x.new_tensor([1e38]), data, sf)
    norm = x.new_tensor([1e38])
    dual = DualPack(32, 256, torch.float32, device, SimpleNamespace(names=("data", "sf"), count=2), "data", "sf")
    token_data = torch.empty((32, 128), device=device, dtype=torch.uint8)
    token_sf = torch.empty((32, 16), device=device, dtype=torch.float8_e4m3fn)
    ptrs = torch.tensor([token_data.data_ptr(), token_sf.data_ptr()], device=device)
    dual.run(x, norm, ptrs, torch.zeros(1, dtype=torch.int64, device=device), 0)
    assert torch.equal(dual.payload, expected)
    assert torch.equal(dual.sf.view(torch.uint8), physical_scales(expected_sf.to(torch.float8_e4m3fn)).view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_sf = pack(x.T, norm)
    for scale in (1.0, 1e38):
        norm.fill_(scale)
        graph.replay()
        expected_token, expected_token_sf = pack(x.T, scale)
        assert torch.equal(captured, expected_token) and torch.equal(captured_sf.float(), expected_token_sf.float())
    assert torch.equal(token_data, captured) and torch.equal(token_sf.float(), captured_sf.float())
    ref, ref_sf, _ = reference_pack(x, 1e38)
    for payload, scales in ((master, master_sf), (data, sf), (ref, ref_sf)):
        assert torch.equal(payload, expected)
        assert torch.equal(scales.float(), expected_sf)


def test_saved_activation_decode_before_token_axis_repack():
    from cudnn.moe_ep._nvfp4_pack import PackTile, RepackSaved

    device = torch.device("cuda", torch.cuda.current_device())
    data = torch.zeros((256, 16), dtype=torch.uint8, device=device)
    scales = torch.zeros((256, 2), dtype=torch.float8_e4m3fn, device=device)
    data[0].fill_(0x55)  # Two E2M1 threes per byte.
    data[1].fill_(0x77)
    scales[0].fill_(0.8125)
    scales[1].fill_(448)
    norms = torch.tensor([[7.0, 7.0]], device=device)
    encode = torch.tensor([321.6410217285156], device=device)
    owners = torch.zeros(1, device=device, dtype=torch.int64)
    live = torch.ones_like(owners)
    token = torch.empty((32, 128), dtype=torch.uint8, device=device)
    sf = torch.empty((32, 16), dtype=scales.dtype, device=device)
    # RN((3 * 0.8125) / 7) crosses the E2M1 midpoint after encoding.
    # Multiplying 3 by RN(0.8125 / 7) instead lands exactly on the tie.
    decoded = BlockScaledTensor(data, scales, MoeFormat.NVFP4, (256, 32)).dequantize()
    expected, expected_sf, _ = reference_pack((decoded.double() / 7).float().T, encode)
    assert torch.all(expected[:, 0] == 0x71)
    repack = RepackSaved(32, 1)
    scalar = PackTile(32, torch.uint8, True, True)
    for batched in (True, False):
        if batched:
            repack.run(data, scales, token, sf, norms, owners, encode, live)
        else:
            scalar.run(data, encode, token, sf, source_sf=scales, source_norm=norms[:, 0])
        assert torch.equal(token, expected)
        assert torch.equal(sf.view(torch.uint8), expected_sf.view(torch.uint8))

    # Sweep every finite positive E4M3 scale and all 16 payload nibbles,
    # including negative zero, in the fast path and near FP32 underflow.
    scale_bytes = scales.view(torch.uint8)
    positions = torch.arange(256, dtype=torch.int32, device=device)[:, None]
    nibbles = torch.arange(16, dtype=torch.uint8, device=device)
    data.copy_((nibbles | ((15 - nibbles) << 4)).expand_as(data))
    for source in (1.1, 7.0, 2.0**-110, 2.0**110, 2.0**126):
        norms.fill_(source)
        encode.fill_(source)
        scale_bytes.copy_((positions % 127).expand_as(scale_bytes))
        decoded = BlockScaledTensor(data, scales, MoeFormat.NVFP4, (256, 32)).dequantize()
        expected, expected_sf, _ = reference_pack((decoded.double() / norms[0, 1].double()).float().T, encode)
        repack.run(data, scales, token, sf, norms, owners, encode, live)
        assert torch.equal(token, expected)
        assert torch.equal(sf.view(torch.uint8), expected_sf.view(torch.uint8))


def test_two_cta_grouped_wgrad_reuses_tmem_across_expert_boundaries():
    from cudnn.moe_ep._nvfp4_wgrad import GroupedWgrad

    device = torch.device("cuda", torch.cuda.current_device())
    dim, tiles = 1024, 3
    ends = torch.zeros(2, device=device, dtype=torch.int32)
    owners = torch.zeros(tiles, device=device, dtype=torch.int64)
    op = GroupedWgrad(dim, dim, tiles * 256, 2, ends, device, expected_routes=1024, norm=1.0)
    rng = torch.Generator(device=device).manual_seed(53)

    def operand():
        data = torch.randint(256, (tiles, dim, 128), device=device, dtype=torch.uint8, generator=rng)
        sf = torch.randint(1, 4, (tiles, dim, 16), device=device, generator=rng).to(torch.float8_e4m3fn)
        dense = BlockScaledTensor(data, sf, MoeFormat.NVFP4, (tiles, dim, 256)).dequantize().transpose(0, 1).reshape(dim, -1)
        return data, sf, dense

    a, sa, x = operand()
    b, sb, y = operand()
    grad = torch.empty((2, dim, dim), device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.b.bind(b, sb, owners, ends)
        op.run(grad, a, sa, owners, ends)
    for first_end in (256, 0, tiles * 256):
        ends.copy_(ends.new_tensor([first_end, tiles * 256]))
        owners.copy_((torch.arange(tiles, device=device) >= first_end // 256).long())
        grad.fill_(float("nan"))
        graph.replay()
        for expert, (start, stop) in enumerate(((0, first_end), (first_end, tiles * 256))):
            # Small dyadic operands make this entire FP32 dot product exact.
            torch.testing.assert_close(grad[expert], x[:, start:stop] @ y[:, start:stop].T, rtol=0, atol=0)
