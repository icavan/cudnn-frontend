# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert-grouped forward for one peer-mapped NVLink domain."""

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from ._nvfp4 import mma_operand
from ._nvfp4_pack import _forward_scale, _gate_values
from ._nvfp4_peer import PeerKernels


def _vector(dtype, length):
    return make_fake_compact_tensor(dtype, (length,), stride_order=(0,), assumed_align=dtype.width // 8)


def _checkpoint_scale():
    return cute.runtime.make_fake_tensor(cutlass.Float32, (cute.sym_int64(),), stride=(cute.sym_int64(),), assumed_align=4)


class ForwardPack:
    """Produce both the saved logical scales and the MMA scale layout in one pass."""

    def __init__(self, state, tiles, *, swiglu):
        self.swiglu = swiglu
        self.limit = state.path.gate_up_clamp
        self.width = state.i if swiglu else state.h
        self.h = state.h
        self.tokens, self.k = state.max_tokens, state.k
        self.rows = tiles * 256
        device = state.device
        self.data = torch.empty((self.rows, self.width // 2), dtype=torch.uint8, device=device)
        self.scales = torch.empty((1, self.rows // 128, self.width // 64, 32, 4, 4), dtype=torch.float8_e4m3fn, device=device)
        f, l, u = cutlass.Float32, cutlass.Int64, cutlass.Float8E4M3FN
        self.launch = cute.compile(
            self,
            _vector(l, state.topology.ep_size),
            _vector(l, cute.sym_int64()),
            _vector(l, tiles),
            _vector(l, 2),
            cutlass.Int32(0),
            _vector(f, self.rows),
            _checkpoint_scale(),
            _checkpoint_scale() if swiglu else None,
            _vector(f, self.rows * 2 * self.width) if swiglu else None,
            _vector(f, cute.sym_int64()) if swiglu else None,
            _vector(l, cute.sym_int64()),
            _vector(u, cute.sym_int64()),
            _vector(l, self.data.numel() // 8),
            _vector(u, self.scales.numel()),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def kernel(self, peers, slots, owners, bounds, base, prob, norm, alpha, acc, pre, saved_data, saved_sf, data, sf):
        tid = cute.arch.thread_idx()[0]
        job = cute.arch.block_idx()[0] * 128 + tid
        groups = self.width // 16
        live = min(self.rows, (bounds[1] - base) * 256)
        while job < live * groups:
            row, block = job // groups, job % groups
            global_row = cutlass.Int64(base) * 256 + row
            expert = owners[row // 256]
            vals = cute.make_rmem_tensor((16,), cutlass.Float32)
            if cutlass.const_expr(self.swiglu):
                scale = alpha[expert]
                p = prob[row]
                at = row * 2 * self.width + block * 16
                saved = global_row * 2 * self.width + block * 16
                gate = cute.make_rmem_tensor((16,), cutlass.Float32)
                up = cute.make_rmem_tensor((16,), cutlass.Float32)
                cute.autovec_copy(cute.make_tensor((acc.iterator + at).align(16), cute.make_layout(16)), gate)
                cute.autovec_copy(cute.make_tensor((acc.iterator + at + self.width).align(16), cute.make_layout(16)), up)
                gate.store(gate.load() * scale)
                up.store(up.load() * scale)
                cute.autovec_copy(gate, cute.make_tensor((pre.iterator + saved).align(16), cute.make_layout(16)))
                cute.autovec_copy(up, cute.make_tensor((pre.iterator + saved + self.width).align(16), cute.make_layout(16)))
                for j in cutlass.range_constexpr(16):
                    gc, uc, sigmoid = _gate_values(gate[j], up[j], self.limit)
                    vals[j] = ((uc * gc) * sigmoid) * p
                encode = norm[expert]
            else:
                slot = slots[global_row]
                source = max(slot, cutlass.Int64(0))
                token = source // self.k
                addr = peers[token // self.tokens]
                vals.fill(0.0)
                if slot >= 0:
                    x = cute.make_ptr(cutlass.BFloat16, addr, cute.AddressSpace.gmem)
                    src = x + (token % self.tokens) * self.h + block * 16
                    raw = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                    cute.autovec_copy(cute.make_tensor(src.align(16), cute.make_layout(16)), raw)
                    vals.store(raw.load().to(cutlass.Float32))
                if block == 0:
                    p = cutlass.Float32(0)
                    if slot >= 0:
                        pptr = cute.make_ptr(cutlass.Float32, addr + self.tokens * self.h * 2, cute.AddressSpace.gmem)
                        p = cute.arch.load(pptr + source % (self.tokens * self.k), cutlass.Float32)
                    prob[row] = p
                encode = norm[0]
            amax = cute.math.abs(vals.load()).reduce(cute.ReductionOp.MAX, cutlass.Float32(0), 0)
            rounded, inv = _forward_scale(amax, encode, not self.swiglu)
            packed = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
            packed.store((vals.load() * inv).to(cutlass.Float4E2M1FN))
            word = cute.recast_tensor(packed, cutlass.Int64)[0]
            data[job] = word
            saved_data[global_row * groups + block] = word
            saved_sf[global_row * groups + block] = rounded
            atom = row // 128 * (groups // 4) + block // 4
            sf[atom * 512 + row % 32 * 16 + row % 128 // 32 * 4 + block % 4] = rounded
            job += cute.arch.grid_dim()[0] * 128

    @cute.jit
    def __call__(self, peers, slots, owners, bounds, base: cutlass.Int32, prob, norm, alpha, acc, pre, saved_data, saved_sf, data, sf, stream: cuda.CUstream):
        self.kernel(peers, slots, owners, bounds, base, prob, norm, alpha, acc, pre, saved_data, saved_sf, data, sf).launch(
            grid=(min(1024, self.rows * self.width // (16 * 128)), 1, 1), block=(128, 1, 1), stream=stream
        )


class ForwardReturn:
    def __init__(self, state, tiles):
        self.h, self.k, self.slots = state.h, state.k, state.max_tokens * state.k
        self.rows = tiles * 256
        self.dx_offset = PeerKernels(state.max_tokens, state.h, state.k).dx
        self.multi_rank = state.topology.ep_size > 1
        self.publish = cute.compile(
            self.put,
            _vector(cutlass.Int64, state.topology.ep_size),
            _vector(cutlass.Int64, cute.sym_int64()),
            _vector(cutlass.Int64, tiles),
            _vector(cutlass.Int64, 2),
            cutlass.Int32(0),
            _vector(cutlass.Float32, self.rows * self.h),
            _checkpoint_scale(),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
        self.combine = cute.compile(
            self.reduce,
            _vector(cutlass.BFloat16, cute.sym_int64()),
            _vector(cutlass.BFloat16, cute.sym_int64()),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def scatter(self, peers, slots, owners, bounds, base, acc, alpha):
        at = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        words = self.h // 4
        live = min(self.rows, (bounds[1] - base) * 256)
        while at < live * words:
            row = at // words
            slot = slots[cutlass.Int64(base) * 256 + row]
            if slot >= 0:
                ptr = cute.make_ptr(cutlass.BFloat16, peers[slot // self.slots] + self.dx_offset, cute.AddressSpace.gmem)
                scale = alpha[owners[row // 256]]
                raw = cute.make_rmem_tensor((4,), cutlass.Float32)
                result = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                cute.autovec_copy(cute.make_tensor((acc.iterator + at * 4).align(16), cute.make_layout(4)), raw)
                result.store((raw.load() * scale).to(cutlass.BFloat16))
                dst = ptr + (slot % self.slots) * self.h + at % words * 4
                cute.autovec_copy(result, cute.make_tensor(dst.align(8), cute.make_layout(4)))
            at += cute.arch.grid_dim()[0] * 128

    @cute.jit
    def put(self, peers, slots, owners, bounds, base: cutlass.Int32, acc, alpha, stream: cuda.CUstream):
        self.scatter(peers, slots, owners, bounds, base, acc, alpha).launch(grid=(min(1024, self.rows * self.h // 512), 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def sum_slots(self, values, out, count):
        at = cutlass.Int64(cute.arch.block_idx()[0]) * 128 + cute.arch.thread_idx()[0]
        while at < cutlass.Int64(count) * self.h:
            row = cutlass.Int64(at) // self.h
            first = row * self.k * self.h + at % self.h
            total = values[first].to(cutlass.Float32)
            if cutlass.const_expr(self.multi_rank):
                # The reference sums each BF16 route with zero on the other ranks.
                if total == 0.0:
                    total = cutlass.Float32(0)
            for slot in cutlass.range_constexpr(1, self.k):
                total += values[first + slot * self.h].to(cutlass.Float32)
            out[at] = total.to(cutlass.BFloat16)
            at += cute.arch.grid_dim()[0] * 128

    @cute.jit
    def reduce(self, values, out, count: cutlass.Int32, stream: cuda.CUstream):
        self.sum_slots(values, out, count).launch(grid=(1024, 1, 1), block=(128, 1, 1), stream=stream)


class ExpertForward:
    """Two grouped GEMMs over a bounded window. Route counts remain device data."""

    def __init__(self, state):
        h, i, dev = state.h, state.i, state.device
        cap = state.rows // 256
        # Amortize launches on long batches while bounding the FP32 accumulator scratch.
        max_tiles = 128 if state.max_tokens * state.k >= 131072 else 32
        self.tiles = min(max_tiles, 1 << (cap.bit_length() - 1))
        rows = self.tiles * 256
        groups = (cap + self.tiles - 1) // self.tiles
        self.starts = torch.arange(groups * self.tiles, device=dev) * 256
        self.owners = torch.empty_like(self.starts)
        self.bounds = torch.zeros(2, device=dev, dtype=torch.int64)
        self.prob = torch.empty(rows, device=dev, dtype=torch.float32)
        self.x = ForwardPack(state, self.tiles, swiglu=False)
        self.act = ForwardPack(state, self.tiles, swiglu=True)
        self.ret = ForwardReturn(state, self.tiles)
        e = state.topology.experts_per_rank
        self.g1, self.g2 = (state.gemms[256, n, k] for n, k in ((2 * i, h), (h, i)))
        for gemm, n, k in ((self.g1, 2 * i, h), (self.g2, h, i)):
            if not hasattr(gemm, "expert_sf"):
                gemm.prepare_expert_bank(e, n, k, dev)
        self.fc1 = self.g1.compile_dgrad(e, 2 * i, h, self.tiles, live=True)
        self.fc2 = self.g2.compile_dgrad(e, h, i, self.tiles, live=True)
        self.acc1, self.acc2 = [torch.empty((1, rows, n), device=dev, dtype=torch.float32).permute(1, 2, 0) for n in (2 * i, h)]

    def run(self, exchange, x, p, weights, record):
        count = x.shape[0]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        exchange.dy[:count].copy_(x)
        exchange.prob[:count].copy_(p)
        if exchange.handle is not None:
            exchange.handle.barrier()
        torch.searchsorted(record.expert_offsets, self.starts, right=True, out=self.owners)
        torch.div(record.expert_offsets[-1], 256, rounding_mode="floor", out=self.bounds[1])
        bank1, bank2 = self.g1.bind_dgrad(weights.fc1), self.g2.bind_dgrad(weights.fc2)
        # The public checkpoint accepts strided scale views.
        norm1 = weights.input_norm.view(1)
        norm2, alpha1, alpha2 = weights.fc1_norm_const, weights.fc31_alpha, weights.fc2_alpha
        slots = record.route_slots

        def packs(data, sf):
            return data.view(torch.int64).view(-1), sf.view(-1)

        x_record = packs(record.fc1_a, record.fc1_sfa)
        a_record = packs(record.saved_a, record.saved_sfa)
        x_pack = packs(self.x.data, self.x.scales)
        a_pack = packs(self.act.data, self.act.scales)
        for base in range(0, self.starts.numel(), self.tiles):
            owners = self.owners[base : base + self.tiles]
            route = exchange.ptrs, slots, owners, self.bounds, base
            self.x.launch(*route, self.prob, norm1, None, None, None, *x_record, *x_pack, stream)
            self.fc1(mma_operand(self.x.data), bank1, self.x.scales, self.g1.expert_sf, self.acc1, self.g1.amax, stream, owners, self.bounds, base)
            self.act.launch(*route, self.prob, norm2, alpha1, self.acc1.view(-1), record.fc1_preact.view(-1), *a_record, *a_pack, stream)
            self.fc2(mma_operand(self.act.data), bank2, self.act.scales, self.g2.expert_sf, self.acc2, self.g2.amax, stream, owners, self.bounds, base)
            self.ret.publish(*route, self.acc2.view(-1), alpha2, stream)
        if exchange.handle is not None:
            exchange.handle.barrier()
        self.ret.combine(exchange.dx.view(torch.bfloat16).view(-1), record.output.view(-1), count, stream)
        return record.output[:count]
