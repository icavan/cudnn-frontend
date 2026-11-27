# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared reduction-axis FP4 packing, including direct feature-to-token requantization."""

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from cudnn.frost.tile_dsl.pointwise import div_rn_f32, opaque_f32_bits
from ._math import NVFP4_INVERSE_SPLIT

# Keep FP32 6.0 in a register for div.rn.f32 on CuTeDSL 4.7.
_FP4_MAX_BITS = 0x40C00000


@cute.jit
def _forward_scale(amax, encode, bf16_input: cutlass.Constexpr):
    # TRT-LLM uses different reciprocal sequences at the two forward boundaries.
    if cutlass.const_expr(bf16_input):
        scaled_max = encode * (amax * cute.arch.rcp_approx(cutlass.Float32(6.0)))
    else:
        scaled_max = (amax * cutlass.Float32(1.0 / 6.0)) * encode
    rounded = cute.math.min(scaled_max, cutlass.Float32(448)).to(cutlass.Float8E4M3FN)
    scale = rounded.to(cutlass.Float32)
    inv = cutlass.Float32(0)
    if cutlass.const_expr(bf16_input):
        if amax != 0:
            inv = cute.arch.rcp_approx(scale * cute.arch.rcp_approx(encode))
    elif scale != 0:
        inv = cute.arch.fmin(encode * cute.arch.rcp_approx(scale), cutlass.Float32(3.4028234663852886e38))
    return rounded, inv


@cute.jit
def _backward_payload(values, encode, scale):
    if encode > cutlass.Float32(NVFP4_INVERSE_SPLIT):
        # Keep the reciprocal finite even at the smallest E4M3 scale.
        encode = encode * cutlass.Float32(1.0 / 512.0)
        values.store(values.load() * cutlass.Float32(512.0))
    inv = cutlass.Float32(0)
    if scale > 0.0:
        inv = div_rn_f32(encode, scale)
    packed = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
    packed.store((values.load() * inv).to(cutlass.Float4E2M1FN))
    return cute.recast_tensor(packed, cutlass.Int64)[0]


class PackTile:
    def __init__(self, width, dtype, token_axis, dequantize, *, forward=False, tiles=1):
        self.width = width
        self.tiles = tiles
        self.forward = forward
        self.bf16_input = dtype == torch.bfloat16
        self.token_axis = token_axis
        self.dequantize = dequantize
        self.rows, self.k = (width, 256) if token_axis else (256, width)

        def fake(dt, *shape):
            return make_fake_compact_tensor(dt, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        input_type = cutlass.Uint8 if dequantize else cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float32
        self.launch = cute.compile(
            self,
            fake(input_type, tiles * 256, width // 2 if dequantize else width),
            fake(cutlass.Float8E4M3FN, tiles * 256, width // 16) if dequantize else None,
            fake(cutlass.Float32, tiles) if dequantize else None,
            fake(cutlass.Float32, 1),
            fake(cutlass.Int64, tiles * self.rows, self.k // 16),
            fake(cutlass.Float8E4M3FN, tiles * self.rows, self.k // 16),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def quantize(self, x: cute.Tensor, source_sf, source_norm, norm: cute.Tensor, payload: cute.Tensor, sf: cute.Tensor):
        group = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        tile = cute.arch.block_idx()[1]
        if group < self.rows * (self.k // 16):
            if cutlass.const_expr(self.token_axis and self.width % 32 == 0):
                # A warp reads adjacent features of each route. Each lane owns
                # one 16-route reduction, preserving the scalar rounding order.
                row = (group // (32 * 16)) * 32 + group % 32
                block = (group // 32) % 16
            else:
                row = group // (self.k // 16)
                block = group % (self.k // 16)
            vals = cute.make_rmem_tensor((16,), cutlass.Float32)
            amax = cutlass.Float32(0)
            for i in cutlass.range_constexpr(16):
                col = block * 16 + i
                r, c = (col, row) if cutlass.const_expr(self.token_axis) else (row, col)
                r += tile * 256
                if cutlass.const_expr(self.dequantize):
                    byte = x[r, c // 2].to(cutlass.Int32)
                    nibble = (byte >> ((c % 2) * 4)) & 15
                    exp = (nibble & 7) >> 1
                    v = (nibble & 1).to(cutlass.Float32) * 0.5
                    if exp != 0:
                        v = ((2 + (nibble & 1)) * (1 << exp)).to(cutlass.Float32) * 0.25
                    if nibble >= 8:
                        v = -v
                    v = v * source_sf[r, c // 16].to(cutlass.Float32)
                    if source_norm[tile] != 1.0:
                        v = div_rn_f32(v, source_norm[tile])
                else:
                    v = x[r, c].to(cutlass.Float32)
                vals[i] = v
                amax = cute.math.max(amax, cute.math.abs(v))
            encode = norm[0]
            if cutlass.const_expr(self.forward):
                rounded, inv = _forward_scale(amax, encode, self.bf16_input)
                packed = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
                packed.store((vals.load() * inv).to(cutlass.Float4E2M1FN))
                word = cute.recast_tensor(packed, cutlass.Int64)[0]
            else:
                rounded = cute.math.min(div_rn_f32(amax * encode, opaque_f32_bits(_FP4_MAX_BITS)), cutlass.Float32(448)).to(cutlass.Float8E4M3FN)
                word = _backward_payload(vals, encode, rounded.to(cutlass.Float32))
            row += tile * self.rows
            sf[row, block] = rounded
            payload[row, block] = word

    quantize.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(self, x: cute.Tensor, source_sf, source_norm, norm: cute.Tensor, payload: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream):
        groups = self.rows * self.k // 16
        self.quantize(x, source_sf, source_norm, norm, payload, sf).launch(grid=((groups + 127) // 128, self.tiles, 1), block=(128, 1, 1), stream=stream)

    def run(self, x, norm, data, sf, *, source_sf=None, source_norm=None):
        self.launch(
            x,
            source_sf,
            source_norm,
            norm.reshape(1),
            data.view(torch.int64).view(self.tiles * self.rows, -1),
            sf.view(self.tiles * self.rows, -1),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


@cute.jit
def _gate_values(g, u, limit: cutlass.Constexpr):
    if cutlass.const_expr(limit is not None):
        g = cute.math.min(g, cutlass.Float32(limit))
        u = cute.math.max(cute.math.min(u, cutlass.Float32(limit)), cutlass.Float32(-limit))
    exponent = cute.math.exp2(g * cutlass.Float32(-1.4426950408889634), fastmath=True)
    sigmoid = cute.arch.rcp_approx(cutlass.Float32(1) + exponent)
    return g, u, sigmoid


@cute.jit
def _swiglu_grad(g, u, grad, prob, total, limit: cutlass.Constexpr):
    gc, uc, sigmoid = _gate_values(g, u, limit)
    silu = gc * sigmoid
    ds = grad * prob
    total += grad * ((uc * gc) * sigmoid)
    dg = (ds * uc) * (sigmoid + silu * (1.0 - sigmoid))
    du = ds * silu
    if cutlass.const_expr(limit is not None):
        if g >= limit:
            dg = cutlass.Float32(0.0)
        if cute.math.abs(u) >= limit:
            du = cutlass.Float32(0.0)
    return dg, du, total


class SwiGLUForward:
    def __init__(self, width, limit, device):
        self.width, self.limit = width, limit
        self.result = torch.empty((256, width), dtype=torch.float32, device=device)

        def fake(*shape):
            return make_fake_compact_tensor(cutlass.Float32, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        self.launch = cute.compile(
            self, fake(256, 2 * width), fake(256), fake(256, width), make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi"
        )

    @cute.kernel
    def kernel(self, pre: cute.Tensor, p: cute.Tensor, result: cute.Tensor):
        index = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        row, col = index // self.width, index % self.width
        g, u, sigmoid = _gate_values(pre[row, col], pre[row, self.width + col], self.limit)
        result[row, col] = ((u * g) * sigmoid) * p[row]

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(self, pre: cute.Tensor, p: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(pre, p, result).launch(grid=(256 * self.width // 128, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, pre, p):
        self.launch(pre, p, self.result, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return self.result


class SwiGLUBackward:
    """One CTA per route reduces intermediate-dimension partials in chunk order."""

    def __init__(self, width, limit, device, rows=256, *, record_field=None, fields=0):
        self.width, self.limit = width, limit
        self.rows, self.field = rows, record_field
        self.bound = record_field is not None
        self.chunks = (width + 255) // 256
        self.warps = min(8, self.chunks)
        self.dgu = torch.empty((rows, 2 * width), dtype=torch.float32, device=device)

        def fake(dtype, *shape):
            return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        f = cutlass.Float32
        self.launch = cute.compile(
            self,
            fake(cutlass.Int64, fields) if self.bound else fake(f, rows, 2 * width),
            fake(f, rows, width),
            fake(f, rows),
            fake(f, rows, 2 * width),
            fake(f, rows),
            fake(cutlass.Int64, 1) if self.bound else None,
            cutlass.Int32(0),
            fake(f, 1) if self.bound else None,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def kernel(self, pre, da, p, dgu, dp, base, offset, divisor):
        row = cute.arch.block_idx()[0]
        warp = cute.arch.thread_idx()[0] // 32
        lane = cute.arch.lane_idx()
        parts = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(self.chunks), byte_alignment=4)
        if cutlass.const_expr(self.bound):
            start = (base[0] + offset) * 256 * 2 * self.width
            ptr = cute.make_ptr(cutlass.Float32, pre[self.field], cute.AddressSpace.gmem) + start
            preact = cute.make_tensor(ptr, cute.make_layout((self.rows, 2 * self.width), stride=(2 * self.width, 1)))
        else:
            preact = pre
        for group in cutlass.range_constexpr((self.chunks + self.warps - 1) // self.warps):
            chunk = group * self.warps + warp
            if chunk < self.chunks:
                total = cutlass.Float32(0)
                for col_group in cutlass.range_constexpr(8):
                    col = chunk * 256 + col_group * 32 + lane
                    if col < self.width:
                        g = preact[row, col]
                        u = preact[row, col + self.width]
                        grad = da[row, col]
                        if cutlass.const_expr(self.bound):
                            if divisor[0] != 1.0:
                                grad = div_rn_f32(grad, divisor[0])
                        dg, du, total = _swiglu_grad(g, u, grad, p[row], total, self.limit)
                        dgu[row, col] = dg
                        dgu[row, col + self.width] = du
                total = cute.arch.warp_reduction_sum(total)
                if lane == 0:
                    parts[chunk] = total
        cute.arch.sync_threads()
        if cute.arch.thread_idx()[0] == 0:
            result = parts[0]
            for c in cutlass.range_constexpr(1, self.chunks):
                result += parts[c]
            dp[row] = result

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self, pre: cute.Tensor, da: cute.Tensor, p: cute.Tensor, dgu: cute.Tensor, dp: cute.Tensor, base, offset: cutlass.Int32, divisor, stream: cuda.CUstream
    ):
        self.kernel(pre, da, p, dgu, dp, base, offset, divisor).launch(grid=(self.rows, 1, 1), block=(self.warps * 32, 1, 1), stream=stream)

    def run(self, pre, da, p, dp, base=None, offset=0, divisor=None):
        self.launch(pre, da, p, self.dgu, dp, base, offset, divisor, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return self.dgu


@cute.jit
def _encode_block(values, norm):
    maximum = cute.math.abs(values.load()).reduce(cute.ReductionOp.MAX, cutlass.Float32(0), 0)
    scale = cute.math.min(div_rn_f32(maximum * norm, opaque_f32_bits(_FP4_MAX_BITS)), cutlass.Float32(448)).to(cutlass.Float8E4M3FN)
    return _backward_payload(values, norm, scale.to(cutlass.Float32)), scale


@cute.jit
def _stage_vectors(raw, sm, tid, width: cutlass.Constexpr):
    # Interleave the eight feature groups at 16-byte granularity. Both the
    # producer's vector stores and the transposed reader access adjacent banks.
    for v in cutlass.range_constexpr(16 // width):
        at = tid // 8 * 128 + (v * 8 + tid % 8) * width
        segment = cute.make_tensor(raw.iterator + v * width, cute.make_layout(width))
        cute.autovec_copy(segment, cute.make_tensor((sm.iterator + at).align(16), cute.make_layout(width)))


@cute.jit
def _staged_channel(tid, width: cutlass.Constexpr):
    return (tid // width % 8) * 16 + (tid // (8 * width)) * width + tid % width


class DualPack:
    """Build feature- and token-axis packs from local, peer, or SwiGLU gradient inputs."""

    def __init__(self, width, rows, dtype, device, bindings, data_name, sf_name, *, activation=None, peers=None, masked=False):
        self.masked = masked
        self.width, self.rows = width, rows
        self.data_field = bindings.names.index(data_name)
        self.scale_field = bindings.names.index(sf_name)
        self.dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float32
        self.vector = 8 if dtype == torch.bfloat16 else 4
        blocks = (width + 63) // 64
        self.payload = torch.empty((rows, width // 2), dtype=torch.uint8, device=device)
        allocate_scales = torch.zeros if width % 64 else torch.empty
        self.sf = allocate_scales((1, rows // 128, blocks, 32, 4, 4), dtype=torch.float8_e4m3fn, device=device)

        def flat(dt, size):
            return make_fake_compact_tensor(dt, (size,), stride_order=(0,), assumed_align=16)

        elements = rows * width
        if activation is not None:
            self.limit, self.pre_field, fields = activation
            self.chunks = width // 512
            self.partials = torch.empty((rows, self.chunks), dtype=torch.float32, device=device)
            self.fn = cute.compile(
                self.swiglu,
                flat(cutlass.Int64, fields),
                flat(cutlass.Float32, elements // 2),
                flat(cutlass.Float32, rows),
                make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4),
                flat(cutlass.Float32, 1),
                flat(cutlass.Int64, elements // 16),
                flat(cutlass.Float8E4M3FN, self.sf.numel()),
                flat(cutlass.Int64, bindings.count),
                flat(cutlass.Int64, 2 if masked else 1),
                cutlass.Int32(0),
                flat(cutlass.Float32, rows * self.chunks),
                flat(cutlass.Float32, rows),
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
            return
        peer_args = ()
        if peers is not None:
            ranks, self.tokens, self.top_k = peers
            peer_args = (flat(cutlass.Int64, rows), flat(cutlass.Float32, rows))
        self.fn = cute.compile(
            self,
            flat(self.dtype, elements) if peers is None else flat(cutlass.Int64, ranks),
            flat(cutlass.Float32, 1),
            flat(cutlass.Int64, elements // 16),
            flat(cutlass.Float8E4M3FN, self.sf.numel()),
            flat(cutlass.Int64, bindings.count),
            flat(cutlass.Int64, 2 if masked else 1),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            *peer_args,
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def swiglu_pack(self, pre, da, prob, divisor, norm, feature, fs, table, base, offset, partials):
        rb, chunk, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = cute.arch.lane_idx()
        row = rb * 16 + warp
        inner = self.width // 2
        tile_base = base[0] + offset
        pre_ptr = cute.make_ptr(cutlass.Float32, pre[self.pre_field], cute.AddressSpace.gmem)
        pre_ptr = pre_ptr + (tile_base * 256 + row) * self.width
        pitch = 260
        sm = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(2 * 16 * pitch), byte_alignment=16)
        active = True
        if cutlass.const_expr(self.masked):
            active = tile_base + rb // 16 < base[1]
        if active:
            total = cutlass.Float32(0)
            for part in cutlass.range_constexpr(8):
                col = chunk * 256 + part * 32 + lane
                g = cute.arch.load(pre_ptr + col, cutlass.Float32)
                u = cute.arch.load(pre_ptr + inner + col, cutlass.Float32)
                grad = da[row * inner + col]
                if divisor[0] != 1.0:
                    grad = div_rn_f32(grad, divisor[0])
                dg, du, total = _swiglu_grad(g, u, grad, prob[row], total, self.limit)
                at = warp * pitch + part * 32 + lane
                sm[at] = dg
                sm[16 * pitch + at] = du
            total = cute.arch.warp_reduction_sum(total)
            if lane == 0:
                partials[row * self.chunks + chunk] = total
            cute.arch.sync_threads()

            # Each quarter CTA packs a 16 x 128 fragment in both directions.
            # dG/dU move from FP32 registers to shared memory without a conversion.
            half = tid // 256
            panel = tid // 128 % 2
            thread = tid % 128
            first_col = half * inner + chunk * 256 + panel * 128
            frag = sm.iterator + half * 16 * pitch + panel * 128
            row = rb * 16 + thread // 8
            col = first_col + thread % 8 * 16
            vals = cute.make_rmem_tensor((16,), cutlass.Float32)
            ptr = frag + thread // 8 * pitch + thread % 8 * 16
            cute.autovec_copy(cute.make_tensor(ptr.align(16), cute.make_layout(16)), vals)
            packed, scale = _encode_block(vals, norm[0])
            feature[row * (self.width // 16) + col // 16] = packed
            sf_block = (row // 128) * (self.width // 64) + col // 64
            fs[sf_block * 512 + row % 32 * 16 + (row % 128 // 32) * 4 + col % 64 // 16] = scale
            for j in cutlass.range_constexpr(16):
                vals[j] = cute.arch.load(frag + j * pitch + thread, cutlass.Float32)
            packed, scale = _encode_block(vals, norm[0])
            at = ((tile_base + rb // 16) * self.width + first_col + thread) * 16 + rb % 16
            data = cute.make_ptr(cutlass.Int64, table[self.data_field], cute.AddressSpace.gmem)
            scales = cute.make_ptr(cutlass.Uint8, table[self.scale_field], cute.AddressSpace.gmem)
            cute.arch.store(data + at, packed)
            cute.arch.store(scales + at, scale.bitcast(cutlass.Uint8))
        elif lane == 0:
            partials[(rb * 16 + warp) * self.chunks + chunk] = 0.0

    swiglu_pack.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.kernel
    def finish_dp(self, partials, dp):
        row = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        total = partials[row * self.chunks]
        for c in cutlass.range_constexpr(1, self.chunks):
            total += partials[row * self.chunks + c]
        dp[row] = total

    finish_dp.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def swiglu(
        self,
        pre: cute.Tensor,
        da: cute.Tensor,
        prob: cute.Tensor,
        divisor: cute.Tensor,
        norm: cute.Tensor,
        feature: cute.Tensor,
        fs: cute.Tensor,
        table: cute.Tensor,
        base: cute.Tensor,
        offset: cutlass.Int32,
        partials: cute.Tensor,
        dp: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.swiglu_pack(pre, da, prob, divisor, norm, feature, fs, table, base, offset, partials).launch(
            grid=(self.rows // 16, self.chunks, 1), block=(512, 1, 1), stream=stream
        )
        self.finish_dp(partials, dp).launch(grid=(self.rows // 128, 1, 1), block=(128, 1, 1), stream=stream)

    def run_swiglu(self, pre, da, prob, divisor, norm, table, base, offset, dp):
        self.fn(
            pre,
            da.view(-1),
            prob,
            divisor,
            norm,
            self.payload.view(torch.int64).view(-1),
            self.sf.view(-1),
            table,
            base.reshape(-1),
            offset,
            self.partials.view(-1),
            dp,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def receive(self, ptrs, slots, prob, packed, norm, table, base, offset, stream):
        payload, sf = packed
        self.fn(ptrs, norm.view(1), payload.view(torch.int64).view(-1), sf.view(-1), table, base.reshape(-1), offset, stream, slots, prob)

    @cute.kernel
    def kernel(self, x, norm, feature, fs, table, base, offset, slots, prob):
        row_block, col_block, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        row = row_block * 16 + tid // 8
        col = col_block * 128 + (tid % 8) * 16
        sm = cutlass.utils.SmemAllocator().allocate_tensor(self.dtype, cute.make_layout(16 * 128), byte_alignment=16)
        active = True
        if cutlass.const_expr(self.masked):
            active = base[0] + offset + row_block // 16 < base[1]
        if active:
            vals = cute.make_rmem_tensor((16,), cutlass.Float32)
            if col < self.width:
                raw = cute.make_rmem_tensor((16,), self.dtype)
                if cutlass.const_expr(slots is None):
                    cute.autovec_copy(cute.make_tensor(x.iterator + row * self.width + col, cute.make_layout(16)), raw)
                else:
                    # Gather the BF16 source value once into registers for both
                    # pack directions. Padding contributes exactly zero to both.
                    slot = slots[row]
                    raw.fill(0.0)
                    p = cutlass.Float32(0)
                    if slot >= 0:
                        token = slot // self.top_k
                        peer = x[token // self.tokens]
                        ptr = cute.make_ptr(cutlass.BFloat16, peer, cute.AddressSpace.gmem)
                        src = ptr + token % self.tokens * self.width + col
                        cute.autovec_copy(cute.make_tensor(src.align(16), cute.make_layout(16)), raw)
                        if col == 0:
                            prob_ptr = cute.make_ptr(cutlass.Float32, peer + self.tokens * self.width * 2, cute.AddressSpace.gmem)
                            p = cute.arch.load(prob_ptr + slot % (self.top_k * self.tokens), cutlass.Float32)
                    if col == 0:
                        prob[row] = p
                vals.store(raw.load().to(cutlass.Float32))
                packed, scale = _encode_block(vals, norm[0])
                feature[row * (self.width // 16) + col // 16] = packed
                sf_block = (row // 128) * ((self.width + 63) // 64) + col // 64
                fs[sf_block * 512 + row % 32 * 16 + (row % 128 // 32) * 4 + col % 64 // 16] = scale
                _stage_vectors(raw, sm, tid, self.vector)
            cute.arch.sync_threads()
            # Consume shared memory in physical order, then undo the permutation
            # for the public token-axis payload and scale addresses.
            channel = col_block * 128 + _staged_channel(tid, self.vector)
            if channel < self.width:
                for j in cutlass.range_constexpr(16):
                    vals[j] = sm[j * 128 + tid].to(cutlass.Float32)
                packed, scale = _encode_block(vals, norm[0])
                at = ((base[0] + offset + row_block // 16) * self.width + channel) * 16 + row_block % 16
                data = cute.make_ptr(cutlass.Int64, table[self.data_field], cute.AddressSpace.gmem)
                scales = cute.make_ptr(cutlass.Uint8, table[self.scale_field], cute.AddressSpace.gmem)
                cute.arch.store(data + at, packed)
                cute.arch.store(scales + at, scale.bitcast(cutlass.Uint8))

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        norm: cute.Tensor,
        feature: cute.Tensor,
        fs: cute.Tensor,
        table: cute.Tensor,
        base: cute.Tensor,
        offset: cutlass.Int32,
        stream: cuda.CUstream,
        slots=None,
        prob=None,
    ):
        self.kernel(x, norm, feature, fs, table, base, offset, slots, prob).launch(
            grid=(self.rows // 16, (self.width + 127) // 128, 1), block=(128, 1, 1), stream=stream
        )

    def run(self, x, norm, table, base, offset):
        self.fn(
            x.view(-1),
            norm.view(1),
            self.payload.view(torch.int64).view(-1),
            self.sf.view(-1),
            table,
            base.reshape(-1),
            offset,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


class RepackSaved:
    """Convert a live prefix of forward E2M1 tiles directly into the token-axis bundle."""

    def __init__(self, width, norm_column):
        self.width, self.column = width, norm_column

        def flat(dt, length):
            return make_fake_compact_tensor(dt, (length,), stride_order=(0,), assumed_align=8)

        groups = cute.sym_int()
        self.fn = cute.compile(
            self,
            flat(cutlass.Int64, groups),
            flat(cutlass.Float8E4M3FN, groups),
            flat(cutlass.Int64, groups),
            flat(cutlass.Float8E4M3FN, groups),
            flat(cutlass.Float32, cute.sym_int()),
            flat(cutlass.Int64, cute.sym_int()),
            flat(cutlass.Float32, 1),
            flat(cutlass.Int64, 1),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def kernel(self, x, xs, data, sf, norms, experts, encode, live):
        tid = cute.arch.thread_idx()[0]
        groups = (self.width + 127) // 128
        job = cute.arch.block_idx()[0]
        limit = live[0] * (16 * groups)
        sm = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(16 * 128), byte_alignment=16)
        while job < limit:
            tile = job // (16 * groups)
            rb = job // groups % 16
            cb = job % groups
            row = tile * 256 + rb * 16 + tid // 8
            col = cb * 128 + tid % 8 * 16
            if col < self.width:
                group = row.to(cutlass.Int64) * (self.width // 16) + col // 16
                raw = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
                cute.recast_tensor(raw, cutlass.Int64)[0] = x[group]
                vals = cute.make_rmem_tensor((16,), cutlass.Float32)
                vals.store(raw.load().to(cutlass.Float32))
                decoded = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_scale = xs[group].to(cutlass.Float32)
                source_norm = norms[experts[tile] * 2 + self.column]
                if source_norm == 1.0:
                    decoded.store(vals.load() * block_scale)
                elif source_norm >= cutlass.Float32(2.0**-110) and source_norm <= cutlass.Float32(2.0**110):
                    # All nonzero E2M1 magnitudes are 1 or 1.5 times a power
                    # of two. Every nonzero decoded E4M3/E2M1 product here is
                    # normal and finite, so power-of-two scaling commutes with RN.
                    one = div_rn_f32(block_scale, source_norm)
                    one_and_half = div_rn_f32(1.5 * block_scale, source_norm)
                    # E2M1 has only one fraction bit. Clearing it in the FP32
                    # representation leaves the signed power of two (or signed zero).
                    bits = cute.recast_tensor(vals, cutlass.Uint32)
                    for j in cutlass.range_constexpr(16):
                        word = bits[j]
                        factor = cutlass.select_((word & 0x00400000) != 0, one_and_half, one)
                        decoded[j] = (word & cutlass.Uint32(0xFFBFFFFF)).bitcast(cutlass.Float32) * factor
                else:
                    for j in cutlass.range_constexpr(16):
                        decoded[j] = div_rn_f32(vals[j] * block_scale, source_norm)
                _stage_vectors(decoded, sm, tid, 4)
            cute.arch.sync_threads()
            channel = cb * 128 + _staged_channel(tid, 4)
            if channel < self.width:
                vals2 = cute.make_rmem_tensor((16,), cutlass.Float32)
                for j in cutlass.range_constexpr(16):
                    vals2[j] = sm[j * 128 + tid]
                packed, scale = _encode_block(vals2, encode[0])
                out = (tile.to(cutlass.Int64) * self.width + channel) * 16 + rb
                data[out] = packed
                sf[out] = scale
            cute.arch.sync_threads()
            job += cute.arch.grid_dim()[0]

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        data: cute.Tensor,
        sf: cute.Tensor,
        norms: cute.Tensor,
        experts: cute.Tensor,
        encode: cute.Tensor,
        live: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, xs, data, sf, norms, experts, encode, live).launch(grid=(1024, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, x, xs, data, sf, norms, experts, encode, live):
        self.fn(
            x.view(torch.int64).view(-1),
            xs.view(-1),
            data.view(torch.int64).view(-1)[: xs.numel()],
            sf.view(-1)[: xs.numel()],
            norms.view(-1),
            experts,
            encode.view(1),
            live.view(1),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
