# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared reduction-axis FP4 packing, including direct feature-to-token requantization."""

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from cudnn.frost.tile_dsl.pointwise import div_rn_f32


class PackTile:
    def __init__(self, width, dtype, token_axis, dequantize, *, forward=False):
        self.width = width
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
            fake(input_type, 256, width // 2 if dequantize else width),
            fake(cutlass.Float8E4M3FN, 256, width // 16) if dequantize else None,
            fake(cutlass.Float32, 1) if dequantize else None,
            fake(cutlass.Float32, 1),
            fake(cutlass.Int64, self.rows, self.k // 16),
            fake(cutlass.Float8E4M3FN, self.rows, self.k // 16),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def quantize(self, x: cute.Tensor, source_sf, source_norm, norm: cute.Tensor, payload: cute.Tensor, sf: cute.Tensor):
        group = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if group < self.rows * (self.k // 16):
            row = group // (self.k // 16)
            block = group % (self.k // 16)
            vals = cute.make_rmem_tensor((16,), cutlass.Float32)
            amax = cutlass.Float32(0)
            for i in cutlass.range_constexpr(16):
                col = block * 16 + i
                r, c = (col, row) if cutlass.const_expr(self.token_axis) else (row, col)
                if cutlass.const_expr(self.dequantize):
                    byte = x[r, c // 2].to(cutlass.Int32)
                    nibble = (byte >> ((c % 2) * 4)) & 15
                    exp = (nibble & 7) >> 1
                    v = (nibble & 1).to(cutlass.Float32) * 0.5
                    if exp != 0:
                        v = ((2 + (nibble & 1)) * (1 << exp)).to(cutlass.Float32) * 0.25
                    if nibble >= 8:
                        v = -v
                    v = v * div_rn_f32(source_sf[r, c // 16].to(cutlass.Float32), source_norm[0])
                else:
                    v = x[r, c].to(cutlass.Float32)
                vals[i] = v
                amax = cute.math.max(amax, cute.math.abs(v))
            encode = norm[0]
            if cutlass.const_expr(self.forward):
                # TRT-LLM uses different reciprocal sequences at the two forward boundaries.
                if cutlass.const_expr(self.bf16_input):
                    scaled_max = encode * (amax * cute.arch.rcp_approx(cutlass.Float32(6.0)))
                else:
                    scaled_max = (amax * cutlass.Float32(1.0 / 6.0)) * encode
                rounded = cute.math.min(scaled_max, cutlass.Float32(448)).to(cutlass.Float8E4M3FN)
                scale = rounded.to(cutlass.Float32)
                inv = cutlass.Float32(0)
                if cutlass.const_expr(self.bf16_input):
                    if amax != 0:
                        inv = cute.arch.rcp_approx(scale * cute.arch.rcp_approx(encode))
                elif scale != 0:
                    inv = cute.arch.fmin(encode * cute.arch.rcp_approx(scale), cutlass.Float32(3.4028234663852886e38))
            else:
                rounded = cute.math.min(div_rn_f32(amax * encode, cutlass.Float32(6)), cutlass.Float32(448)).to(cutlass.Float8E4M3FN)
                scale = rounded.to(cutlass.Float32)
                inv = cutlass.Float32(0)
                if scale > 0.0:
                    inv = div_rn_f32(encode, scale)
            sf[row, block] = rounded
            packed = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
            packed.store((vals.load() * inv).to(cutlass.Float4E2M1FN))
            payload[row, block] = cute.recast_tensor(packed, cutlass.Int64)[0]

    @cute.jit
    def __call__(self, x: cute.Tensor, source_sf, source_norm, norm: cute.Tensor, payload: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream):
        groups = self.rows * self.k // 16
        self.quantize(x, source_sf, source_norm, norm, payload, sf).launch(grid=((groups + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, x, norm, data, sf, *, source_sf=None, source_norm=None):
        self.launch(x, source_sf, source_norm, norm.reshape(1), data.view(torch.int64), sf, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


@cute.jit
def _gate_values(g, u, limit: cutlass.Constexpr):
    if cutlass.const_expr(limit is not None):
        g = cute.math.min(g, cutlass.Float32(limit))
        u = cute.math.max(cute.math.min(u, cutlass.Float32(limit)), cutlass.Float32(-limit))
    exponent = cute.math.exp2(g * cutlass.Float32(-1.4426950408889634), fastmath=True)
    sigmoid = cute.arch.rcp_approx(cutlass.Float32(1) + exponent)
    return g, u, sigmoid


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

    @cute.jit
    def __call__(self, pre: cute.Tensor, p: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(pre, p, result).launch(grid=(256 * self.width // 128, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, pre, p):
        self.launch(pre, p, self.result, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return self.result


class SwiGLUBackward:
    """One warp per 256 intermediate features. Only the CTA's own route row is consumed."""

    def __init__(self, width, limit, device):
        self.width = width
        self.limit = limit
        chunks = (width + 255) // 256
        self.dgu = torch.empty((256, 2 * width), dtype=torch.float32, device=device)
        self.partial = torch.empty((chunks, 256), dtype=torch.float32, device=device)

        def fake(*shape):
            return make_fake_compact_tensor(cutlass.Float32, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)

        self.launch = cute.compile(
            self,
            fake(256, 2 * width),
            fake(256, width),
            fake(256),
            fake(256, 2 * width),
            fake(chunks, 256),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def kernel(self, pre: cute.Tensor, da: cute.Tensor, p: cute.Tensor, dgu: cute.Tensor, partial: cute.Tensor):
        warp = cute.arch.thread_idx()[0] // 32
        lane = cute.arch.thread_idx()[0] % 32
        rows, chunk, _ = cute.arch.block_idx()
        row = rows * 4 + warp
        total = cutlass.Float32(0.0)
        for col_group in cutlass.range_constexpr(8):
            col = chunk * 256 + col_group * 32 + lane
            if col < self.width:
                g = pre[row, col]
                u = pre[row, col + self.width]
                gc, uc, sigmoid = _gate_values(g, u, self.limit)
                silu = gc * sigmoid
                ds = da[row, col] * p[row]
                total += da[row, col] * ((uc * gc) * sigmoid)
                dg = (ds * uc) * (sigmoid + silu * (1.0 - sigmoid))
                du = ds * silu
                if cutlass.const_expr(self.limit is not None):
                    if g >= self.limit:
                        dg = cutlass.Float32(0.0)
                    if cute.math.abs(u) >= self.limit:
                        du = cutlass.Float32(0.0)
                dgu[row, col] = dg
                dgu[row, col + self.width] = du
        total = cute.arch.warp_reduction_sum(total)
        if lane == 0:
            partial[chunk, row] = total

    @cute.jit
    def __call__(self, pre: cute.Tensor, da: cute.Tensor, p: cute.Tensor, dgu: cute.Tensor, partial: cute.Tensor, stream: cuda.CUstream):
        self.kernel(pre, da, p, dgu, partial).launch(grid=(64, (self.width + 255) // 256, 1), block=(128, 1, 1), stream=stream)

    def run(self, pre, da, p, dp):
        self.launch(pre, da, p, self.dgu, self.partial, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        dp.copy_(self.partial[0])
        for j in range(1, self.partial.shape[0]):
            dp.add_(self.partial[j])
        return self.dgu
