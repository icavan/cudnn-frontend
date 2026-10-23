# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA projection residuals: retain BF16 MMA accumulators until Y is formed.

The projection is already FP32. BF16 paths form V - projection and apply beta in FP32,
then round once into the existing packed Y operand. FP16 paths retain their
previous packed subtraction/multiplication semantics. No memory or pipeline
ownership changes: the caller still publishes the packed word as before.
"""

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, mul_f16x2, sub_f16x2


@cute.jit
def residual_pair(value_word, projection0=None, projection1=None, *, dtype: cutlass.Constexpr):
    """Return a residual pair for dBeta; BF16 residuals remain unrounded FP32."""
    if cutlass.const_expr(projection0 is None):
        return f16x2_to_f32(value_word, dtype=dtype)
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        v0, v1 = f16x2_to_f32(value_word, dtype=dtype)
        return v0 - projection0, v1 - projection1
    projection_word = fp32_to_fp16(projection0, projection1, dtype=dtype)
    return f16x2_to_f32(sub_f16x2(value_word, projection_word, dtype), dtype=dtype)


@cute.jit
def residual_operand(value_word, projection0=None, projection1=None, beta0=None, beta1=None, *, dtype: cutlass.Constexpr):
    """Pack Y after FP32 residual/beta arithmetic for BF16, or legacy FP16 math.

    value_word contains two io-dtype values; projections and optional beta
    values are FP32. Omit both beta values for prep-fed records, which already
    fold beta into their matrices. Omit projections for an unseeded first
    chunk; M summaries pass a packed zero value, not a constant float cast.
    """
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        r0, r1 = residual_pair(value_word, projection0, projection1, dtype=dtype)
        if cutlass.const_expr(beta0 is not None):
            r0, r1 = r0 * beta0, r1 * beta1
        return fp32_to_fp16(r0, r1, dtype=dtype)
    if cutlass.const_expr(projection0 is None):
        result = value_word
    else:
        projection_word = fp32_to_fp16(projection0, projection1, dtype=dtype)
        result = sub_f16x2(value_word, projection_word, dtype)
    if cutlass.const_expr(beta0 is not None):
        result = mul_f16x2(fp32_to_fp16(beta0, beta1, dtype=dtype), result, dtype)
    return result
