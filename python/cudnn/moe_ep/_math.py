# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric helpers shared by public contracts and private backends."""

from __future__ import annotations

import math
from ctypes import c_float
from numbers import Real

# E4M3's smallest positive scale is 2^-9. Above this multiplier, form
# (norm * 2^-9) / scale and compensate by multiplying the input by 2^9.
NVFP4_INVERSE_SPLIT = float.fromhex("0x1.fffffep+118")


def positive_fp32(value: Real, name: str) -> float:
    """Round a host encoding multiplier to its actual device representation."""
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            result = c_float(value).value
        except OverflowError:
            pass
        else:
            if math.isfinite(result) and result > 0:
                return result
    raise ValueError(f"{name} must be a finite positive FP32 encoding multiplier")


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


__all__ = ["ceil_div", "positive_fp32", "round_up"]
