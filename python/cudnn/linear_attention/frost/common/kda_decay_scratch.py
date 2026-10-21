# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time storage contract for the steep-decay kernels.

Lengths are elements of the region's declared type. Kernels allocate separate
16-byte-aligned arrays; nbytes budgets their aggregate storage. Stage lifetimes
and synchronization remain owned by each kernel's schedule.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScratchRegion:
    """One independently allocated region's element count and byte budget."""

    elements: int
    nbytes: int


@dataclass(frozen=True)
class SteepScratchLayout:
    """Disjoint correction and metadata budgets for one kernel schedule."""

    corr: ScratchRegion
    dadm: ScratchRegion
    last: ScratchRegion
    scan: ScratchRegion
    tile: ScratchRegion
    norm: ScratchRegion
    nbytes: int


def steep_scratch_layout(
    *,
    b_t: int,
    d_k: int,
    scan_warps: int,
    scan_stages: int,
    last_stages: int,
    decay_stages: int,
    norm_stages: int,
    dadm_stages: int = 0,
    io_bytes: int = 2,
) -> SteepScratchLayout:
    """Budget disjoint, 16-byte-aligned regions for each schedule's ring depths."""
    offset = 0

    def region(elements, width=4):
        nonlocal offset
        offset = (offset + 15) // 16 * 16
        result = ScratchRegion(elements, elements * width)
        offset += result.nbytes
        return result

    corr = region(decay_stages * 2 * b_t * b_t)
    dadm = region(dadm_stages * 2 * b_t * b_t, io_bytes)
    last = region(last_stages * d_k)
    scan = region(scan_stages * scan_warps)
    tile = region(decay_stages * scan_warps)
    norm = region(norm_stages * b_t)
    return SteepScratchLayout(corr, dadm, last, scan, tile, norm, (offset + 15) // 16 * 16)
