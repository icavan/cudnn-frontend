# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counted CTA barriers for warp-specialized control flow."""

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm


@cute.jit
def sync_warps(barrier_id: cutlass.Constexpr, threads: cutlass.Constexpr):
    # PTX bar.sync is the aligned form. Only the named subset participates here.
    llvm.inline_asm(None, [], f"barrier.sync {barrier_id}, {threads};", "~{memory}", has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
