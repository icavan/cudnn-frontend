# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the two-query union used by the SM100 H32 paired backward path.

The output index packs a two-bit membership mask into bits 29:30::

    encoded = kv_index | (membership << 29)

``membership & 1`` selects the first query and ``membership & 2`` selects
the second.  Keeping the mask beside the index avoids another global tensor
read in the sparse KV loader.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.utils.smem_allocator import SmemAllocator


@cute.jit
def _warp_scan_inclusive(value: Int32, lane: Int32) -> Int32:
    for log_step in cutlass.range_constexpr(5):
        step = 1 << log_step
        other = cute.arch.shuffle_sync_up(value, step, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= step:
            value = value + other
    return value


@cute.jit
def _block_scan_inclusive(value: Int32, warp_sums: cute.Tensor, tidx: Int32) -> Int32:
    """Inclusive scan across the fixed 256-thread preprocessing CTA."""
    warp = tidx // 32
    lane = tidx % 32
    value = _warp_scan_inclusive(value, lane)
    if lane == 31:
        warp_sums[warp] = value
    cute.arch.barrier(barrier_id=1, number_of_threads=256)

    if warp == 0:
        warp_value = Int32(0)
        if lane < 8:
            warp_value = warp_sums[lane]
        warp_value = _warp_scan_inclusive(warp_value, lane)
        if lane < 8:
            warp_sums[lane] = warp_value
    cute.arch.barrier(barrier_id=1, number_of_threads=256)

    if warp > 0:
        value = value + warp_sums[warp - 1]
    return value


class H32PairTopkUnion:
    """One CTA per adjacent pair of query tokens."""

    block_threads = 256

    def __init__(self, seqlen_kv: int, max_topk: int):
        self.seqlen_kv = int(seqlen_kv)
        self.max_topk = int(max_topk)
        self.max_union = min(self.seqlen_kv, 2 * self.max_topk)
        self.num_words = (self.seqlen_kv + 31) // 32

    @cute.kernel
    def kernel(
        self,
        topk_idxs: cute.Tensor,
        topk_length: Optional[cute.Tensor],
        union_idxs: cute.Tensor,
        union_length: cute.Tensor,
    ):
        pair_idx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        smem = SmemAllocator()
        membership_bits = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((2 * self.num_words,), order=(0,)),
            byte_alignment=128,
        )
        warp_sums = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=32,
        )
        control = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=16,
        )

        for word_idx in cutlass.range(tidx, 2 * self.num_words, self.block_threads):
            membership_bits[word_idx] = Int32(0)
        if tidx == 0:
            control[0] = Int32(0)
        cute.arch.sync_threads()

        # Mark membership in a dense CTA-local map. Top-k indices are unique
        # within one query, so OR is sufficient for both fixed and compact
        # top-k inputs.
        for query_in_pair in cutlass.range_constexpr(2):
            query_idx = pair_idx * 2 + query_in_pair
            valid = Int32(self.max_topk)
            if const_expr(topk_length is not None):
                valid = topk_length[query_idx]
            for pos in cutlass.range(tidx, self.max_topk, self.block_threads):
                if pos < valid:
                    kv_idx = topk_idxs[query_idx, pos]
                    if kv_idx >= 0 and kv_idx < self.seqlen_kv:
                        word_idx = kv_idx // 32
                        bit_idx = kv_idx % 32
                        nvvm.atomicrmw(
                            "or",
                            membership_bits.iterator + query_in_pair * self.num_words + word_idx,
                            Int32(1) << bit_idx,
                            space=nvvm.SharedSpace.shared_cta,
                        )
        cute.arch.sync_threads()

        # Compact one bitset word per thread. For the core 4K-KV shape all 128
        # words fit in a single block scan; larger supported contexts use a
        # small number of 256-word stripes.
        num_stripes = (self.num_words + self.block_threads - 1) // self.block_threads
        for stripe in cutlass.range_constexpr(num_stripes):
            word_idx = stripe * self.block_threads + tidx
            bits0 = Int32(0)
            bits1 = Int32(0)
            if word_idx < self.num_words:
                bits0 = membership_bits[word_idx]
                bits1 = membership_bits[self.num_words + word_idx]
            union_bits = bits0 | bits1
            count = cute.arch.popc(union_bits)
            rank = _block_scan_inclusive(count, warp_sums, tidx)
            base = control[0]
            out_pos = base + rank - count
            while union_bits != 0:
                low_bit = union_bits & -union_bits
                bit_idx = cute.arch.popc(low_bit - Int32(1))
                kv_idx = word_idx * 32 + bit_idx
                member = Int32(0)
                if (bits0 & low_bit) != 0:
                    member = member | Int32(1)
                if (bits1 & low_bit) != 0:
                    member = member | Int32(2)
                if kv_idx < self.seqlen_kv and out_pos < self.max_union:
                    union_idxs[pair_idx, out_pos] = kv_idx | (member << 29)
                out_pos = out_pos + 1
                union_bits = union_bits ^ low_bit
            cute.arch.sync_threads()
            if tidx == self.block_threads - 1:
                control[0] = base + rank
            cute.arch.sync_threads()

        if tidx == 0:
            union_length[pair_idx] = control[0]

    @cute.jit
    def __call__(
        self,
        topk_idxs: cute.Tensor,
        topk_length: Optional[cute.Tensor],
        union_idxs: cute.Tensor,
        union_length: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_pairs = cute.size(union_idxs.shape[0])
        self.kernel(topk_idxs, topk_length, union_idxs, union_length).launch(
            grid=(num_pairs, 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )
