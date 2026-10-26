# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic top-k locality bucketing shared by SM100 H64 training."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cudnn._cutlass_compat import SmemAllocator

H64_TOPK_BUCKETS = 8


class TopKBucketSm100:
    """Group a fixed-width top-k row into coarse key-address buckets."""

    THREADS = 256

    def __init__(self, topk: int, valid_buckets: int = H64_TOPK_BUCKETS):
        if topk <= 0 or topk % self.THREADS:
            raise ValueError("topk must be a positive multiple of 256")
        if valid_buckets <= 0 or valid_buckets & (valid_buckets - 1):
            raise ValueError("valid_buckets must be a positive power of two")
        self.topk = topk
        self.valid_buckets = valid_buckets
        self.invalid_bucket = valid_buckets
        self.tail_bucket = valid_buckets + 1
        self.buckets = valid_buckets + 2
        self.items_per_thread = topk // self.THREADS

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mOutputLength: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        max_seqlen_kv: Int32,
        bucket_shift: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mInput, mOutput, mOutputLength, mTopkLength, max_seqlen_kv, bucket_shift).launch(
            grid=(mInput.shape[0], 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mOutputLength: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        max_seqlen_kv: Int32,
        bucket_shift: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        smem = SmemAllocator()
        histogram = smem.allocate_tensor(Int32, cute.make_layout((self.buckets,)), byte_alignment=16)
        counters = smem.allocate_tensor(Int32, cute.make_layout((self.buckets,)), byte_alignment=16)
        warp_histogram = smem.allocate_tensor(Int32, cute.make_layout((8, self.buckets)), byte_alignment=16)

        for bucket in cutlass.range(tidx, self.buckets, self.THREADS, unroll=1):
            histogram[bucket] = Int32(0)
            counters[bucket] = Int32(0)
        cute.arch.sync_threads()

        indices = [Int32(0)] * self.items_per_thread
        buckets = [Int32(0)] * self.items_per_thread
        valid_length = mTopkLength[row] if cutlass.const_expr(mTopkLength is not None) else Int32(self.topk)
        for item in cutlass.range_constexpr(self.items_per_thread):
            pos = tidx + Int32(item * self.THREADS)
            index = mInput[row, pos]
            in_prefix = pos < valid_length
            valid = in_prefix and index >= Int32(0) and index < max_seqlen_kv
            bucket = index >> bucket_shift if valid else Int32(self.invalid_bucket)
            if not in_prefix:
                bucket = Int32(self.tail_bucket)
            elif bucket >= Int32(self.valid_buckets):
                bucket = Int32(self.valid_buckets - 1) if valid else Int32(self.invalid_bucket)
            indices[item] = index
            buckets[item] = bucket

            peers = cute.arch.match_sync(cutlass.Uint32(0xFFFFFFFF), bucket)
            leader = Int32(cute.arch.bfind(peers))
            if lane == leader:
                cute.arch.atomic_add(
                    (histogram.iterator + bucket).llvm_ptr,
                    Int32(cute.arch.popc(peers)),
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

        if tidx == Int32(0):
            offset = Int32(0)
            for bucket in cutlass.range_constexpr(self.valid_buckets):
                counters[bucket] = offset
                offset += histogram[bucket]
            mOutputLength[row] = offset
            counters[self.invalid_bucket] = offset
            offset += histogram[self.invalid_bucket]
            counters[self.tail_bucket] = offset
        cute.arch.sync_threads()

        for item in cutlass.range_constexpr(self.items_per_thread):
            bucket = buckets[item]
            for entry in cutlass.range(tidx, 8 * self.buckets, self.THREADS, unroll=1):
                warp_histogram[entry // self.buckets, entry % self.buckets] = Int32(0)
            cute.arch.sync_threads()

            peers = cute.arch.match_sync(cutlass.Uint32(0xFFFFFFFF), bucket)
            leader = Int32(cute.arch.bfind(peers))
            if lane == leader:
                warp_histogram[tidx // 32, bucket] = Int32(cute.arch.popc(peers))
            cute.arch.sync_threads()

            rank = Int32(cute.arch.popc(peers & cute.arch.lanemask_lt()))
            for warp in cutlass.range_constexpr(8):
                if Int32(warp) < tidx // 32:
                    rank += warp_histogram[warp, bucket]
            mOutput[row, counters[bucket] + rank] = indices[item]
            cute.arch.sync_threads()

            if tidx < Int32(self.buckets):
                phase_count = Int32(0)
                for warp in cutlass.range_constexpr(8):
                    phase_count += warp_histogram[warp, tidx]
                counters[tidx] += phase_count
            cute.arch.sync_threads()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
