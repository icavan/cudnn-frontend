# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.bindings.driver as cuda
import math
from typing import Tuple, Type, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass._mlir.dialects import arith, llvm, nvvm, vector


class FlashAttentionDSABackwardSm100:
    arch = 100

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
        lse_includes_sink: bool = False,
        num_dkv_shards: int | None = None,
        num_load_kv_warps: int = 16,
        pair_mask_encoded: bool = False,
    ):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.same_hdim_kv = head_dim == head_dim_v
        self.block_tile = block_tile
        self.max_topk = max_topk
        self.lse_includes_sink = lse_includes_sink
        self.pair_mask_encoded = pair_mask_encoded
        # Keep one FP32 accumulation buffer by default. Callers can still
        # request multiple shards to trade workspace for lower atomic
        # contention; all shards are reduced in FP32 before BF16 conversion.
        default_num_dkv_shards = 1
        self.num_dkv_shards = default_num_dkv_shards if num_dkv_shards is None else num_dkv_shards
        assert self.num_dkv_shards >= 1
        self.QK_mma_tiler = (block_tile, block_tile, head_dim)
        # head_dim_main: 128-aligned portion for the main 4 sub-tiles
        head_dim_main = (head_dim // 128) * 128
        self.head_dim_main = head_dim_main
        self.dOP_mma_tiler = (128, block_tile, block_tile)
        self.dOP_cta_tiler = (head_dim_v, block_tile, block_tile)
        self.dOV_mma_tiler = (block_tile, block_tile, head_dim_v)
        # V is the first head_dim_v columns of KV. Stack Q_main and dO along
        # M so the main contribution to S and dP share one M=128 UMMA.
        self.SdP_mma_tiler = (2 * block_tile, block_tile, head_dim_v)
        # The same physical Q_main/dO allocation is viewed transposed for dKV.
        # Its interleaved K=128 blocks are paired with the corresponding dS/P
        # blocks below, preserving dO^T@P followed by Q^T@dS accumulation.
        self.fused_dKV_mma_tiler = (128, block_tile, 2 * block_tile)
        self.fused_dKV_cta_tiler = (head_dim_v, block_tile, 2 * block_tile)
        self.KdS_mma_tiler = (128, block_tile, block_tile)
        self.KdS_cta_tiler = (head_dim_main, block_tile, block_tile)
        self.QdS_mma_tiler = (128, block_tile, block_tile)
        self.QdS_cta_tiler = (head_dim_main, block_tile, block_tile)
        self.cluster_shape_mn = (1, 1)

        if element_dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise ValueError(f"Unsupported element dtype: {element_dtype}")
        self.element_dtype = element_dtype
        # dKV accumulation stays FP32; element_dtype only controls element
        # and output storage.
        self.acc_dtype = Float32

        # =============== Sum OdO ================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 81 if max_topk == 2048 else 40 if max_topk == 1024 else 41
        self.sum_OdO_num_threads_d = 8 if max_topk == 2048 else 16
        self.sum_OdO_num_threads_q = self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        self.sum_OdO_elem_per_load = 8 if max_topk == 2048 else 4
        self.dSink_block_q = 256
        self.dSink_num_threads = 32

        # =============== Bwd ====================
        assert block_tile % (4 * num_load_kv_warps) == 0
        self.num_load_KV_warps = num_load_kv_warps
        self.num_compute_warps = 4
        self.num_reduce_warps = 8

        self.load_KV_warp_id = tuple(range(self.num_load_KV_warps))
        compute_warp_begin = self.num_load_KV_warps
        self.compute_warp_id = tuple(range(compute_warp_begin, compute_warp_begin + self.num_compute_warps))
        reduce_warp_begin = compute_warp_begin + self.num_compute_warps
        self.reduce_warp_id = tuple(range(reduce_warp_begin, reduce_warp_begin + self.num_reduce_warps))
        self.mma_warp_id = reduce_warp_begin + self.num_reduce_warps
        self.load_warp_id = self.mma_warp_id + 1
        self.empty_warp_id = self.load_warp_id + 1

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_load_KV_warps + self.num_compute_warps + self.num_reduce_warps + 4)

        # self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (self.num_compute_warps + self.num_reduce_warps + 1),
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.load_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.load_KV_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.num_load_KV_warps * self.threads_per_warp,
        )
        self.t2r_dKV01_done_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )
        self.t2r_dKV4_done_barrier = pipeline.NamedBarrier(
            barrier_id=8,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )
        # Order all final reduce-warp TMEM reads before compute warp 0 frees
        # the allocation. The reducers only arrive; compute warp 0 waits.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )
        # In the 576/512 specialization, dKV2/dKV3 alias the dKV0/dKV1 TMEM
        # columns. This barrier orders their T2R reads before the next tile
        # overwrites the aliased columns.
        self.t2r_dKV23_done_barrier = pipeline.NamedBarrier(
            barrier_id=10,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )

        self.tmem_S_offset = 0
        self.tmem_dP_offset = 0
        self.tmem_dKV0_offset = self.tmem_dP_offset + block_tile
        self.tmem_dKV1_offset = self.tmem_dKV0_offset + block_tile
        self.tmem_dKV2_offset = self.tmem_dKV0_offset
        self.tmem_dKV3_offset = self.tmem_dKV1_offset
        self.tmem_dQ0_offset = self.tmem_dKV3_offset + block_tile
        self.tmem_dQ1_offset = self.tmem_dQ0_offset + block_tile
        self.tmem_dQ2_offset = self.tmem_dQ1_offset + block_tile
        self.tmem_dQ3_offset = self.tmem_dQ2_offset + block_tile
        self.tmem_dQ4_offset = self.tmem_dQ3_offset + block_tile
        # The 64-wide tail reuses the S half of TMEM after S/dP have been
        # consumed for the current tile.  Keeping it separate from dKV0/dKV2
        # removes two tail-specific reuse barriers from the MMA critical path.
        self.tmem_dKV4_offset = self.tmem_S_offset

        self.dQ4_mma_tiler = (64, block_tile, block_tile)
        self.dKV4_mma_tiler = (64, block_tile, block_tile)

        # The asymmetric path needs more registers in the sparse gather for
        # long-lived top-k indices and copy addresses. Compute stays below its
        # 128-register cap, so this redistribution also lowers the CTA total.
        self.num_regs_load_KV = 40 if num_load_kv_warps <= 8 else 24
        self.num_regs_compute = 128
        self.num_regs_reduce = 96
        self.num_regs_mma = 40
        self.num_regs_empty = 40
        self.num_regs_load = 40

        self.buffer_align_bytes = 1024
        self.non_tma_align_bytes = 128

    def _setup_attributes(self):
        self.load_mma_QdO_stage = 1
        self.load_mma_K_stage = 1
        # self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_compute_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_reduce_dKV_stage = 2
        self.reduce_store_dKV_stage = 1
        self.compute_tmastore_dQ_stage = 1

    @staticmethod
    def _get_workspace_size_LSE_OdO(q: int, d: int, h: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        # q is total seqlen, b=1
        d = (d + 7) // 8 * 8  # round up to 8
        q = (q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # Avoid single workspace bytes exceeds 32bit range
        return (b, h, q, workspace_bytes)

    @staticmethod
    def _get_workspace_size_dKV(
        k: int,
        d: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
        num_shards: int = 1,
    ):
        d = (d + 7) // 8 * 8  # round up to 8
        k = (k + 7) // 8 * 8  # round up to 8
        # FP32 versions of dKV
        workspace_bytes = d * acc_dtype.width // 8
        return (b, num_shards, k, workspace_bytes)

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        total_seqlen_Q: Int32,
        total_seqlen_KV: Int32,
        acc_dtype: Type[cutlass.Numeric],
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        # problem_shape contains the max seqlen of Q and K
        max_Q, max_K, D, HB = (
            problem_shape[0],
            problem_shape[1],
            problem_shape[2],
            problem_shape[3],
        )
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])

        D = cute.round_up(D, 8)
        total_seqlen_Q = cute.round_up(total_seqlen_Q, 8)

        acc_bytes = acc_dtype.width // 8
        sum_OdO_bytes = cute.assume(H * total_seqlen_Q * acc_bytes, divby=acc_bytes * 64)

        sum_OdO_iter = workspace_LSE_OdO.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dKV_acc_iter = workspace_dKV.iterator

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dKV_acc_iter = cute.recast_ptr(dKV_acc_iter, dtype=self.acc_dtype)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout((H, (total_seqlen_Q, 1)), stride=(1, (cute.assume(H, divby=64), 0))),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout((H, (total_seqlen_Q, 1)), stride=(1, (cute.assume(H, divby=64), 0))),
        )
        dKV_acc = cute.make_tensor(
            dKV_acc_iter,
            cute.make_layout((D, total_seqlen_KV * self.num_dkv_shards, (1, 1)), stride=(1, D, (0, 0))),
        )

        return sum_OdO, scaled_lse, dKV_acc

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        block_q: int,
    ) -> Tuple[int, int, int]:
        grid = (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )
        return grid

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """
        Forward pass for DeepSeek Sparse Attention.
        """

        # [M, H, D] -> [H, D, (M, 1)]
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout((mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)), stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)))
        )

        # [N, D] -> [N, D, (1, 1)]
        mKV = cute.make_tensor(mKV.iterator, cute.make_layout((mKV.shape[0], mKV.shape[1], (1, 1)), stride=(mKV.stride[0], mKV.stride[1], (0, 0))))

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mOut = cute.make_tensor(
            mOut.iterator, cute.make_layout((mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)), stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)))
        )

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mdO = cute.make_tensor(
            mdO.iterator, cute.make_layout((mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)), stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)))
        )
        # [M, H, D] -> [D, H, (M, 1)]
        mdQ = cute.make_tensor(
            mdQ.iterator, cute.make_layout((mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)), stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)))
        )
        # [N, D] -> [D, N, (1, 1)]
        mdKV = cute.make_tensor(mdKV.iterator, cute.make_layout((mdKV.shape[1], mdKV.shape[0], (1, 1)), stride=(mdKV.stride[1], mdKV.stride[0], (0, 0))))

        # [M, H] -> [H, (M, 1)]
        mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout((mLSE.shape[1], (mLSE.shape[0], 1)), stride=(mLSE.stride[1], (mLSE.stride[0], 0))))

        # [H] -> [H, (1, 1)]
        mdSink = cute.make_tensor(mdSink.iterator, cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0))))
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)

        # [M, TopK] -> [TopK, (M, 1)]
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator, cute.make_layout((mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)), stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)))
        )
        # [M] -> [M, (1, 1)] when provided; None means non-compact (use full topk, -1 entries in topk_idxs)
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(mTopkLength.iterator, cute.make_layout((mTopkLength.shape[0], (1, 1)), stride=(mTopkLength.stride[0], (0, 0))))

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # S = Q @ KV
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cta_group, self.QK_mma_tiler[:2]
        )

        # dP = dO @ KV
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cta_group, self.dOV_mma_tiler[:2]
        )
        SdP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cta_group, self.SdP_mma_tiler[:2]
        )

        # dKV = dO^T @ P
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.dOP_mma_tiler[:2]
        )
        # dKV = Q^T @ dS
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.QdS_mma_tiler[:2]
        )
        # dQ = KV @ dS^T
        KdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cta_group, self.KdS_mma_tiler[:2]
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            # dKV4: Q^T[512:575] @ dS -> (64, 64) output
            dKV4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.dKV4_mma_tiler[:2]
            )
            # dQ4: K[512:575] @ dS^T -> (64, 64) output
            dQ4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cta_group, self.dQ4_mma_tiler[:2]
            )
        else:
            dKV4_tiled_mma = None
            dQ4_tiled_mma = None

        self.cluster_layout_vmnk = cute.make_layout(((1), (1, 1, 1)), stride=((0), (0, 0, 0)))

        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_QdO_stage)
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_K_stage)
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_QdO_stage)
        SdP_smem_layout_staged = sm100_utils.make_smem_layout_a(SdP_tiled_mma, self.SdP_mma_tiler, self.element_dtype, self.load_mma_QdO_stage)
        SdP_V_smem_layout_staged = sm100_utils.make_smem_layout_b(SdP_tiled_mma, self.SdP_mma_tiler, self.element_dtype, self.load_mma_K_stage)
        fused_QdO_T_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOP_tiled_mma,
            self.fused_dKV_cta_tiler,
            self.element_dtype,
            self.load_mma_QdO_stage,
        )
        fused_PdS_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOP_tiled_mma,
            self.fused_dKV_mma_tiler,
            self.element_dtype,
            self.compute_mma_P_stage,
        )
        Q_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QK_tiled_mma,
            (self.block_tile, self.block_tile, self.block_tile),
            self.element_dtype,
            self.load_mma_QdO_stage,
        )
        fused_group_stride = 2 * self.block_tile * self.block_tile
        # Physically interleave Q and dO in 16-row groups.  An M=128 UMMA
        # writes contiguous TMEM lanes, while an M=64 consumer addresses four
        # 16-lane stripes.  This layout makes those two representations agree:
        # Q[0:16], dO[0:16], Q[16:32], dO[16:32], ... .
        Q_fused_smem_layout_staged = cute.make_composed_layout(
            Q_smem_layout_staged.inner,
            0,
            cute.make_layout(
                (((16, 4), 16), 1, (4, self.head_dim_main // 64), self.load_mma_QdO_stage),
                stride=(((64, 2048), 1), 0, (16, fused_group_stride), 0),
            ),
        )
        dO_fused_smem_layout_staged = cute.make_composed_layout(
            dO_smem_layout_staged.inner,
            0,
            cute.make_layout(
                (((16, 4), 16), 1, (4, self.head_dim_v // 64), self.load_mma_QdO_stage),
                stride=(((64, 2048), 1), 0, (16, fused_group_stride), 0),
            ),
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            V_smem_layout_staged = sm100_utils.make_smem_layout_b(dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_K_stage)
        else:
            V_smem_layout_staged = K_smem_layout_staged

        dOT_smem_layout_staged = sm100_utils.make_smem_layout_a(dOP_tiled_mma, self.dOP_cta_tiler, self.element_dtype, self.load_mma_QdO_stage)
        P_smem_layout_staged = sm100_utils.make_smem_layout_b(dOP_tiled_mma, self.dOP_mma_tiler, self.element_dtype, self.compute_mma_P_stage)
        P_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.COL_MAJOR, self.QK_mma_tiler[:2], self.compute_mma_P_stage
        )
        K_smem_layout_staged_2 = sm100_utils.make_smem_layout_a(KdS_tiled_mma, self.KdS_cta_tiler, self.element_dtype, self.load_mma_K_stage)
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sK with 64-wide blocks, giving head_dim/64 sub-tiles
            K_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dQ4_tiled_mma, (self.head_dim, self.block_tile, self.block_tile), self.element_dtype, self.load_mma_K_stage
            )
        else:
            K_tail_smem_layout_staged = None
        dST_smem_layout_staged = sm100_utils.make_smem_layout_b(KdS_tiled_mma, self.KdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage)
        QT_smem_layout_staged = sm100_utils.make_smem_layout_a(QdS_tiled_mma, self.QdS_cta_tiler, self.element_dtype, self.load_mma_QdO_stage)
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sQ with 64-wide blocks
            QT_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dKV4_tiled_mma, (self.head_dim, self.block_tile, self.block_tile), self.element_dtype, self.load_mma_QdO_stage
            )
        else:
            QT_tail_smem_layout_staged = None
        dS_smem_layout_staged = sm100_utils.make_smem_layout_b(QdS_tiled_mma, self.QdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage)
        dS_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.COL_MAJOR, self.dOV_mma_tiler[:2], self.compute_mma_dS_stage
        )

        dQ_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.from_tensor(mdQ), (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]), self.mma_compute_dQ_stage
        )

        dKV_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype, utils.LayoutEnum.from_tensor(mdKV), (self.dOP_mma_tiler[0], self.dOP_mma_tiler[1] // 2), self.mma_reduce_dKV_stage
        )

        LSE_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_sum_OdO_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(Q_fused_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            Q_smem_layout,
            (self.block_tile, self.block_tile, self.head_dim_main),
            QK_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            Q_tail_smem_layout = cute.select(
                Q_tail_smem_layout_staged,
                mode=[0, 1, 2],
            )
            tma_atom_Q_tail, tma_tensor_Q_tail = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                Q_tail_smem_layout,
                (self.block_tile, self.block_tile, self.block_tile),
                QK_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_atom_Q_tail = None
            tma_tensor_Q_tail = None

        dO_smem_layout = cute.select(dO_fused_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mdO, dO_smem_layout, self.dOV_mma_tiler, dOV_tiled_mma, self.cluster_layout_vmnk.shape
        )

        dQ_smem_layout = cute.select(dQ_smem_layout_staged, mode=[0, 1])
        tma_atom_dQ, tma_tensor_dQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            dQ_smem_layout,
            (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            dQ4_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.element_dtype, utils.LayoutEnum.from_tensor(mdQ), (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]), self.mma_compute_dQ_stage
            )
            dQ4_smem_layout = cute.select(dQ4_smem_layout_staged, mode=[0, 1])
            tma_atom_dQ_64, tma_tensor_dQ_64 = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mdQ,
                dQ4_smem_layout,
                (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]),
            )
        else:
            dQ4_smem_layout_staged = None
            tma_atom_dQ_64 = None
            tma_tensor_dQ_64 = None

        element_bytes = self.element_dtype.width // 8
        self.tma_copy_Q_bytes = self.block_tile * self.head_dim * element_bytes
        self.tma_copy_dO_bytes = self.block_tile * self.head_dim_v * element_bytes
        self.tma_copy_QdO_bytes = self.tma_copy_Q_bytes + self.tma_copy_dO_bytes

        _max_smem_bytes = 227 * 1024

        @cute.struct
        class SharedStorage:
            load_mma_QdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QdO_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_compute_LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            mma_compute_dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dQ_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_reduce_dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_reduce_dKV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            sdO: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(dO_smem_layout_staged)], self.buffer_align_bytes]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(K_smem_layout_staged)], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(P_smem_layout_staged)], self.non_tma_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(dS_smem_layout_staged)], self.non_tma_align_bytes]
            sLSE: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)], self.non_tma_align_bytes]
            sSum_OdO: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)], self.non_tma_align_bytes]
            sTopkMask: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.block_tile],
                self.non_tma_align_bytes,
            ]

        assert (
            SharedStorage.size_in_bytes() <= _max_smem_bytes
        ), f"SharedStorage ({SharedStorage.size_in_bytes()} bytes) exceeds {_max_smem_bytes} bytes (227KB)"
        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, mdKV_acc = self.get_workspace_tensor(
            problem_shape,
            workspace_LSE_OdO,
            workspace_dKV,
            mQ.shape[2][0],
            mKV.shape[0],
            self.acc_dtype,
        )
        # ============ Sum OdO ============
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)
        self.sum_OdO(
            mOut,
            mdO,
            sum_OdO,
            mLSE,
            mAttnSink,
            scaled_LSE,
            sum_OdO_scale,
            LSE_scale,
            problem_shape,
        ).launch(
            grid=sum_OdO_grid,
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        num_head_blocks = cute.ceil_div(problem_shape[3][0], self.block_tile)
        bwd_grid = (problem_shape[0], num_head_blocks, problem_shape[3][1])
        self.bwd(
            problem_shape,
            QK_tiled_mma,
            dOV_tiled_mma,
            SdP_tiled_mma,
            dOP_tiled_mma,
            QdS_tiled_mma,
            KdS_tiled_mma,
            dKV4_tiled_mma,
            dQ4_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_Q_tail,
            tma_tensor_Q_tail,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ,
            tma_tensor_dQ,
            tma_atom_dQ_64,
            tma_tensor_dQ_64,
            mQ,
            mdO,
            mKV,
            mdQ,
            mdKV_acc,
            mdSink,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            scaled_LSE,
            sum_OdO,
            softmax_scale,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            dO_smem_layout_staged,
            V_smem_layout_staged,
            SdP_smem_layout_staged,
            SdP_V_smem_layout_staged,
            fused_QdO_T_smem_layout_staged,
            fused_PdS_smem_layout_staged,
            Q_tail_smem_layout_staged,
            Q_fused_smem_layout_staged,
            dO_fused_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            P_smem_layout_store_staged,
            K_smem_layout_staged_2,
            K_tail_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            QT_tail_smem_layout_staged,
            dS_smem_layout_staged,
            dS_smem_layout_store_staged,
            dKV_smem_layout_staged,
            dQ_smem_layout_staged,
            dQ4_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.block_seq = 4 if self.max_topk == 2048 else 32
        self.num_threads_D_convert = 32
        self.num_threads_seq = 4 if self.max_topk == 2048 else self.block_seq
        self.convert_elem_per_load = 4

        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
        convert_grid = [
            convert_grid_x,
            1,
            1,
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]
        self.convert(
            mdKV_acc,
            mdKV,
            mKV.shape[0],
        ).launch(
            grid=convert_grid,
            block=convert_block,
            stream=stream,
        )

        dSink_grid = (
            cute.ceil_div(problem_shape[0], self.dSink_block_q),
            problem_shape[3][0],
            problem_shape[3][1],
        )
        self.sum_dSink(
            sum_OdO,
            scaled_LSE,
            mAttnSink,
            mdSink,
            problem_shape,
        ).launch(
            grid=dSink_grid,
            block=[self.dSink_num_threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def convert(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,  # (D, N, (1, B))
        seqlen: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        (
            seq_block_idx,
            _,
            batch_idx,
        ) = cute.arch.block_idx()

        seq_id = self.block_seq * seq_block_idx + tidy

        if seq_id < seqlen:
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            # Tiles from 128-wide store_dKV: layout = groups of 4 per lane
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    cur_tile_mdKV_acc = Float32(0.0)
                    for shard_idx in cutlass.range_constexpr(self.num_dkv_shards):
                        cur_mdKV_acc_row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, batch_idx)]
                        tile_mdKV_acc_row = cute.flat_divide(cur_mdKV_acc_row, (64,))  # (64, D/64)
                        tile_mdKV_acc_row = cute.flat_divide(tile_mdKV_acc_row, (32,))  # (32, 2, D/64)
                        cur_tile_mdKV_acc += tile_mdKV_acc_row[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dim_idx] = self.element_dtype(cur_tile_mdKV_acc)
            # Last tile from 64-wide store_dKV_64: layout = groups of 2 per lane
            # Layout F (M=64, 16dp): dp_idx//4=k → warp=k//8, lane=k%8
            # pos 2k holds M=warp*16+lane, pos 2k+1 holds M=warp*16+lane+8
            # Unscramble: p=tidx+j*32, k=p//2 → dim = base + (k//8)*16 + k%8 + (p%2)*8
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    cur_tile_mdKV_acc = Float32(0.0)
                    for shard_idx in cutlass.range_constexpr(self.num_dkv_shards):
                        cur_mdKV_acc_row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, batch_idx)]
                        tile_mdKV_acc_row = cute.flat_divide(cur_mdKV_acc_row, (64,))  # (64, D/64)
                        tile_mdKV_acc_row = cute.flat_divide(tile_mdKV_acc_row, (32,))  # (32, 2, D/64)
                        cur_tile_mdKV_acc += tile_mdKV_acc_row[tidx, j, num_128_tiles]
                    k = tidx // 2 + j * 16
                    dim_idx = self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    cur_mdKV_row[dim_idx] = self.element_dtype(cur_tile_mdKV_acc)

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        attn_sink: cute.Tensor,
        scaled_lse: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, tidz = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        offset = 0

        for idx_q_t in cutlass.range(tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[bidy, None, (idx_q + offset, bidz)]
                O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
                dO_bhq = dO[bidy, None, (idx_q + offset, bidz)]
                dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step):
                    O_frag = O_bhq[None, idx_d].load()
                    dO_frag = dO_bhq[None, idx_d].load()
                    prod_frag = O_frag * dO_frag
                    prod_frag = prod_frag.to(self.acc_dtype)
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.sum_OdO_num_threads_d)

                if tidx == 0:
                    lse_bhq = lse[bidy, (idx_q + offset, bidz)]
                    sum_OdO_bhq = sum_OdO_scale * acc

                    log2_e = -lse_scale
                    if cutlass.const_expr(self.lse_includes_sink):
                        scaled_lse_bhq = lse_scale * lse_bhq
                    else:
                        # The compatibility path consumes FlashMLA's original
                        # KV-only LSE and folds in the sink here.
                        attn_sink_bh = attn_sink[bidy, (0, bidz)]
                        lse_log2 = lse_bhq * log2_e
                        sink_log2 = attn_sink_bh * log2_e
                        lse_max_log2 = cute.arch.fmax(lse_log2, sink_log2)
                        sum_exp2 = Float32(cute.math.exp2(lse_log2 - lse_max_log2) + cute.math.exp2(sink_log2 - lse_max_log2))
                        lse_with_sink_log2 = lse_max_log2 + cute.math.log2(sum_exp2)
                        scaled_lse_bhq = -lse_with_sink_log2

                    if lse_bhq == Float32(float("inf")):
                        scaled_lse_bhq = Float32(float("-inf"))

                    sum_OdO[bidy, (idx_q + offset, bidz)] = sum_OdO_bhq
                    scaled_lse[bidy, (idx_q + offset, bidz)] = scaled_lse_bhq

    @cute.kernel
    def sum_dSink(
        self,
        sum_OdO: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        dSink: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        q_block_idx, head_idx, batch_idx = cute.arch.block_idx()
        tidx, _, batch_idx = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        q_end = min(seqlen_q, (q_block_idx + 1) * self.dSink_block_q)
        q_idx = q_block_idx * self.dSink_block_q + tidx

        log2_e = Float32(math.log2(math.e))
        sink_log2 = attn_sink[head_idx, (0, batch_idx)] * log2_e
        acc = Float32(0.0)

        while q_idx < q_end:
            p_sink = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
            acc += p_sink * sum_OdO[head_idx, (q_idx, batch_idx)]
            q_idx += self.dSink_num_threads

        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.dSink_num_threads)

        if tidx == 0:
            dSink_ptr = dSink.iterator + cute.crd2idx((head_idx, (0, batch_idx)), dSink.layout)
            cute.arch.atomic_add(dSink_ptr.llvm_ptr, acc)

    @cute.kernel
    def bwd(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        SdP_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_Q_tail: Optional[cute.CopyAtom],
        tma_tensor_Q_tail: Optional[cute.Tensor],
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mKV: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mdSink: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        scale_softmax: Float32 | float,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        SdP_smem_layout_staged: cute.ComposedLayout,
        SdP_V_smem_layout_staged: cute.ComposedLayout,
        fused_QdO_T_smem_layout_staged: cute.ComposedLayout,
        fused_PdS_smem_layout_staged: cute.ComposedLayout,
        Q_tail_smem_layout_staged: cute.ComposedLayout,
        Q_fused_smem_layout_staged: cute.ComposedLayout,
        dO_fused_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_store_staged: cute.ComposedLayout,
        K_smem_layout_staged_2: cute.ComposedLayout,
        K_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        QT_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dS_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_store_staged: cute.ComposedLayout,
        dKV_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        dQ4_smem_layout_staged: Optional[cute.ComposedLayout],
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        tidx, _, batch_idx = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        max_seqlen_q, max_seqlen_kv, head_dim, (num_heads, batch_size) = problem_shape

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]

        # A zero-tile warp-specialized pipeline cannot make progress. Handle
        # empty (and defensively, malformed negative) rows before initializing
        # barriers or allocating TMEM. Such a row contributes zero dQ/dKV.
        if topk <= 0:
            for linear_idx in cutlass.range(tidx, self.head_dim * self.block_tile, self.threads_per_cta):
                head_offset = linear_idx // self.head_dim
                dim_idx = linear_idx % self.head_dim
                head_idx = head_block_idx * self.block_tile + head_offset
                if head_idx < num_heads:
                    mdQ[dim_idx, head_idx, (token_idx, batch_idx)] = mdQ.element_type(0.0)
            cute.arch.nvvm.exit()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dQ)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_QdO_pipeline = self.make_and_init_load_mma_QdO_pipeline(
            storage.load_mma_QdO_mbar_ptr.data_ptr(),
        )
        load_mma_K_pipeline = self.make_and_init_load_mma_K_pipeline(
            storage.load_mma_K_mbar_ptr.data_ptr(),
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_LSE_mbar_ptr.data_ptr(),
        )
        load_compute_sum_OdO_pipeline = self.make_and_init_load_compute_sum_OdO_pipeline(
            storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr(),
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr(),
        )
        mma_compute_dQ_pipeline = self.make_and_init_mma_compute_dQ_pipeline(
            storage.mma_compute_dQ_mbar_ptr.data_ptr(),
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr(),
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr(),
        )
        mma_reduce_dKV_pipeline = self.make_and_init_mma_reduce_dKV_pipeline(
            storage.mma_reduce_dKV_mbar_ptr.data_ptr(),
        )
        compute_tmastore_dQ_pipeline = self.make_and_init_compute_tmastore_dQ_pipeline()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sK.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
        sP_store = storage.sP.get_tensor(P_smem_layout_store_staged.outer, swizzle=P_smem_layout_store_staged.inner)
        sQdO_base = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sQ_load_ptr = cute.recast_ptr(sQdO_base.iterator, Q_fused_smem_layout_staged.inner)
        sQ_load = cute.make_tensor(sQ_load_ptr, Q_fused_smem_layout_staged.outer)
        sdO_load_ptr = cute.recast_ptr(
            sQdO_base.iterator + 16 * self.block_tile,
            dO_fused_smem_layout_staged.inner,
        )
        sdO_load = cute.make_tensor(sdO_load_ptr, dO_fused_smem_layout_staged.outer)
        sSdP_ptr = cute.recast_ptr(sQdO_base.iterator, SdP_smem_layout_staged.inner)
        sSdP = cute.make_tensor(sSdP_ptr, SdP_smem_layout_staged.outer)
        sFusedQdOT_ptr = cute.recast_ptr(sQdO_base.iterator, fused_QdO_T_smem_layout_staged.inner)
        sFusedQdOT = cute.make_tensor(sFusedQdOT_ptr, fused_QdO_T_smem_layout_staged.outer)
        # The fused main allocation consumes all of sdO plus the first 64x512
        # elements of the adjacent sQ field. Address the dense tail from the
        # owning sQ field so pointer arithmetic stays within that allocation.
        sQ_storage = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer,
            swizzle=Q_smem_layout_staged.inner,
        )
        sQ_tail_ptr = cute.recast_ptr(
            sQ_storage.iterator + self.block_tile * self.head_dim_v,
            Q_tail_smem_layout_staged.inner,
        )
        sQ_tail = cute.make_tensor(sQ_tail_ptr, Q_tail_smem_layout_staged.outer)
        sSdP_V_ptr = cute.recast_ptr(sK.iterator, SdP_V_smem_layout_staged.inner)
        sSdP_V = cute.make_tensor(sSdP_V_ptr, SdP_V_smem_layout_staged.outer)
        sdS = storage.sdS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
        sdS_store = storage.sdS.get_tensor(dS_smem_layout_store_staged.outer, swizzle=dS_smem_layout_store_staged.inner)
        # reuse sK
        sdQ_ptr = cute.recast_ptr(sK.iterator, dQ_smem_layout_staged.inner)
        sdQ = cute.make_tensor(sdQ_ptr, dQ_smem_layout_staged.outer)

        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)
        sTopkMask = storage.sTopkMask.get_tensor(cute.make_layout((self.block_tile,)))

        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)

        sFusedPdS_ptr = cute.recast_ptr(sP.iterator, fused_PdS_smem_layout_staged.inner)
        sFusedPdS = cute.make_tensor(sFusedPdS_ptr, fused_PdS_smem_layout_staged.outer)

        sK_2_ptr = cute.recast_ptr(sK.iterator, K_smem_layout_staged_2.inner)
        sK_2 = cute.make_tensor(sK_2_ptr, K_smem_layout_staged_2.outer)

        if cutlass.const_expr(not self.same_hdim_kv):
            # sK_tail: view sK storage with 64-wide partitioning, access block 8 (cols 512:575)
            # K_tail_smem_layout_staged partitions head_dim=576 into 64-wide blocks → 9 blocks
            sK_tail_ptr = cute.recast_ptr(sK.iterator, K_tail_smem_layout_staged.inner)
            sK_tail_full = cute.make_tensor(sK_tail_ptr, K_tail_smem_layout_staged.outer)
            sK_tail = sK_tail_full[None, 8, None, None]  # block 8 = cols 512:575

            sQT_tail_ptr = cute.recast_ptr(sQ_tail.iterator, QT_tail_smem_layout_staged.inner)
            sQT_tail_full = cute.make_tensor(sQT_tail_ptr, QT_tail_smem_layout_staged.outer)
            sQT_tail = sQT_tail_full[None, 0, None, None]

            # sdQ4: reuse sK for the 64×64 dQ4 epilogue
            sdQ4_ptr = cute.recast_ptr(sK.iterator, dQ4_smem_layout_staged.inner)
            sdQ4 = cute.make_tensor(sdQ4_ptr, dQ4_smem_layout_staged.outer)

        pipeline.pipeline_init_wait()

        tile_count = cute.ceil_div(topk, self.block_tile)

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                QK_tiled_mma,
                dOV_tiled_mma,
                tma_atom_Q,
                tma_tensor_Q,
                tma_atom_Q_tail,
                tma_tensor_Q_tail,
                tma_atom_dO,
                tma_tensor_dO,
                mLSE,
                mSum_OdO,
                sQ_load,
                sdO_load,
                sQ_tail,
                sLSE,
                sSum_OdO,
                (load_mma_QdO_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline),
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )
            tSdP_shape = SdP_tiled_mma.partition_shape_C(cute.select(self.SdP_mma_tiler, mode=[0, 1]))
            tSdP_base = SdP_tiled_mma.make_fragment_C(tSdP_shape)
            tSdP = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tSdP_base.layout)

            # (MMA, MMA_M, MMA_K, STAGE)
            tSrQ = QK_tiled_mma.make_fragment_A(sQ_tail)
            # (MMA, MMA_N, MMA_K, STAGE)
            tSrK = QK_tiled_mma.make_fragment_B(sK)
            tSdPrA = SdP_tiled_mma.make_fragment_A(sSdP)
            tSdPrV = SdP_tiled_mma.make_fragment_B(sSdP_V)

            tdKVrQdOT = dOP_tiled_mma.make_fragment_A(sFusedQdOT)
            tdKVrPdS = dOP_tiled_mma.make_fragment_B(sFusedPdS)
            # The compact fused view is contiguous only for the tuned
            # single-stage P/dS pipeline. Keep independent views as a
            # correctness fallback for staged-store configurations, whose
            # physical allocation is P[all stages] followed by dS[all stages].
            tdKVrP = dOP_tiled_mma.make_fragment_B(sP)
            tdKVrdS = QdS_tiled_mma.make_fragment_B(sdS)
            tdKVrQdOT_shape = (
                tdKVrQdOT.shape[0],
                1,
                tdKVrQdOT.shape[1],
                tdKVrQdOT.shape[2],
                tdKVrQdOT.shape[3],
            )
            tdKVrQdOT_stride = (
                tdKVrQdOT.stride[0],
                0,
                tdKVrQdOT.stride[1],
                tdKVrQdOT.stride[2],
                tdKVrQdOT.stride[3],
            )
            tdKVrQdOT = cute.make_tensor(
                tdKVrQdOT.iterator,
                cute.make_layout(tdKVrQdOT_shape, stride=tdKVrQdOT_stride),
            )

            tdQrK = KdS_tiled_mma.make_fragment_A(sK_2)
            tdQrdST = KdS_tiled_mma.make_fragment_B(sdST)

            tdQrK_shape = (tdQrK.shape[0], 1, tdQrK.shape[1], tdQrK.shape[2], tdQrK.shape[3])
            tdQrK_stride = (tdQrK.stride[0], 0, tdQrK.stride[1], tdQrK.stride[2], tdQrK.stride[3])
            tdQrK = cute.make_tensor(tdQrK.iterator, cute.make_layout(tdQrK_shape, stride=tdQrK_stride))

            if cutlass.const_expr(not self.same_hdim_kv):
                # dQ4 fragment: sK_tail (64-wide, single M-block) @ dS^T
                # sK_tail has 3 modes after slicing: (tile, K_blocks, stage)
                # make_fragment_A returns 3 modes: (MMA, MMA_K, STAGE)
                # Reshape to 5 modes: (MMA, 1_dummy, 1_M_block, MMA_K, STAGE)
                tdQrK_tail = dQ4_tiled_mma.make_fragment_A(sK_tail)
                tdQrK_tail_shape = (tdQrK_tail.shape[0], 1, 1, tdQrK_tail.shape[1], tdQrK_tail.shape[2])
                tdQrK_tail_stride = (tdQrK_tail.stride[0], 0, 0, tdQrK_tail.stride[1], tdQrK_tail.stride[2])
                tdQrK_tail = cute.make_tensor(tdQrK_tail.iterator, cute.make_layout(tdQrK_tail_shape, stride=tdQrK_tail_stride))

                # dKV4 fragment: sQT_tail (64-wide, single M-block) @ dS
                tdKVrQT_tail = dKV4_tiled_mma.make_fragment_A(sQT_tail)
                tdKVrQT_tail_shape = (tdKVrQT_tail.shape[0], 1, 1, tdKVrQT_tail.shape[1], tdKVrQT_tail.shape[2])
                tdKVrQT_tail_stride = (tdKVrQT_tail.stride[0], 0, 0, tdKVrQT_tail.stride[1], tdKVrQT_tail.stride[2])
                tdKVrQT_tail = cute.make_tensor(tdKVrQT_tail.iterator, cute.make_layout(tdKVrQT_tail_shape, stride=tdKVrQT_tail_stride))

                tdKVrdS_4 = dKV4_tiled_mma.make_fragment_B(sdS)
            else:
                tdQrK_tail = None
                tdKVrQT_tail = None
                tdKVrdS_4 = None

            self.mma(
                QK_tiled_mma,
                dOV_tiled_mma,
                SdP_tiled_mma,
                dOP_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tSrQ,
                tSrK,
                tSdPrA,
                tSdPrV,
                tdKVrQdOT,
                tdKVrPdS,
                tdKVrP,
                tdKVrdS,
                tdQrK,
                tdQrdST,
                tdQrK_tail,
                tdKVrQT_tail,
                tdKVrdS_4,
                tStS,
                tdPtdP,
                tSdP,
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                tile_count,
                sdS,
                (
                    load_mma_QdO_pipeline,
                    load_mma_K_pipeline,
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    mma_compute_dQ_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_reduce_dKV_pipeline,
                ),
            )

        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )

            self.compute(
                tma_atom_dQ,
                tma_tensor_dQ,
                tma_atom_dQ_64,
                tma_tensor_dQ_64,
                dQ4_tiled_mma,
                tStS,
                tdPtdP,
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                sLSE,
                sSum_OdO,
                sP_store,
                sdS,
                sdS_store,
                sdQ,
                sdQ4 if not self.same_hdim_kv else None,
                sTopkMask,
                scale_softmax,
                tile_count,
                (
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    load_compute_LSE_pipeline,
                    load_compute_sum_OdO_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_compute_dQ_pipeline,
                    compute_tmastore_dQ_pipeline,
                ),
            )

            if warp_idx == self.compute_warp_id[0]:
                self.tmem_dealloc_barrier.arrive_and_wait()
                cute.arch.dealloc_tmem(tmem_ptr_base, self.num_tmem_alloc_cols)

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )

            self.reduce_dKV(
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                mdKV_acc,
                mTopkIdxs,
                max_seqlen_kv,
                tile_count,
                topk,
                mma_reduce_dKV_pipeline,
            )
            # All T2R operations issued by this reducer are fenced before the
            # function returns. Signal compute warp 0 that deallocation is safe.
            self.tmem_dealloc_barrier.arrive()

        elif warp_idx in self.load_KV_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load_KV)
            self.load_KV(
                mKV,
                mTopkIdxs,
                sK,
                tile_count,
                topk,
                load_mma_K_pipeline,
                mTopkLength,
                sTopkMask,
            )

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.jit
    def load(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_Q_tail: Optional[cute.CopyAtom],
        tma_tensor_Q_tail: Optional[cute.Tensor],
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sQ_tail: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        pipelines,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp

        load_mma_QdO_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline = pipelines

        load_mma_QdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_QdO_stage)
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage)

        gQ = cute.local_tile(
            tma_tensor_Q,
            (self.block_tile, self.head_dim_main),
            (None, None, (token_idx, batch_idx)),
        )
        gdO = cute.local_tile(tma_tensor_dO, cute.select(self.dOV_mma_tiler, mode=[0, 2]), (None, None, (token_idx, batch_idx)))

        QK_thr_mma = QK_tiled_mma.get_slice(0)
        tSgQ = QK_thr_mma.partition_A(gQ)
        tQsQ, tQgQ_mkl = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            gQ_tail = cute.local_tile(
                tma_tensor_Q_tail,
                (self.block_tile, self.block_tile),
                (None, None, (token_idx, batch_idx)),
            )
            tSgQ_tail = QK_thr_mma.partition_A(gQ_tail)
            tQsQ_tail, tQgQ_tail_mkl = cpasync.tma_partition(
                tma_atom_Q_tail,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_tail, 0, 3),
                cute.group_modes(tSgQ_tail, 0, 3),
            )
        dOV_thr_mma = dOV_tiled_mma.get_slice(0)
        tdPgdO = dOV_thr_mma.partition_A(gdO)
        tdPsdO, tdPgdO_mkl = cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPgdO, 0, 3),
        )

        load_mma_QdO_pipeline.producer_acquire(load_mma_QdO_producer_state)

        tma_barrier = load_mma_QdO_pipeline.producer_get_barrier(load_mma_QdO_producer_state)
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[None, head_block_idx, 0],
            tQsQ[None, load_mma_QdO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            cute.copy(
                tma_atom_Q_tail,
                tQgQ_tail_mkl[
                    None,
                    head_block_idx,
                    self.head_dim_main // self.block_tile,
                ],
                tQsQ_tail[None, load_mma_QdO_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )
        cute.copy(
            tma_atom_dO,
            tdPgdO_mkl[None, head_block_idx, 0],
            tdPsdO[None, load_mma_QdO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )
        load_mma_QdO_producer_state.advance()

        async_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS), self.acc_dtype, num_bits_per_copy=64)
        thr_layout = cute.make_layout((32), stride=(1))
        val_layout = cute.make_layout((2), stride=(1))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        thr_async_copy = async_tiled_copy.get_slice(local_tidx)

        # (64, 1, M, B)
        gLSE = cute.flat_divide(mLSE, (self.block_tile,))
        gSum_OdO = cute.flat_divide(mSum_OdO, (self.block_tile,))

        # Load LSE
        load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

        gLSE_for_copy = thr_async_copy.partition_S(gLSE[None, head_block_idx, (token_idx, batch_idx)])
        sLSE_for_copy = thr_async_copy.partition_D(sLSE)

        cute.copy(
            async_copy_atom,
            gLSE_for_copy[None, 0],
            sLSE_for_copy[None, 0, load_compute_LSE_producer_state.index],
        )
        load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
        load_compute_LSE_producer_state.advance()

        # Load Sum_OdO
        load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)
        gSum_OdO_for_copy = thr_async_copy.partition_S(gSum_OdO[None, head_block_idx, (token_idx, batch_idx)])
        sSum_OdO_for_copy = thr_async_copy.partition_D(sSum_OdO)

        cute.copy(
            async_copy_atom,
            gSum_OdO_for_copy[None, 0],
            sSum_OdO_for_copy[None, 0, load_compute_sum_OdO_producer_state.index],
        )

        load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
        load_compute_sum_OdO_producer_state.advance()

    @cute.jit
    def _copy_kv_row(
        self,
        mKV: cute.Tensor,
        topk_idx: Int32,
        batch_idx: Int32,
        tile_sK: cute.Tensor,
        lane_in_subwarp: Int32,
        async_copy_atom: cute.CopyAtom,
        async_thr_copy: cute.TiledCopy,
    ):
        """Copy one KV row with an eight-lane subgroup.

        Four independent rows are issued by each loader warp.  Each subgroup
        walks the 64-element column groups of its row, increasing sparse-load
        memory-level parallelism without changing the shared-memory tile.
        """
        gK_row = mKV[topk_idx, None, (0, batch_idx)]
        tile_gK = cute.composition(gK_row, cute.make_layout(tile_sK.shape))
        for group_idx in cutlass.range_constexpr(self.head_dim // 64):
            cur_gK = tile_gK[None, group_idx]
            cur_sK = tile_sK[None, group_idx]
            tSgK = async_thr_copy.partition_S(cur_gK)
            tSsK = async_thr_copy.partition_D(cur_sK)
            cute.copy(async_copy_atom, tSgK, tSsK)

    @cute.jit
    def _zero_kv_row(
        self,
        tile_sK: cute.Tensor,
        lane_in_subwarp: Int32,
    ):
        for group_idx in cutlass.range_constexpr(self.head_dim // 64):
            cur_sK = tile_sK[None, group_idx]
            cur_sK = cute.flat_divide(cur_sK, (8,))
            cur_sK = cur_sK[None, lane_in_subwarp]
            cur_sK.fill(0.0)

    @cute.jit
    def _load_kv_rows(
        self,
        mKV: cute.Tensor,
        sK_slice: cute.Tensor,
        topk_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        mTopkLength: Optional[cute.Tensor],
        is_first: bool,
        local_tidx: Int32,
        row: Int32,
        async_copy_atom: cute.CopyAtom,
        async_thr_copy: cute.TiledCopy,
    ):
        """Load one row per eight-lane subgroup."""
        _, _, batch_idx = cute.arch.block_idx()
        lane_in_subwarp = local_tidx % 8
        idx = tile_index * self.block_tile + row
        tile_sK = sK_slice[row, (None, None)]

        if cutlass.const_expr(mTopkLength is not None):
            if cutlass.const_expr(is_first):
                if idx < topk:
                    self._copy_kv_row(
                        mKV,
                        topk_idx,
                        batch_idx,
                        tile_sK,
                        lane_in_subwarp,
                        async_copy_atom,
                        async_thr_copy,
                    )
                else:
                    self._zero_kv_row(tile_sK, lane_in_subwarp)
            else:
                self._copy_kv_row(
                    mKV,
                    topk_idx,
                    batch_idx,
                    tile_sK,
                    lane_in_subwarp,
                    async_copy_atom,
                    async_thr_copy,
                )
        else:
            if idx < topk:
                if topk_idx >= 0:
                    self._copy_kv_row(
                        mKV,
                        topk_idx,
                        batch_idx,
                        tile_sK,
                        lane_in_subwarp,
                        async_copy_atom,
                        async_thr_copy,
                    )
                else:
                    self._zero_kv_row(tile_sK, lane_in_subwarp)
            else:
                self._zero_kv_row(tile_sK, lane_in_subwarp)

    @cute.jit
    def load_KV(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        sK: cute.Tensor,
        tile_count: Int32,
        topk: Int32,
        load_mma_K_pipeline,
        mTopkLength: Optional[cute.Tensor],
        sTopkMask: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp
        local_warp_idx = tidx // self.threads_per_warp
        subgroup = local_tidx // 8
        lane_in_subwarp = local_tidx % 8

        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_layout((8,))
        val_layout = cute.make_layout((8,))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        async_thr_copy = async_tiled_copy.get_slice(lane_in_subwarp)

        load_mma_K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_K_stage)

        tile_index = tile_count - 1
        full_tiles = (topk % self.block_tile) == 0
        rows_per_warp = self.block_tile // self.num_load_KV_warps
        row_batches = rows_per_warp // 4
        rTopkIdx = cute.make_rmem_tensor((row_batches,), cutlass.Int32)

        while tile_index >= 0:
            for row_batch in cutlass.range_constexpr(row_batches):
                row = local_warp_idx * rows_per_warp + row_batch * 4 + subgroup
                idx = tile_index * self.block_tile + row
                topk_idx = Int32(-1)
                if lane_in_subwarp == 0:
                    if idx < self.max_topk:
                        topk_idx = mTopkIdxs[idx, (token_idx, batch_idx)]
                rTopkIdx[row_batch] = cute.arch.shuffle_sync(topk_idx, subgroup * 8)

            load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
            sK_slice = sK[(None, None), 0, (None, None), load_mma_K_producer_state.index]
            sK_slice = cute.composition(sK_slice, cute.make_layout((self.block_tile, self.head_dim)))

            for row_batch in cutlass.range_constexpr(row_batches):
                row = local_warp_idx * rows_per_warp + row_batch * 4 + subgroup
                idx = tile_index * self.block_tile + row
                topk_idx = rTopkIdx[row_batch]
                if cutlass.const_expr(self.pair_mask_encoded):
                    member = Int32(0)
                    if idx < topk and topk_idx >= 0:
                        member = (topk_idx >> 29) & Int32(3)
                        topk_idx = topk_idx & Int32(0x1FFFFFFF)
                    if lane_in_subwarp == 0:
                        sTopkMask[row] = member
                if full_tiles:
                    self._load_kv_rows(
                        mKV,
                        sK_slice,
                        topk_idx,
                        tile_index,
                        topk,
                        mTopkLength,
                        is_first=False,
                        local_tidx=local_tidx,
                        row=row,
                        async_copy_atom=async_copy_atom,
                        async_thr_copy=async_thr_copy,
                    )
                else:
                    self._load_kv_rows(
                        mKV,
                        sK_slice,
                        topk_idx,
                        tile_index,
                        topk,
                        mTopkLength,
                        is_first=True,
                        local_tidx=local_tidx,
                        row=row,
                        async_copy_atom=async_copy_atom,
                        async_thr_copy=async_thr_copy,
                    )

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            self.load_KV_sync_barrier.arrive_and_wait()
            load_mma_K_pipeline.producer_commit(load_mma_K_producer_state)
            load_mma_K_producer_state.advance()
            tile_index -= 1

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        SdP_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tSdPrA: cute.Tensor,
        tSdPrV: cute.Tensor,
        tdKVrQdOT: cute.Tensor,
        tdKVrPdS: cute.Tensor,
        tdKVrP: cute.Tensor,
        tdKVrdS: cute.Tensor,
        tdQrK: cute.Tensor,
        tdQrdST: cute.Tensor,
        tdQrK_tail: Optional[cute.Tensor],
        tdKVrQT_tail: Optional[cute.Tensor],
        tdKVrdS_4: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tSdP: cute.Tensor,
        tdKVtdKV: Tuple,
        tdQtdQ: Tuple,
        tile_count: Int32,
        sdS: cute.Tensor,
        pipelines,
    ):
        (
            load_mma_QdO_pipeline,
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            mma_compute_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_reduce_dKV_pipeline,
        ) = pipelines
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % self.threads_per_warp
        token_idx, _, batch_idx = cute.arch.block_idx()

        load_mma_QdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_QdO_stage)
        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_compute_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        mma_reduce_dKV_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dKV_stage)
        use_fused_pds_view = self.compute_mma_P_stage == 1 and self.compute_mma_dS_stage == 1
        if cutlass.const_expr(use_fused_pds_view):
            dkv_k_blocks = cute.size(tdKVrPdS, mode=[2]) // 2
        else:
            dkv_k_blocks = cute.size(tdKVrP, mode=[2])

        load_mma_QdO_pipeline.consumer_wait(load_mma_QdO_consumer_state)
        mma_compute_dQ_pipeline.producer_acquire(mma_compute_dQ_producer_state)

        tile_index = tile_count - 1
        is_first_mma = True
        while tile_index >= 0:
            # dKV4 from the previous tile lives in the S TMEM region.  By the
            # time the next tile reaches here, dKV2/dKV3 MMA work has provided
            # enough overlap for the reducer to finish its TMEM readback.
            if cutlass.const_expr(not self.same_hdim_kv):
                if not is_first_mma:
                    self.t2r_dKV4_done_barrier.arrive_and_wait()

            load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

            # [S_main; dP] = [Q_main; dO] @ V^T.  The physical 16-row
            # interleave places the two M=64 outputs in complementary TMEM
            # lane stripes within the same columns.
            SdP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSdPrA, mode=[2]), unroll=4):
                cute.gemm(
                    SdP_tiled_mma,
                    tSdP,
                    tSdPrA[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tSdPrV[None, None, k_block, load_mma_K_consumer_state.index],
                    tSdP,
                )
                SdP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # S has one additional 64-wide Q/K tail when Dqk=576.
            if cutlass.const_expr(not self.same_hdim_kv):
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                for tail_k_block in cutlass.range(
                    0,
                    cute.size(tSrQ, mode=[2]),
                    unroll_full=True,
                ):
                    cute.gemm(
                        QK_tiled_mma,
                        tStS,
                        tSrQ[None, None, tail_k_block, load_mma_QdO_consumer_state.index],
                        tSrK[
                            None,
                            None,
                            tail_k_block + self.head_dim_main // 16,
                            load_mma_K_consumer_state.index,
                        ],
                        tStS,
                    )

            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()
            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            # Gemm dKV = dO @ P part1
            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)

            # Start the P-dependent half before dS is ready.  dKV0/dKV1 alias
            # the previous tile's dKV2/dKV3 columns, so retain the existing
            # T2R ordering before issuing into those accumulators.
            if not is_first_mma:
                if cutlass.const_expr(self.same_hdim_kv):
                    self.t2r_dKV4_done_barrier.arrive_and_wait()
                else:
                    self.t2r_dKV23_done_barrier.arrive_and_wait()

            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    p_fragment = tdKVrPdS[None, None, k_block, compute_mma_P_consumer_state.index]
                else:
                    p_fragment = tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV0,
                    tdKVrQdOT[None, None, 0, 2 * k_block + 1, load_mma_QdO_consumer_state.index],
                    p_fragment,
                    tdKVtdKV0,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    p_fragment = tdKVrPdS[None, None, k_block, compute_mma_P_consumer_state.index]
                else:
                    p_fragment = tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV1,
                    tdKVrQdOT[None, None, 1, 2 * k_block + 1, load_mma_QdO_consumer_state.index],
                    p_fragment,
                    tdKVtdKV1,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            # Consume K as soon as dS is ready. Releasing the single-stage K
            # buffer here lets the sparse gather for the next tile overlap
            # the remaining dKV work, which no longer depends on K.
            # dQ0
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ0,
                    tdQrK[None, None, 0, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ0,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ1
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ1,
                    tdQrK[None, None, 1, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ1,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ2
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ2,
                    tdQrK[None, None, 2, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ2,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ3
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ3,
                    tdQrK[None, None, 3, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ3,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ4 (tail 64 cols: K[512:575] @ dS^T)
            if cutlass.const_expr(not self.same_hdim_kv):
                dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
                for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                    cute.gemm(
                        dQ4_tiled_mma,
                        tdQtdQ4,
                        tdQrK_tail[None, None, 0, k_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ4,
                    )
                    dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
            load_mma_K_consumer_state.advance()

            # Complete dKV0/dKV1 with Q^T @ dS after K has been released.
            # dKV0
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    ds_fragment = tdKVrPdS[
                        None,
                        None,
                        k_block + dkv_k_blocks,
                        compute_mma_dS_consumer_state.index,
                    ]
                else:
                    ds_fragment = tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV0,
                    tdKVrQdOT[None, None, 0, 2 * k_block, load_mma_QdO_consumer_state.index],
                    ds_fragment,
                    tdKVtdKV0,
                )
            # dKV1
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    ds_fragment = tdKVrPdS[
                        None,
                        None,
                        k_block + dkv_k_blocks,
                        compute_mma_dS_consumer_state.index,
                    ]
                else:
                    ds_fragment = tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV1,
                    tdKVrQdOT[None, None, 1, 2 * k_block, load_mma_QdO_consumer_state.index],
                    ds_fragment,
                    tdKVtdKV1,
                )

            # Notify to reduce the first part of dKV (dKV0, dKV1)
            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # Gemm dKV4 = Q^T[512:575] @ dS (round 1.5, only GEMM5, no GEMM3).
            # Its TMEM region held S/dP earlier in this tile and is free after
            # compute_mma_dS_pipeline.consumer_wait above.
            if cutlass.const_expr(not self.same_hdim_kv):
                dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrdS_4, mode=[2]), unroll=2):
                    cute.gemm(
                        dKV4_tiled_mma,
                        tdKVtdKV4,
                        tdKVrQT_tail[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS_4[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV4,
                    )
                    dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Commit dKV4 on pipeline (no acquire needed) to notify consumer
                mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                mma_reduce_dKV_producer_state.advance()

            # Gemm dKV = dO @ P part2.  dKV4 is in the disjoint S TMEM region,
            # so dKV2 can start as soon as its pipeline stage is available.
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
            # In the 512/512 specialization dKV2/dKV3 alias dKV0/dKV1, while
            # their notifications occupy different pipeline generations.
            if cutlass.const_expr(self.same_hdim_kv):
                self.t2r_dKV01_done_barrier.arrive_and_wait()
            # dKV2
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    p_fragment = tdKVrPdS[None, None, k_block, compute_mma_P_consumer_state.index]
                else:
                    p_fragment = tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV2,
                    tdKVrQdOT[None, None, 2, 2 * k_block + 1, load_mma_QdO_consumer_state.index],
                    p_fragment,
                    tdKVtdKV2,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dKV3
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    p_fragment = tdKVrPdS[None, None, k_block, compute_mma_P_consumer_state.index]
                else:
                    p_fragment = tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV3,
                    tdKVrQdOT[None, None, 3, 2 * k_block + 1, load_mma_QdO_consumer_state.index],
                    p_fragment,
                    tdKVtdKV3,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # P is used
            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            # Gemm dKV = Q @ dS
            # dKV2
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    ds_fragment = tdKVrPdS[
                        None,
                        None,
                        k_block + dkv_k_blocks,
                        compute_mma_dS_consumer_state.index,
                    ]
                else:
                    ds_fragment = tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV2,
                    tdKVrQdOT[None, None, 2, 2 * k_block, load_mma_QdO_consumer_state.index],
                    ds_fragment,
                    tdKVtdKV2,
                )
            # dKV3
            for k_block in cutlass.range(0, dkv_k_blocks, unroll=2):
                if cutlass.const_expr(use_fused_pds_view):
                    ds_fragment = tdKVrPdS[
                        None,
                        None,
                        k_block + dkv_k_blocks,
                        compute_mma_dS_consumer_state.index,
                    ]
                else:
                    ds_fragment = tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index]
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV3,
                    tdKVrQdOT[None, None, 3, 2 * k_block, load_mma_QdO_consumer_state.index],
                    ds_fragment,
                    tdKVtdKV3,
                )

            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # dS is used
            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            is_first_mma = False
            tile_index -= 1

        # Balance the final reducer arrivals that are otherwise consumed at
        # the beginning of the next tile.
        self.t2r_dKV4_done_barrier.arrive_and_wait()
        if cutlass.const_expr(not self.same_hdim_kv):
            self.t2r_dKV23_done_barrier.arrive_and_wait()

        mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)
        mma_compute_dQ_producer_state.advance()

        load_mma_QdO_pipeline.consumer_release(load_mma_QdO_consumer_state)
        load_mma_QdO_consumer_state.advance()

    @cute.jit
    def compute(
        self,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: Tuple,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        sP_store: cute.Tensor,
        sdS: cute.Tensor,
        sdS_store: cute.Tensor,
        sdQ: cute.Tensor,
        sdQ4: Optional[cute.Tensor],
        sTopkMask: cute.Tensor,
        scale_softmax: Float32,
        tile_count: Int32,
        pipelines,
    ):
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dQ_pipeline,
            compute_tmastore_dQ_pipeline,
        ) = pipelines

        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.compute_warp_id[0] * self.threads_per_warp
        tidx_in_warp = tidx % self.threads_per_warp

        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        mma_compute_dQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage)
        compute_tmastore_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_tmastore_dQ_stage)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        # (((16,4),64), 1, 1):(((65536,2097152),1),0,0)
        tStS = tStS[(None, None), 0, 0]
        tdPtdP = tdPtdP[(None, None), 0, 0]

        dp_idx = tidx_in_wg % 128
        cS = cute.make_identity_tensor(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        cS = cute.composition(cS, sP_store[None, None, compute_mma_P_producer_state.index].layout)
        cdP = cute.make_identity_tensor(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
        cdP = cute.composition(cdP, sdS_store[None, None, compute_mma_dS_producer_state.index].layout)

        tiled_t2r_S = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        tiled_t2r_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPtdP)
        thr_t2r_S = tiled_t2r_S.get_slice(tidx % 128)
        thr_t2r_dP = tiled_t2r_dP.get_slice(tidx % 128)

        tTR_cS = thr_t2r_S.partition_D(cS)
        tTR_sS = thr_t2r_S.partition_D(sP_store[None, None, 0])
        tTR_rS = cute.make_rmem_tensor(tTR_sS.shape, self.acc_dtype)

        tTR_tS = thr_t2r_S.partition_S(tStS)

        tTR_cdP = thr_t2r_dP.partition_D(cdP)
        tTR_sdP = thr_t2r_dP.partition_D(sdS_store[None, None, 0])
        tTR_rdP = cute.make_rmem_tensor(tTR_sdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r_dP.partition_S(tdPtdP)

        load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)

        log2_e = Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e

        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.element_dtype,
        )
        smem_store_p = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_S)
        thr_smem_store_p = smem_store_p.get_slice(tidx % 128)
        tRS_sP = thr_smem_store_p.partition_D(sP_store)
        tRS_rP = cute.make_rmem_tensor(tRS_sP[None, None, None, 0].shape, self.element_dtype)

        smem_store_ds = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_dP)
        thr_smem_store_ds = smem_store_ds.get_slice(tidx % 128)
        tRS_sdS = thr_smem_store_ds.partition_D(sdS_store)
        tRS_rdS = cute.make_rmem_tensor(tRS_sdS[None, None, None, 0].shape, self.element_dtype)

        tile_index = tile_count - 1
        while tile_index >= 0:
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)

            cute.copy(tiled_t2r_S, tTR_tS, tTR_rS)

            for i in cutlass.range(0, cute.size(tTR_rS), 2, unroll_full=True):

                lse = (
                    sLSE[cute.get(tTR_cS[i], mode=[0]), load_compute_LSE_consumer_state.index],
                    sLSE[cute.get(tTR_cS[i + 1], mode=[0]), load_compute_LSE_consumer_state.index],
                )

                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)

                if cutlass.const_expr(self.pair_mask_encoded):
                    head0 = cute.get(tTR_cS[i], mode=[0])
                    row0 = cute.get(tTR_cS[i], mode=[1])
                    required0 = Int32(1) if head0 < self.block_tile // 2 else Int32(2)
                    if (sTopkMask[row0] & required0) == 0:
                        tTR_rS[i] = Float32(0.0)

                    head1 = cute.get(tTR_cS[i + 1], mode=[0])
                    row1 = cute.get(tTR_cS[i + 1], mode=[1])
                    required1 = Int32(1) if head1 < self.block_tile // 2 else Int32(2)
                    if (sTopkMask[row1] & required1) == 0:
                        tTR_rS[i + 1] = Float32(0.0)

            tTR_rS_f16 = self.quantize(tTR_rS, 4)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            # ======= stsm ============
            tRS_rP.store(smem_store_p.retile(tTR_rS_f16).load())
            if cutlass.const_expr(self.compute_mma_P_stage == 1):
                cute.copy(smem_store_p, tRS_rP, tRS_sP[None, None, None, 0])
            else:
                cute.copy(smem_store_p, tRS_rP, tRS_sP[None, None, None, compute_mma_P_producer_state.index])

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            # Publish P before constructing dS.  This lets the MMA warp issue
            # dO^T @ P while the compute warps evaluate P * (dP - D).
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            cute.copy(tiled_t2r_dP, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (tTR_rS[i], tTR_rS[i + 1]),
                )

            tTR_rdP_f16 = self.quantize(tTR_rdP, 4, scale_softmax)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            tRS_rdS.store(smem_store_ds.retile(tTR_rdP_f16).load())
            if cutlass.const_expr(self.compute_mma_dS_stage == 1):
                cute.copy(smem_store_ds, tRS_rdS, tRS_sdS[None, None, None, 0])
            else:
                cute.copy(smem_store_ds, tRS_rdS, tRS_sdS[None, None, None, compute_mma_dS_producer_state.index])

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            tile_index -= 1

        load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)

        # Store dQ
        tdQtdQ0 = tdQtdQ0[(None, None), 0, 0]
        tdQtdQ1 = tdQtdQ1[(None, None), 0, 0]
        tdQtdQ2 = tdQtdQ2[(None, None), 0, 0]
        tdQtdQ3 = tdQtdQ3[(None, None), 0, 0]

        # (512, 64)
        gdQ = cute.local_tile(tma_tensor_dQ, cute.select(self.KdS_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
        # (128, 64)
        gdQ0 = gdQ[None, None, 0, head_block_idx]
        gdQ1 = gdQ[None, None, 1, head_block_idx]
        gdQ2 = gdQ[None, None, 2, head_block_idx]
        gdQ3 = gdQ[None, None, 3, head_block_idx]

        # sdQ: ((64,2),(8,8),(1,1))
        sdQ_slice = sdQ[None, None, mma_compute_dQ_consumer_state.index]

        # ((64,2),(8,8),(1,1))
        tdQsdQ0, tdQgdQ0_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ0, 0, 2),
        )
        tdQsdQ1, tdQgdQ1_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ1, 0, 2),
        )
        tdQsdQ2, tdQgdQ2_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ2, 0, 2),
        )
        tdQsdQ3, tdQgdQ3_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ3, 0, 2),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            tdQtdQ4 = tdQtdQ4[(None, None), 0, 0]
            gdQ4 = cute.local_tile(tma_tensor_dQ_64, cute.select(self.dQ4_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
            gdQ4 = gdQ4[None, None, 8, head_block_idx]

            sdQ4_slice = sdQ4[None, None, mma_compute_dQ_consumer_state.index]

            tdQsdQ4, tdQgdQ4_mkl = cpasync.tma_partition(
                tma_atom_dQ_64,
                0,
                cute.make_layout(1),
                cute.group_modes(sdQ4_slice, 0, 2),
                cute.group_modes(gdQ4, 0, 2),
            )

        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        mma_compute_dQ_pipeline.consumer_wait(mma_compute_dQ_consumer_state)

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        # Wait in all threads for the acquire to complete
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ0,
            tdQgdQ0_mkl,
            tdQtdQ0,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ1,
            tdQgdQ1_mkl,
            tdQtdQ1,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ2,
            tdQgdQ2_mkl,
            tdQtdQ2,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ3,
            tdQgdQ3_mkl,
            tdQtdQ3,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        # Store dQ4 (tail 64 cols)
        if cutlass.const_expr(not self.same_hdim_kv):
            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ_64(
                tma_atom_dQ_64,
                sdQ4_slice,
                tdQsdQ4,
                tdQgdQ4_mkl,
                tdQtdQ4,
                dp_idx,
                wg_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()
            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

        mma_compute_dQ_pipeline.consumer_release(mma_compute_dQ_consumer_state)
        mma_compute_dQ_consumer_state.advance()

        compute_tmastore_dQ_pipeline.producer_tail()

    @cute.jit
    def reduce_dKV(
        self,
        tdKVtdKV: Tuple,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_count: Int32,
        topk: Int32,
        mma_reduce_dKV_pipeline,
    ):
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV

        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)

        num_warp_groups = self.num_reduce_warps // 4

        tdKVtdKV0 = tdKVtdKV0[(None, None), 0, 0]
        tdKVtdKV1 = tdKVtdKV1[(None, None), 0, 0]
        tdKVtdKV2 = tdKVtdKV2[(None, None), 0, 0]
        tdKVtdKV3 = tdKVtdKV3[(None, None), 0, 0]
        if cutlass.const_expr(not self.same_hdim_kv):
            tdKVtdKV4 = tdKVtdKV4[(None, None), 0, 0]

        # Set up identity tensor partition once (all dKV sub-tiles share the same layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV0)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)

        mma_reduce_dKV_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dKV_stage)

        tile_index = tile_count - 1
        rTopkIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        full_tiles = (topk % self.block_tile) == 0
        shard_row_offset = (token_idx % self.num_dkv_shards) * max_seqlen_kv
        while tile_index >= 0:
            # Preload topk indices into rmem (shared across all 4 store_dKV calls)
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                local_row_idx = cute.get(tTR_cdKV[coord_base], mode=[1])
                global_row_idx = tile_index * self.block_tile + local_row_idx
                if full_tiles:
                    topk_idx = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                    if cutlass.const_expr(self.pair_mask_encoded):
                        topk_idx = topk_idx & Int32(0x1FFFFFFF)
                    rTopkIdx[i] = topk_idx + shard_row_offset if topk_idx >= 0 else Int32(-1)
                else:
                    if global_row_idx < topk:
                        topk_idx = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                        if cutlass.const_expr(self.pair_mask_encoded):
                            topk_idx = topk_idx & Int32(0x1FFFFFFF)
                        rTopkIdx[i] = topk_idx + shard_row_offset if topk_idx >= 0 else Int32(-1)
                    else:
                        rTopkIdx[i] = Int32(-1)

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            if cutlass.const_expr(not self.same_hdim_kv):
                # Split T2R and atomic_add so the producer can overlap the
                # following dKV4 MMA with these global reductions.
                rdKV0 = self.t2r_dKV(tdKVtdKV0)
                rdKV1 = self.t2r_dKV(tdKVtdKV1)
                cute.arch.fence_view_async_tmem_load()
                # The TMEM stage is no longer needed once both fragments are
                # resident in registers.  Release it before the slow global
                # reductions so MMA can fill the next stage concurrently.
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                self.reduce_dKV_from_reg(mdKV_acc, rdKV0, rTopkIdx, 0)
                self.reduce_dKV_from_reg(mdKV_acc, rdKV1, rTopkIdx, 1)
            else:
                rdKV0 = self.t2r_dKV(tdKVtdKV0)
                rdKV1 = self.t2r_dKV(tdKVtdKV1)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV01_done_barrier.arrive_and_wait()
                self.reduce_dKV_from_reg(mdKV_acc, rdKV0, rTopkIdx, 0)
                self.reduce_dKV_from_reg(mdKV_acc, rdKV1, rTopkIdx, 1)
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()

            # dKV4 reduce (round 1.5): cols 512:575
            # dKV4 reduce: use pipeline for notification, barrier for T2R safety
            if cutlass.const_expr(not self.same_hdim_kv):
                mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

                # T2R dKV4, then signal MMA that TMEM is free for dKV2/dKV3
                rdKV4 = self.t2r_dKV_64(tdKVtdKV4)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV4_done_barrier.arrive_and_wait()
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                # The 64-wide and 128-wide T2R layouts partition the 64 key
                # rows identically, so the main fragment's top-k registers are
                # also the exact row mapping for dKV4.
                self.reduce_dKV_64_from_reg(mdKV_acc, rdKV4, rTopkIdx)

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            if cutlass.const_expr(self.same_hdim_kv):
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV4_done_barrier.arrive_and_wait()
                self.reduce_dKV_from_reg(mdKV_acc, rdKV2, rTopkIdx, 2)
                self.reduce_dKV_from_reg(mdKV_acc, rdKV3, rTopkIdx, 3)
            else:
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV23_done_barrier.arrive_and_wait()
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                self.reduce_dKV_from_reg(mdKV_acc, rdKV2, rTopkIdx, 2)
                self.reduce_dKV_from_reg(mdKV_acc, rdKV3, rTopkIdx, 3)

            if cutlass.const_expr(self.same_hdim_kv):
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()

            tile_index -= 1

    @cute.jit
    def t2r_dKV(self, tdKVtdKV: cute.Tensor):
        """T2R: load dKV from TMEM to registers. Caller must fence after all T2R calls."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)
        return tTR_rdKV

    @cute.jit
    def reduce_dKV_from_reg(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
    ):
        """Reduce dKV from registers to global memory via atomic_add."""
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))  # (128, 4)
                tile_dKV_row = tile_dKV_row[None, sub_tile_idx]
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (4,))  # (4, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load(), sem="relaxed", scope="gpu")

    @cute.jit
    def t2r_dKV_64(self, tdKVtdKV: cute.Tensor):
        """T2R: load 64-wide dKV from TMEM to registers. Caller must fence."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)
        return tTR_rdKV

    @cute.jit
    def reduce_dKV_64_from_reg(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
    ):
        """Reduce 64-wide dKV from registers to global memory via atomic_add."""
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((2,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (64,))  # (64, D/64)
                tile_dKV_row = tile_dKV_row[None, self.head_dim_main // 64]  # last 64-elem tile
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (2,))  # (2, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load(), sem="relaxed", scope="gpu")

    @cute.jit
    def store_dKV(
        self,
        dKV_acc: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
    ):
        tidx, _, batch_idx = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)

        cute.arch.fence_view_async_tmem_load()

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))  # (128, 4)
                tile_dKV_row = tile_dKV_row[None, sub_tile_idx]
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (4,))  # (4, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load(), sem="relaxed", scope="gpu")

    @cute.jit
    def store_dKV_64(
        self,
        dKV_acc: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
    ):
        tidx, _, batch_idx = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)

        cute.arch.fence_view_async_tmem_load()

        # Same compact store as store_dKV: dp_idx//4 indexes into 2-element
        # groups. convert() handles the different unscramble for this 64-tile.
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((2,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (64,))  # (64, D/64)
                tile_dKV_row = tile_dKV_row[None, self.head_dim_main // 64]  # last 64-elem tile
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (2,))  # (2, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load(), sem="relaxed", scope="gpu")

    @cute.jit
    def store_dQ(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        warp_idx: Int32,
    ):
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        cdQ = cute.make_identity_tensor(cute.select(self.KdS_mma_tiler, mode=[0, 1]))
        num_warp_groups = self.num_compute_warps // 4

        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)

        tTR_cdQ = thr_t2r_dQ.partition_D(cdQ)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)

        tRS_rdQ = self.quantize(tTR_rdQ, 4)

        cute.arch.fence_view_async_tmem_load()

        # ((64,2),(8,8),(1,1))
        thread_layout = cute.make_ordered_layout((128, 64), (0, 1))
        sdQ_slice_tmp = cute.composition(sdQ, thread_layout)
        sdQ_slice = cute.composition(sdQ_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdQ.shape))
        cute.autovec_copy(tRS_rdQ, sdQ_slice)

        self.compute_sync_barrier.arrive_and_wait()

        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )

        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def store_dQ_64(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        wg_idx: Int32,
        warp_idx: Int32,
    ):
        # Use same Ld16x256bOp as store_dKV for the 64-wide TMEM tile
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(2)),
            self.acc_dtype,
        )
        num_warp_groups = self.num_compute_warps // 4

        cdQ = cute.make_identity_tensor(cute.select(self.dQ4_mma_tiler, mode=[0, 1]))

        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)

        tTR_cdQ_p = thr_t2r_dQ.partition_D(cdQ)
        tTR_cdQ = self.split_wg(tTR_cdQ_p, num_warp_groups, wg_idx)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)
        tTR_tdQ = self.split_wg(tTR_tdQ, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)

        cute.arch.fence_view_async_tmem_load()

        # Write dQ4 to smem element-by-element using coord tensor
        for i in cutlass.range_constexpr(cute.size(tTR_rdQ)):
            row = cute.get(tTR_cdQ[i], mode=[0])
            col = cute.get(tTR_cdQ[i], mode=[1])
            sdQ[row, col] = self.element_dtype(tTR_rdQ[i])

        self.compute_sync_barrier.arrive_and_wait()

        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )

        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
        softmax_scale: Optional[Float32] = None,
    ):
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            if cutlass.const_expr(softmax_scale is not None):
                frg_vec = frg_vec * softmax_scale
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    def split_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 4):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], t.shape[2], (cute.size(t, mode=[3]) // num_warp_groups, num_warp_groups))))
            ret = p[None, None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], (cute.size(t, mode=[2]) // num_warp_groups, num_warp_groups))))
            ret = p[None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(t, cute.make_layout((t.shape[0], (cute.size(t, mode=[1]) // num_warp_groups, num_warp_groups))))
            ret = p[None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 1):
            p = cute.composition(t, cute.make_layout((t.shape[0] // num_warp_groups, num_warp_groups)))
            ret = p[None, wg_idx]
        return ret

    def interleave_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
        """Interleave split on last mode across warp groups.
        For shape (16, 4) with 2 warp groups: last mode (4) → (2, 2),
        wg0 gets tiles 0,2 and wg1 gets tiles 1,3 → each wg gets (16, 2).
        """
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(t, cute.make_layout((t.shape[0], (num_warp_groups, cute.size(t, mode=[1]) // num_warp_groups))))
            ret = p[None, (wg_idx, None)]
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups))))
            ret = p[None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def get_tmem_tensor(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tmem_ptr_base: cute.Pointer,
    ):
        tStS_shape = QK_tiled_mma.partition_shape_C(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
        # The interleaved SMEM input places Q/S in the standard even 16-lane
        # stripes used by an M=64 fragment.
        tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

        tdPtdP_shape = dOV_tiled_mma.partition_shape_C(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
        tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
        # dO/dP occupy the complementary odd 16-lane stripes.
        tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset + (16 << 16), tdPtdP.layout)

        tdKVtdKV_shape = QdS_tiled_mma.partition_shape_C(cute.select(self.QdS_mma_tiler, mode=[0, 1]))
        tdKVtdKV_base = QdS_tiled_mma.make_fragment_C(tdKVtdKV_shape)
        tdKVtdKV0 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV0_offset, tdKVtdKV_base.layout)
        tdKVtdKV1 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV1_offset, tdKVtdKV_base.layout)
        if cutlass.const_expr(self.same_hdim_kv):
            tdKVtdKV2 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ4_offset, tdKVtdKV_base.layout)
            tdKVtdKV3 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV1_offset, tdKVtdKV_base.layout)
        else:
            tdKVtdKV2 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV2_offset, tdKVtdKV_base.layout)
            tdKVtdKV3 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV3_offset, tdKVtdKV_base.layout)

        tdQtdQ_shape = KdS_tiled_mma.partition_shape_C(cute.select(self.KdS_mma_tiler, mode=[0, 1]))
        tdQtdQ_base = KdS_tiled_mma.make_fragment_C(tdQtdQ_shape)
        tdQtdQ0 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ0_offset, tdQtdQ_base.layout)
        tdQtdQ1 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ1_offset, tdQtdQ_base.layout)
        tdQtdQ2 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ2_offset, tdQtdQ_base.layout)
        tdQtdQ3 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ3_offset, tdQtdQ_base.layout)

        if cutlass.const_expr(not self.same_hdim_kv):
            tdKVtdKV4_shape = dKV4_tiled_mma.partition_shape_C(cute.select(self.dKV4_mma_tiler, mode=[0, 1]))
            tdKVtdKV4_base = dKV4_tiled_mma.make_fragment_C(tdKVtdKV4_shape)
            tdKVtdKV4 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV4_offset, tdKVtdKV4_base.layout)

            tdQtdQ4_shape = dQ4_tiled_mma.partition_shape_C(cute.select(self.dQ4_mma_tiler, mode=[0, 1]))
            tdQtdQ4_base = dQ4_tiled_mma.make_fragment_C(tdQtdQ4_shape)
            tdQtdQ4 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ4_offset, tdQtdQ4_base.layout)
        else:
            tdKVtdKV4 = None
            tdQtdQ4 = None

        return tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4

    def make_and_init_load_mma_QdO_pipeline(self, load_mma_QdO_mbar_ptr):
        load_mma_QdO_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_QdO_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_QdO_mbar_ptr,
            num_stages=self.load_mma_QdO_stage,
            producer_group=load_mma_QdO_producer_group,
            consumer_group=load_mma_QdO_consumer_group,
            tx_count=self.tma_copy_QdO_bytes,
            defer_sync=True,
        )

    def make_and_init_load_mma_K_pipeline(self, load_mma_K_mbar_ptr):
        load_mma_K_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * self.num_load_KV_warps)
        load_mma_K_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=load_mma_K_mbar_ptr,
            num_stages=self.load_mma_K_stage,
            producer_group=load_mma_K_producer_group,
            consumer_group=load_mma_K_consumer_group,
            defer_sync=True,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
            defer_sync=True,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(self, load_compute_sum_OdO_mbar_ptr):
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dQ_pipeline(self, mma_compute_dQ_mbar_ptr):
        mma_compute_dQ_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        mma_compute_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dQ_mbar_ptr,
            num_stages=self.mma_compute_dQ_stage,
            producer_group=mma_compute_dQ_producer_group,
            consumer_group=mma_compute_dQ_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_reduce_dKV_pipeline(self, mma_reduce_dKV_mbar_ptr):
        mma_reduce_dKV_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        mma_reduce_dKV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_reduce_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dKV_mbar_ptr,
            num_stages=self.mma_reduce_dKV_stage,
            producer_group=mma_reduce_dKV_producer_group,
            consumer_group=mma_reduce_dKV_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_tmastore_dQ_pipeline(self):
        compute_tmastore_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.compute_tmastore_dQ_stage,
            producer_group=compute_tmastore_dQ_producer_group,
        )
