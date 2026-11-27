# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic expert-grouped NVFP4 weight gradients over a ragged token K."""

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from cudnn.gemm.cutedsl.grouped.wgrad._blockscaled_api import GroupedGemmWgradBlockScaledAPI


class GroupedOperand:
    def __init__(self, dim, rows, experts, device):
        self.dim = dim
        self.padded = (dim + 127) // 128 * 128
        self.experts = experts
        self.data = torch.empty((dim, rows // 2), dtype=torch.uint8, device=device)
        self.sf = torch.empty((self.padded, rows // 16), dtype=torch.float8_e4m3fn, device=device)

        def flat(dtype, size):
            return make_fake_compact_tensor(dtype, (size,), stride_order=(0,), assumed_align=4)

        self.launch = cute.compile(
            self.reorder,
            flat(cutlass.Int64, dim * rows // 16),
            flat(cutlass.Uint8, dim * rows // 16),
            flat(cutlass.Int64, rows // 256),
            flat(cutlass.Int32, experts),
            flat(cutlass.Int64, dim * rows // 16),
            flat(cutlass.Uint8, self.padded * rows // 16),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    @cute.kernel
    def kernel(self, data, sf, experts, ends, ordered, blocked):
        # One CTA copies one 128-feature fragment of a route tile. Iterate
        # only over the compact prefix published by the forward record.
        groups = (self.dim + 127) // 128
        job = cute.arch.block_idx()[0]
        limit = ends[self.experts - 1] // 256 * groups
        while job < limit:
            tile = job // groups
            e = experts[tile].to(cutlass.Int32)
            start = cutlass.Int32(0)
            if e > 0:
                start = ends[e - 1] // 256
            blocks = ends[e] // 256 - start
            for part in cutlass.range_constexpr(2):
                tid = cute.arch.thread_idx()[0] + 256 * part
                row = job % groups * 128 + tid // 4
                gi = tid % 4 * 4
                if row < self.dim:
                    at = (cutlass.Int64(tile) * self.dim + row) * 16 + gi
                    col = (tile - start) * 16 + gi
                    dst = cutlass.Int64(start) * self.dim * 16 + cutlass.Int64(row) * blocks * 16 + col
                    # Four adjacent K scales stay adjacent in both the public
                    # format and the MMA scale atom. Move their payload together.
                    words = cute.make_rmem_tensor((4,), cutlass.Int64)
                    cute.autovec_copy(cute.make_tensor((data.iterator + at).align(16), cute.make_layout(4)), words)
                    cute.autovec_copy(words, cute.make_tensor((ordered.iterator + dst).align(16), cute.make_layout(4)))
                    sf_atom = cutlass.Int64(row // 128) * blocks * 4 + col // 4
                    swz = (row % 32) * 16 + (row % 128 // 32) * 4 + col % 4
                    index = cutlass.Int64(start) * self.padded * 16 + sf_atom * 512 + swz
                    source = cute.recast_ptr((sf.iterator + at).align(4), dtype=cutlass.Uint32)
                    dest = cute.recast_ptr((blocked.iterator + index).align(4), dtype=cutlass.Uint32)
                    cute.arch.store(dest, cute.arch.load(source, cutlass.Uint32))
            job += cute.arch.grid_dim()[0]

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def reorder(
        self, data: cute.Tensor, sf: cute.Tensor, experts: cute.Tensor, ends: cute.Tensor, ordered: cute.Tensor, blocked: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(data, sf, experts, ends, ordered, blocked).launch(grid=(512, 1, 1), block=(256, 1, 1), stream=stream)

    def bind(self, data, sf, experts, ends):
        self.launch(
            data.view(torch.int64).view(-1),
            sf.view(torch.uint8).view(-1),
            experts,
            ends,
            self.data.view(torch.int64).view(-1),
            self.sf.view(torch.uint8).view(-1),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


class GroupedWgrad:
    def __init__(self, m, n, rows, experts, ends, device, expected_routes, norm):
        self.divisor = torch.tensor(norm, dtype=torch.float32, device=device) if norm != 1 else None
        self.a = GroupedOperand(m, rows, experts, device)
        self.b = GroupedOperand(n, rows, experts, device)
        self.lhs = self.a.data.view(torch.float4_e2m1fn_x2)
        self.rhs = self.b.data.view(torch.float4_e2m1fn_x2).T
        self.one = torch.ones(experts, dtype=torch.float32, device=device)
        sample = torch.empty((experts, m, n), dtype=torch.float32, device=device)
        # A 256-wide N tile reduces FP32 epilogue work for large matrices.
        # A second MMA CTA amortizes its setup at longer per-expert K.
        tile, cluster = (128, 128), (1, 1)
        if min(m, n) >= 1024 and m % 256 == 0 and n % 512 == 0:
            tile, cluster = ((256, 256), (2, 1)) if expected_routes >= 1024 else ((128, 256), (1, 2))
        self.op = GroupedGemmWgradBlockScaledAPI(
            self.lhs,
            self.rhs,
            self.a.sf,
            self.b.sf,
            ends,
            sample,
            sample_global_scale_a=self.one,
            sample_global_scale_b=self.one,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
            input_order="tensor_ragged",
        )
        self.op.check_support()
        self.op.compile()

    def run(self, grad, a, sfa, experts, ends):
        self.a.bind(a, sfa, experts, ends)
        self.op.execute(self.lhs, self.rhs, self.a.sf, self.b.sf, ends, grad, global_scale_a=self.one, global_scale_b=self.one)
        if self.divisor is not None:
            grad.div_(self.divisor)
