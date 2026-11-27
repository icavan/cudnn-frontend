# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blackwell W4A4 tile operands. Logical scales are grouped along the reduction axis."""

from __future__ import annotations

import torch

from ._math import NVFP4_INVERSE_SPLIT


class MasterWeightPacker:
    """Reuse device FP32 constants over the tiles of one master-weight bank."""

    def __init__(self, norm, device):
        if isinstance(norm, torch.Tensor):
            self.norm = norm.to(dtype=torch.float32, device=device)
            self.input_factor = torch.where(self.norm > NVFP4_INVERSE_SPLIT, self.norm.new_full((), 512.0), 1.0)
        else:
            self.norm = torch.full((), norm, dtype=torch.float32, device=device)
            self.input_factor = 512.0 if norm > NVFP4_INVERSE_SPLIT else None
        self.fp4_max = self.norm.new_full((), 6.0)
        self.payload_norm = self.norm / self.input_factor if self.input_factor is not None else self.norm

    def __call__(self, x):
        blocks = x.float().unflatten(-1, (-1, 16))
        sf = (blocks.abs().amax(-1) * self.norm / self.fp4_max).clamp(max=448).to(torch.float8_e4m3fn)
        scale = sf.float()
        if self.input_factor is not None:
            blocks = blocks * self.input_factor
        value = blocks * torch.where(scale > 0, self.payload_norm / scale, 0.0).unsqueeze(-1)
        mag = value.abs()
        # At each midpoint select the code whose low bit is zero (RN-even).
        codes = (mag > 0.25).byte()
        for threshold, round_up in ((0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False)):
            codes.add_(mag >= threshold if round_up else mag > threshold)
        codes.bitwise_or_(torch.signbit(value).byte() << 3)
        codes = codes.flatten(-2)
        return codes[..., ::2] | (codes[..., 1::2] << 4), sf


def pack(x, norm):
    """Return K-packed E2M1 bytes and unswizzled E4M3 scales."""
    return MasterWeightPacker(norm, x.device)(x)


def mma_operand(data):
    return data.view(torch.float4_e2m1fn_x2).unsqueeze(0).permute(1, 2, 0)


class TileGemm:
    """Prepared C = A @ B.T. The caller handles the two global factors in FP32."""

    def __init__(self, m, n, k, device, *, communication_sms=0):
        from cudnn.gemm.cutedsl.dense.amax.api import GemmAmaxSm100

        def sample(rows):
            data = torch.empty((rows, k // 2), dtype=torch.uint8, device=device)
            sf = torch.empty((1, (rows + 127) // 128, (k + 63) // 64, 32, 4, 4), dtype=torch.float8_e4m3fn, device=device)
            return mma_operand(data), sf

        a, sfa = sample(m)
        b, sfb = sample(n)
        self.output = torch.empty((1, m, n), dtype=torch.float32, device=device).permute(1, 2, 0)
        self.amax = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        self.op = GemmAmaxSm100(a, b, sfa, sfb, self.output, self.amax, sf_vec_size=16)
        self.op.num_cluster_overlap_margin = max(self.op.num_cluster_overlap_margin, communication_sms)
        self.blocked = (sfa, sfb)
        self.padded = tuple(torch.zeros((t.shape[1] * 128, t.shape[2] * 4), dtype=torch.uint8, device=device) for t in self.blocked)
        self.op.check_support()
        self.op.compile()

    def prepare_expert_bank(self, experts, n, k, device):
        self.expert_sf = torch.empty((experts, *self.blocked[1].shape[1:]), device=device, dtype=torch.float8_e4m3fn)
        self.expert_sf_pad = None
        if n % 128 or k % 64:
            self.expert_sf_pad = torch.zeros((experts, (n + 127) // 128 * 128, (k + 63) // 64 * 4), dtype=torch.uint8, device=device)
        # Eager graphs need stable addresses. An enclosing graph can bind a
        # contiguous, 16-byte aligned caller bank directly to the TMA descriptor.
        self.fallback_weights = torch.empty((experts, n, k // 2), dtype=torch.uint8, device=device)

    def prepare_dgrad(self, experts, n, k, device):
        self.prepare_expert_bank(experts, n, k, device)
        self.direct = self.compile_dgrad(experts, n, k, 1)

    def compile_dgrad(self, experts, n, k, tiles, *, live=False, peer_store=None):
        from cutlass import cute
        import cutlass
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        op = self.op
        impl = op._kernel(sf_vec_size=16, mma_tiler_mn=op.mma_tiler_mn, cluster_shape_mn=op.cluster_shape_mn, collect_amax=False)
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(op.cluster_shape_mn[0] * op.cluster_shape_mn[1])
        fake = op._make_fake_cute_tensor_from_desc

        def matrix(dtype, m, kk):
            return make_fake_compact_tensor(dtype, (m, kk, 1), stride_order=(1, 0, 2), assumed_align=16)

        store_kwargs = {}
        if peer_store is not None:
            if op.cluster_shape_mn != (1, 1) or n % op.mma_tiler_mn[1]:
                raise ValueError("direct dX return requires single-CTA clusters and full output column tiles")

            def vector(dtype, length):
                return make_fake_compact_tensor(dtype, (length,), stride_order=(0,), assumed_align=dtype.width // 8)

            store_kwargs = dict(
                store_subtile=peer_store.store,
                store_args=(vector(cutlass.Int64, peer_store.peer_count), vector(cutlass.Int64, tiles * 256), vector(cutlass.Float32, 1)),
            )

        return cute.compile(
            impl,
            a_tensor=matrix(cutlass.Float4E2M1FN, tiles * 256, k),
            b_tensor=make_fake_compact_tensor(cutlass.Float4E2M1FN, (n, k, experts), stride_order=(1, 0, 2), assumed_align=16),
            sfa_tensor=make_fake_compact_tensor(
                cutlass.Float8E4M3FN, (1, tiles * 2, *self.blocked[0].shape[2:]), stride_order=(5, 4, 3, 2, 1, 0), assumed_align=16
            ),
            sfb_tensor=make_fake_compact_tensor(cutlass.Float8E4M3FN, tuple(self.expert_sf.shape), stride_order=(5, 4, 3, 2, 1, 0), assumed_align=16),
            c_tensor=matrix(cutlass.Float32, tiles * 256, n),
            amax_tensor=fake(op.amax_desc),
            max_active_clusters=clusters - op.num_cluster_overlap_margin,
            stream=make_fake_stream(use_tvm_ffi_env_stream=False),
            **store_kwargs,
            weight_expert=make_fake_compact_tensor(cutlass.Int64, (tiles,), stride_order=(0,), assumed_align=8),
            **(
                {"route_bounds": make_fake_compact_tensor(cutlass.Int64, (2,), stride_order=(0,), assumed_align=8), "route_offset": cutlass.Int32(0)}
                if live
                else {}
            ),
            options="--enable-tvm-ffi",
        )

    def bind_dgrad(self, weight, *, fixed_address=False):
        """Bind a bank, staging only for an address-stable eager graph or a view.

        The two prebuilt eager WHILE graphs contain TMA descriptors whose base
        addresses were captured during prepare_training(), before caller weights
        existed. They cannot consume a newly returned tensor address: eager replay
        must refresh fallback_weights. An enclosing caller graph instead captures
        the caller's aligned contiguous address directly. Strided/unaligned views
        use an explicit prepared staging path in this training convenience API.
        """
        sf = weight.scale.view(torch.uint8)
        if self.expert_sf_pad is not None:
            self.expert_sf_pad[:, : sf.shape[1], : sf.shape[2]].copy_(sf)
            sf = self.expert_sf_pad
        e, rb, kb = self.expert_sf.shape[:3]
        # Write the TC layout once per complete bank per backward. Replayed
        # graphs consequently also observe in-place weight and scale changes.
        if sf.is_contiguous() and sf.storage_offset() % 4 == 0 and sf.data_ptr() % 4 == 0:
            # The four consecutive K scales are already one Tensor Core word.
            # Move that word intact when transposing the outer scale axes.
            # A contiguous singleton expert axis may have any stride. Flatten
            # before changing itemsize so unused strides do not restrict views.
            words = sf.view(-1).view(torch.int32).view(e, rb, 4, 32, kb)
            self.expert_sf.view(torch.int32).squeeze(-1).copy_(words.permute(0, 1, 4, 3, 2))
        else:
            self.expert_sf.view(torch.uint8).copy_(sf.unflatten(1, (rb, 4, 32)).unflatten(-1, (kb, 4)).permute(0, 1, 4, 3, 2, 5))
        data = weight.data
        if fixed_address or not data.is_contiguous() or data.data_ptr() % 16:
            self.fallback_weights.copy_(data)
            data = self.fallback_weights
        return data.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)

    def dgrad(self, lhs, bank, expert, ready=None):
        from cuda.bindings import driver as cuda

        a, sf = lhs
        self.block_scale(sf, self.padded[0], self.blocked[0])
        if ready is not None:
            ready.record(torch.cuda.current_stream())
        self.direct(
            mma_operand(a), bank, self.blocked[0], self.expert_sf, self.output, self.amax, cuda.CUstream(torch.cuda.current_stream().cuda_stream), expert
        )
        return self.output.squeeze(-1)

    @staticmethod
    def block_scale(sf, pad, blocked):
        pad[: sf.shape[0], : sf.shape[1]].copy_(sf.view(torch.uint8))
        atom = pad.view(blocked.shape[1], 4, 32, blocked.shape[2], 4).permute(0, 3, 2, 1, 4)
        blocked[0].view(torch.uint8).copy_(atom)

    def __call__(self, lhs, rhs, *, ready=None):
        a, sfa = lhs
        b, sfb = rhs
        for sf, pad, blocked in zip((sfa, sfb), self.padded, self.blocked):
            self.block_scale(sf, pad, blocked)
        if ready is not None:
            ready.record(torch.cuda.current_stream())
        self.op.execute(mma_operand(a), mma_operand(b), *self.blocked, self.output, self.amax)
        return self.output.squeeze(-1)
