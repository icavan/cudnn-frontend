# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blackwell W4A4 tile operands. Logical scales are grouped along the reduction axis."""

from __future__ import annotations

import torch


def pack(x, norm):
    """Return K-packed E2M1 bytes and unswizzled E4M3 scales. norm is an encoding multiplier."""
    blocks = x.float().unflatten(-1, (-1, 16))
    sf = (blocks.abs().amax(-1) * norm / 6).clamp(max=448).to(torch.float8_e4m3fn)
    scale = sf.float()
    value = blocks * torch.where(scale > 0, norm / scale, 0.0).unsqueeze(-1)
    mag = value.abs()
    # At each midpoint select the code whose low bit is zero (RN-even).
    codes = (mag > 0.25).byte()
    for threshold, round_up in ((0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False)):
        codes.add_(mag >= threshold if round_up else mag > threshold)
    codes.bitwise_or_(torch.signbit(value).byte() << 3)
    codes = codes.flatten(-2)
    return codes[..., ::2] | (codes[..., 1::2] << 4), sf


def mma_operand(data):
    return data.view(torch.float4_e2m1fn_x2).unsqueeze(0).permute(1, 2, 0)


class TileGemm:
    """Prepared C = A @ B.T. The caller handles the two global factors in FP32."""

    def __init__(self, m, n, k, device, *, expert_weights=False, communication_sms=0):
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
        if expert_weights:
            self.weight_data = b.permute(2, 0, 1).view(torch.uint8)
            self.weight_sf = torch.empty((1, n, k // 16), dtype=torch.uint8, device=device)
        self.op.check_support()
        self.op.compile()

    def select_expert(self, weight, expert):
        torch.index_select(weight.data, 0, expert, out=self.weight_data)
        torch.index_select(weight.scale.view(torch.uint8), 0, expert, out=self.weight_sf)
        return self.weight_data[0], self.weight_sf[0].view(torch.float8_e4m3fn)

    def prepare_dgrad(self, experts, n, k, device):
        from cutlass import cute
        import cutlass
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        self.expert_sf = torch.empty((experts, *self.blocked[1].shape[1:]), device=device, dtype=torch.float8_e4m3fn)
        self.expert_sf_pad = None
        if n % 128 or k % 64:
            self.expert_sf_pad = torch.zeros((experts, (n + 127) // 128 * 128, (k + 63) // 64 * 4), dtype=torch.uint8, device=device)
        # Eager graphs need stable addresses. An enclosing graph can bind a
        # contiguous, 16-byte aligned caller bank directly to the TMA descriptor.
        self.fallback_weights = torch.empty((experts, n, k // 2), dtype=torch.uint8, device=device)
        op = self.op
        impl = op._kernel(sf_vec_size=16, mma_tiler_mn=op.mma_tiler_mn, cluster_shape_mn=op.cluster_shape_mn)
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(op.cluster_shape_mn[0] * op.cluster_shape_mn[1])
        fake = op._make_fake_cute_tensor_from_desc
        self.direct = cute.compile(
            impl,
            a_tensor=fake(op.a_desc),
            b_tensor=make_fake_compact_tensor(cutlass.Float4E2M1FN, (n, k, experts), stride_order=(1, 0, 2), assumed_align=16),
            sfa_tensor=fake(op.sfa_desc),
            sfb_tensor=make_fake_compact_tensor(cutlass.Float8E4M3FN, tuple(self.expert_sf.shape), stride_order=(5, 4, 3, 2, 1, 0), assumed_align=16),
            c_tensor=fake(op.c_desc),
            amax_tensor=fake(op.amax_desc),
            max_active_clusters=clusters - op.num_cluster_overlap_margin,
            stream=make_fake_stream(use_tvm_ffi_env_stream=False),
            weight_expert=make_fake_compact_tensor(cutlass.Int64, (1,), stride_order=(0,), assumed_align=8),
            options="--enable-tvm-ffi",
        )

    def bind_dgrad(self, weight, *, fixed_address=False):
        sf = weight.scale.view(torch.uint8)
        if self.expert_sf_pad is not None:
            self.expert_sf_pad[:, : sf.shape[1], : sf.shape[2]].copy_(sf)
            sf = self.expert_sf_pad
        e, rb, kb = self.expert_sf.shape[:3]
        # Write the TC layout once per complete bank per backward. Replayed
        # graphs consequently also observe in-place weight and scale changes.
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
