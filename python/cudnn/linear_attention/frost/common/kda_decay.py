# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable interval decay for KDA's 16-token tiles.

Normalized BF16 safe-gate plans whose declared lower bound proves the
factorized operands finite retain the original Tensor Core path. Every other
plan follows FLA's non-safe organization: one statically selected launch uses
FP32 prefix differences to form all intra-tile intervals directly. There is no
data-dependent detection, host decision, or replay.

KDA's beta is scalar and multiplies the triangular KK matrix after correction.
The saved backward dM matrix already includes that row beta. Consequently the
channel corrections here do not apply an additional per-channel beta factor.
Kernel schedules own publication, acquire waits and scratch reuse barriers.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cuda.bindings.driver as cuda

from cudnn.frost.tile_dsl.pointwise import fadd2, sigmoid, fp32_to_fp16
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

LOG2_E = 1.4426950408889634


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    if cutlass.const_expr(cfg.log_gate):
        return raw_gate * cutlass.Float32(LOG2_E)
    return cute.math.log2(raw_gate + cutlass.Float32(1e-10), fastmath=True)


@cute.jit
def scan_gate_prefixes(cfg: cutlass.Constexpr, gate_raw, chunk_start, batch_seqlen, cg0_a_log_exp, cg0_dt_bias_value):
    """Condition a channel's gates and scan in the original pairwise order."""
    g_prefix_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
    if cutlass.const_expr(cfg.safe_gate):
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = gate_scale(cfg, cg0_a_log_exp * (gate_raw[row] + cg0_dt_bias_value))
    else:
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = gate_scale(cfg, gate_raw[row])

    # ---- ragged tail chunk: padded rows carry no decay -----------------------
    if chunk_start + cutlass.Int32(cfg.b_t) > batch_seqlen:
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = cutlass.Float32(0.0) if chunk_start + cutlass.Int32(row) >= batch_seqlen else g_prefix_regs[row]

    prefix_acc = cutlass.Float32(0.0)
    for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
        row0 = row_pair * 2
        row1 = row0 + 1
        gate0 = g_prefix_regs[row0]
        gate1 = g_prefix_regs[row1]
        prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
        prefix1 = prefix_acc + row_pair_sum
        g_prefix_regs[row0] = prefix0
        g_prefix_regs[row1] = prefix1
        prefix_acc = prefix1
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        # The exchange aliases raw gates and permutes channels within the warp.
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    return prefix_acc, g_prefix_regs


STEEP_LOG2_PREFIX: float = -120.0
"""bf16 operands: a key channel whose inclusive log2 gate prefix inside a tile falls below this is steep for the tile.
``exp2`` of the prefix reaches the fp32 (and bf16) normal floor, ``2^-126``, six binades further down."""

STEEP_DISABLED: int = -1
"""A statically bounded BF16 plan uses the original factorized path."""

STEEP_FAST: int = 0
"""Legacy dataclass default; builders replace it with disabled or direct mode."""

STEEP_INLINE: int = 2
"""A single-launch direct-interval specialization for every other plan."""


def default_steep_mode(io_dtype, l2norm, safe_gate, gate_scale_log2, b_t, *, state_alias=False):
    """Use Tensor Cores only when declared bounds prove both operands finite."""
    if io_dtype == cutlass.BFloat16 and l2norm and safe_gate and gate_scale_log2 * b_t >= STEEP_LOG2_PREFIX:
        return STEEP_DISABLED
    return STEEP_INLINE


def has_exact_path(cfg) -> bool:
    """Whether the specialization computes direct intervals (a trace-time constant)."""
    return uses_direct_decay(cfg)


def uses_direct_decay(cfg) -> bool:
    """Whether this specialization computes every 16-token interval directly."""
    return cfg.steep_mode == STEEP_INLINE


@cute.jit
def launch_decay_kernel(
    cfg: cutlass.Constexpr,
    kernel: cutlass.Constexpr,
    args,
    grid,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr,
    min_blocks_per_mp: cutlass.Constexpr = 1,
):
    """Launch the statically selected Tensor-Core or direct-decay kernel once."""
    kernel(cfg, *args).launch(grid=grid, block=(cfg.threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl, min_blocks_per_mp=min_blocks_per_mp)


@cute.jit
def steep_exit_unless_flagged(cfg: cutlass.Constexpr, flag_slot):
    """Compatibility no-op: direct-decay plans no longer launch a replay grid."""


@cute.jit
def steep_ticket(cfg: cutlass.Constexpr, mScheduler):
    """Take the next work ticket from the ordinary scheduler word."""
    return cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))


@cute.jit
def exchange_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` fp32 gate-exchange stage."""
    seg = channel // 32
    return seg * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row ^ seg, channel - seg * 32, elem_bytes=4)


@cute.jit
def raw_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` 16-bit row-major operand tile."""
    seg = channel // 64
    return seg * (cfg.b_t * 64) + row * 64 + swizzle_xor_128b(row, channel - seg * 64, elem_bytes=2)


@cute.jit
def track_inverse_max(cfg: cutlass.Constexpr, maximum, x, y):
    """Compatibility passthrough for operand builders shared with the old replay path."""
    return maximum


@cute.jit
def flag_inverse_max(cfg: cutlass.Constexpr, maximum, flag_slot, lane_idx):
    """Compatibility no-op: direct-decay selection is static per plan."""


@cute.jit
def classify_steep_channel(cfg: cutlass.Constexpr, prefix, raw_k_ptr, channel, channel_rows: cutlass.Constexpr, flag_slot, lane_idx):
    """Return the compile-time direct-path channel mask.

    Direct plans cover every channel; bounded plans compile all consumers of
    this value away.  Do not issue a runtime ballot for a static property.
    """
    if cutlass.const_expr(uses_direct_decay(cfg)):
        valid = (1 << min(channel_rows, cfg.threads_per_warp)) - 1
        return cutlass.Boolean(True), cutlass.Int32(valid)
    return cutlass.Boolean(False), cutlass.Int32(0)


@cute.jit
def encode_gate_exchange(
    cfg: cutlass.Constexpr,
    prefix,
    raw_k_ptr,
    channel,
    channel_rows: cutlass.Constexpr,
    flag_slot,
    lane_idx,
    scan_warp,
    token_logs,
    values,
    words_ptr,
    log_last_ptr,
    exponentiate: cutlass.Constexpr,
    channel_active=True,
):
    """Encode exponentials for the factorized path or FP32 prefixes for the direct path.

    Callers own exchange stores, stage selection, and publication barriers;
    this helper never synchronizes.
    """
    channel_steep, word = classify_steep_channel(cfg, prefix, raw_k_ptr, channel, channel_rows, flag_slot, lane_idx)
    if cutlass.const_expr(has_exact_path(cfg)):
        if channel_active and lane_idx < cutlass.Int32(channel_rows):
            # DK=64 duplicates scan lanes to transpose two row halves. The
            # final prefix has one cell per channel, not per half.
            (log_last_ptr + channel).store(values[cfg.b_t - 1])
    for row in cutlass.range_constexpr(cfg.b_t):
        prefix_log = values[row]
        if cutlass.const_expr(exponentiate):
            values[row] = cute.math.exp2(values[row], fastmath=True)
        if cutlass.const_expr(has_exact_path(cfg) and row < cfg.b_t - 1):
            if channel_steep:
                values[row] = prefix_log
    return word


@cute.jit
def steep_tile_mask(cfg: cutlass.Constexpr, words_ptr):
    """Return whether this compile-time specialization uses direct intervals."""
    return cutlass.Int32(1 if has_exact_path(cfg) else 0)


@cute.jit
def lane_steep_bits(cfg: cutlass.Constexpr, words_ptr, lane_in_row_group, channel_rows: cutlass.Constexpr[int]):
    """Return all channel bits owned by a row-stage lane in a direct plan."""
    return cutlass.Int32((1 << (cfg.d_k // 8)) - 1 if has_exact_path(cfg) else 0)


@cute.jit
def steep_channel_bit(words_ptr, channel, channel_rows: cutlass.Constexpr[int]):
    """Every channel reaching this helper belongs to the direct specialization."""
    return cutlass.Int32(1)


@cute.jit
def steep_token_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """Read a steep channel's conditioned per-token log2 gate, including zero padding."""
    value = (log_last_ptr + channel).load()
    if row < cfg.b_t - 1:
        value = (exchange_ptr + exchange_offset(cfg, row, channel)).load()
    return value


@cute.jit
def steep_interval_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, start, end, channel):
    """Return the log2 decay over ``[start, end)`` from FP32 prefixes."""
    total = steep_token_log(cfg, exchange_ptr, log_last_ptr, end - 1, channel)
    if start > 0:
        total = total - steep_token_log(cfg, exchange_ptr, log_last_ptr, start - 1, channel)
    return total


@cute.jit
def steep_decay_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K decay / Q decay scale on ``row``, ``exp2(g[row])``, at most 1 (may underflow to zero)."""
    return cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, 0, row + 1, channel), fastmath=True)


@cute.jit
def steep_restore_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K inv / K restore scale on ``row``, ``exp2(g[BT - 1] - g[row])``, at most 1."""
    return cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, row + 1, cfg.b_t, channel), fastmath=True)


@cute.jit
def gradient_operand_scales(cfg: cutlass.Constexpr, row, channel, value, exchange_ptr, log_last_ptr):
    """Return decay and restore scales before direct gradient assembly.

    Direct plans use bounded prefix/suffix factors. Their intra-tile inverse
    contribution is computed explicitly, so no inverse operand scale is needed.
    """
    decay = value
    restore = cutlass.Float32(0.0)
    if cutlass.const_expr(uses_direct_decay(cfg)):
        decay = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
        restore = steep_restore_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
    return decay, restore


@cute.jit
def row_operand_scales(cfg: cutlass.Constexpr, bits, reg_idx, row, channel, value, last_value, exchange_ptr, log_last_ptr):
    """Return decay, inverse and restore-multiplier scales before operand construction.

    The fast specialization retains the original factorization. Exact channels
    put the bounded restore scale in K inv; K restore then uses multiplier one.
    All other channels retain the original operation order and rounding.
    """
    decay = value
    inverse = cute.math.rcp(value, approx=True, ftz=True)
    restore_multiplier = last_value
    if cutlass.const_expr(uses_direct_decay(cfg)):
        decay = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
        inverse = steep_restore_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
        restore_multiplier = cutlass.Float32(1.0)
    return decay, inverse, restore_multiplier


@cute.jit
def steep_row_corrections(
    cfg: cutlass.Constexpr,
    bits,
    row,
    lane_in_row_group,
    k_norm,
    q_scale,
    exchange_ptr,
    log_last_ptr,
    raw_k_ptr,
    raw_q_ptr,
    k_norm_ptr,
    k_inv_ptr,
    k_decay_ptr,
    q_decay_ptr,
    corr_ptr,
):
    """Form the direct path's KK and A entries for tile row ``row`` in ``corr_ptr``.

    Called by all lanes of the row stage. ``bits`` is the lane's
    :func:`lane_steep_bits`; ``k_norm`` / ``q_scale`` are the row's K
    normalization and Q factor (normalization times attention scale where the
    A tile carries it); ``raw_q_ptr`` is ``None`` without an A tile.
    ``corr_ptr`` is ``[2, BT, BT]`` FP32, the KK rows then the A rows.
    Each row group reduces in a fixed order and one lane stores each entry, for
    ``s <= row`` and every channel,

        A[row, s]  = sum_d q_n[row, d] k_n[s, d] 2^(g[row, d] - g[s, d])
        KK[row, s] = sum_d k_n[row, d] k_n[s, d] 2^(g[row, d] - g[s, d])  (s < row)

    The row-stage barrier must make all raw operand rows visible before this call."""
    with_a = cutlass.const_expr(raw_q_ptr is not None)
    kk_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    attn_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    for source_row in cutlass.range_constexpr(cfg.b_t):
        kk_regs[source_row] = cutlass.Float32(0.0)
        attn_regs[source_row] = cutlass.Float32(0.0)
    for j in cutlass.range(cfg.d_k // 64 * 8, unroll=1):
        channel = (j // 8) * 64 + lane_in_row_group * 8 + (j % 8)
        x = (raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * k_norm
        if cutlass.const_expr(with_a):
            z = (raw_q_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * q_scale
        for step in cutlass.range_constexpr(cfg.b_t):
            source_row = cfg.b_t - 1 - step
            if source_row <= row:
                interval = cutlass.Float32(0.0)
                if source_row < row:
                    interval = steep_interval_log(cfg, exchange_ptr, log_last_ptr, source_row + 1, row + 1, channel)
                k_s = (raw_k_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                if cutlass.const_expr(k_norm_ptr is not None):
                    k_s = k_s * (k_norm_ptr + source_row).load()
                k_decayed = k_s * cute.math.exp2(interval, fastmath=True)
                if source_row < row:
                    kk_regs[source_row] = kk_regs[source_row] + x * k_decayed
                if cutlass.const_expr(with_a):
                    attn_regs[source_row] = attn_regs[source_row] + z * k_decayed
    for source_row in cutlass.range_constexpr(cfg.b_t):
        kk = kk_regs[source_row]
        attn = attn_regs[source_row]
        for step in cutlass.range_constexpr(3):
            distance = 4 >> step
            kk = kk + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, kk, distance, 31, kind=nvvm.Shfl.BFLY))
            if cutlass.const_expr(with_a):
                attn = attn + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, attn, distance, 31, kind=nvvm.Shfl.BFLY))
        if lane_in_row_group == 0:
            (corr_ptr + row * cfg.b_t + source_row).store(kk)
            if cutlass.const_expr(with_a):
                (corr_ptr + cfg.b_t * cfg.b_t + row * cfg.b_t + source_row).store(attn)


@cute.jit
def add_steep_corrections(cfg: cutlass.Constexpr, acc, corr_ptr, lane_idx):
    """Merge the direct/correction tile into an m16n16 accumulator fragment."""
    for accum_idx in cutlass.range_constexpr(8):
        row = lane_idx // 4
        if cutlass.const_expr(accum_idx % 4 >= 2):
            row = row + cutlass.Int32(8)
        col = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col = col + cutlass.Int32(1)
        value = (corr_ptr + row * cfg.b_t + col).load()
        acc[accum_idx] = value
    # Callers release this stage through an elected lane. Join every reader
    # before that lane can let the producer overwrite the correction tile.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def direct_grad_terms(cfg: cutlass.Constexpr, channel, t, scale, exchange_ptr, log_last_ptr, raw_q_ptr, raw_k_ptr, q_norm_ptr, k_norm_ptr, dadm_ptr):
    """The direct path's complete intra-tile gradients for one channel and token.

    ``dadm_ptr`` holds the chunk's strict-lower dA and dM tiles, ``[2, BT, BT]`` in the io dtype, exactly as the
    tcgen05 GEMMs read them.  With ``F(t, s) = 2^(g[t] - g[s])`` and ``K inv[s] = k_n[s] * 2^(g[BT - 1] - g[s])``
    forms

        dq_attn  = sum_{s<t} dA[t, s] k_n[s] F(t, s)
        dk_decay = -sum_{s<t} dM[t, s] k_n[s] F(t, s)
        dk_inv   = sum_{s>t} (scale q_n[s] dA[s, t] - k_n[s] dM[s, t]) F(s, t)

    Every exponent of F is at most zero.  The direct specialization skips the
    four factorized intra-tile Tensor-Core GEMMs, so this helper returns the
    complete terms, not corrections against work that was already done.
    Callers add the two dK terms in their original accumulation order."""
    dq_attn = cutlass.Float32(0.0)
    dkd_attn = cutlass.Float32(0.0)
    dk_inv = cutlass.Float32(0.0)
    tile = cutlass.const_expr(cfg.b_t * cfg.b_t)
    for s in cutlass.range(cfg.b_t, unroll=1):
        if s != t:
            k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
            if cutlass.const_expr(cfg.l2norm):
                k_s = k_s * (k_norm_ptr + s).load()
            if s < t:
                decay = cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, s + 1, t + 1, channel), fastmath=True)
                da = (dadm_ptr + t * cfg.b_t + s).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + t * cfg.b_t + s).load().to(cutlass.Float32)
                dq_attn = dq_attn + da * (k_s * decay)
                dkd_attn = dkd_attn + dm * (k_s * decay)
            else:
                q_s = (raw_q_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    q_s = q_s * (q_norm_ptr + s).load()
                decay = cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, t + 1, s + 1, channel), fastmath=True)
                da = (dadm_ptr + s * cfg.b_t + t).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + s * cfg.b_t + t).load().to(cutlass.Float32)
                dk_inv = dk_inv + ((q_s * scale) * da - (k_s) * dm) * decay
    return scale * dq_attn, dk_inv, -dkd_attn


@cute.jit
def pack_dstate_operand(cfg: cutlass.Constexpr, original, decayed, first_channel, words_ptr, steep_any):
    """Pack the carry once; steep channels use the original because K inv carries their decay.

    Both inputs are already in registers. Leave the decayed FP32 accumulator
    unchanged and let the caller publish the packed operand under its barriers.
    """
    operand = cutlass.Array(cutlass.Float32, 32, alignment=16)
    for channel in cutlass.range_constexpr(32):
        operand[channel] = decayed[channel]
    if cutlass.const_expr(uses_direct_decay(cfg)):
        for channel in cutlass.range_constexpr(32):
            operand[channel] = original[channel]
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for pc in cutlass.range_constexpr(16):
        packed[pc] = fp32_to_fp16(operand[2 * pc], operand[2 * pc + 1], dtype=cfg.io_dtype)
    return packed
