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

"""Stable interval-decay operands for KDA's 16-token tiles.

Safe-gate plans preserve the original MMA operands and replay only hazardous
channels with directly summed token logs. Non-safe plans take the equivalent
FLA-style bounded path statically: each token decay is materialized once in
FP32, then CUDA cores multiply the interval factors directly. This avoids
reciprocal overflow and ``0 * inf`` without a data-dependent launch or host
check, while also avoiding one transcendental evaluation per QK/KK element.

KDA's beta is scalar and multiplies the triangular KK matrix after correction.
The saved backward dM matrix already includes that row beta. Consequently the
channel corrections here do not apply an additional per-channel beta factor.
Kernel schedules own publication, acquire waits and scratch reuse barriers.
"""

import dataclasses

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
    """Condition a channel's gates and scan in the original pairwise order.

    Exact kernels retain token logs in ``gate_raw``. The caller owns exchange
    stores, scan ballots and publication barriers.
    """
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

    if cutlass.const_expr(has_exact_path(cfg)):
        for row in cutlass.range_constexpr(cfg.b_t):
            gate_raw[row] = g_prefix_regs[row]

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

STEEP_LOG2_PREFIX_FP16: float = -14.0
"""fp16 operands: K inv = k * 2^-g reaches the fp16 maximum, ``65504 ~ 2^16``, two binades further down."""


def steep_log2_prefix(io_dtype) -> float:
    """The steep threshold for the operand dtype (a trace-time constant)."""
    return STEEP_LOG2_PREFIX_FP16 if io_dtype == cutlass.Float16 else STEEP_LOG2_PREFIX


STEEP_DISABLED: int = -1
"""A statically bounded BF16 plan needs neither checks nor replay."""

STEEP_FAST: int = 0
"""Kernel ``cfg.steep_mode``: the original arithmetic plus the steep vote; a steep tile raises the call's steep flag.
``host`` follows the launch with a ``STEEP_EXACT`` launch of the same kernel."""

STEEP_EXACT: int = 1
"""The exact kernel of a ``STEEP_FAST`` launch: every thread exits on entry unless the flag is raised, else it redoes
the whole launch with the steep path, pulling tiles from its own ticket."""

STEEP_FLAG_SLOT: int = 1
"""The steep flag and the exact launch's ticket share the second int32 of the kernel's scheduler slot, which the
prologue zeroes with the first: 0 means no steep tile, and the exact launch's tickets start at the 1 the fast launch
stored."""

STEEP_INLINE: int = 2
"""Single-launch exact specialization for plans whose state may alias their input."""


def default_steep_mode(io_dtype, l2norm, safe_gate, gate_scale_log2, b_t, *, state_alias=False):
    """Omit checks only when the declared gate and normalized-key bounds prove safety."""
    if not safe_gate:
        return STEEP_INLINE
    if io_dtype == cutlass.BFloat16 and l2norm and safe_gate and gate_scale_log2 * b_t >= STEEP_LOG2_PREFIX:
        return STEEP_DISABLED
    return STEEP_INLINE if state_alias else STEEP_FAST


def has_exact_path(cfg) -> bool:
    """Whether the specialization includes interval corrections (a trace-time constant)."""
    return cfg.steep_mode in (STEEP_EXACT, STEEP_INLINE)


def steep_launch_cfg(cfg, launch_idx):
    """Compile-time launch order: the requested specialization, then its exact twin."""
    return cfg if launch_idx == 0 else dataclasses.replace(cfg, steep_mode=STEEP_EXACT)


@cute.jit
def state_may_overlap(lhs, rhs):
    """Conservative address-range overlap for native strided states.

    This runs in the compiled host, not on the GPU or in Python execute(). It
    examines current bindings, includes layout holes, and treats empty views as
    disjoint. No tensor data or device flag is read.
    """
    overlap = cutlass.Boolean(False)
    if cutlass.const_expr(lhs is not None and rhs is not None):
        lhs_lo, rhs_lo = cutlass.Int64(0), cutlass.Int64(0)
        lhs_hi, rhs_hi = cutlass.Int64(1), cutlass.Int64(1)
        nonempty = cutlass.Boolean(True)
        for axis in cutlass.range_constexpr(4):
            lhs_span = (cutlass.Int64(lhs.shape[axis]) - 1) * cutlass.Int64(lhs.stride[axis])
            rhs_span = (cutlass.Int64(rhs.shape[axis]) - 1) * cutlass.Int64(rhs.stride[axis])
            lhs_lo += cutlass.min(lhs_span, cutlass.Int64(0))
            rhs_lo += cutlass.min(rhs_span, cutlass.Int64(0))
            lhs_hi += cutlass.max(lhs_span, cutlass.Int64(0))
            rhs_hi += cutlass.max(rhs_span, cutlass.Int64(0))
            nonempty = nonempty & (lhs.shape[axis] > 0) & (rhs.shape[axis] > 0)
        lhs_base, rhs_base = lhs.iterator.toint(), rhs.iterator.toint()
        lhs_start = lhs_base + lhs_lo * (lhs.element_type.width // 8)
        rhs_start = rhs_base + rhs_lo * (rhs.element_type.width // 8)
        lhs_end = lhs_base + lhs_hi * (lhs.element_type.width // 8)
        rhs_end = rhs_base + rhs_hi * (rhs.element_type.width // 8)
        overlap = nonempty & (lhs_start < rhs_end) & (rhs_start < lhs_end)
    return overlap


@cute.jit
def launch_decay_kernel(
    cfg: cutlass.Constexpr,
    kernel: cutlass.Constexpr,
    args,
    grid,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr,
    may_alias,
    min_blocks_per_mp: cutlass.Constexpr = 1,
):
    """Choose precision launches from current state bindings in the native host.

    Overlapping state views require one inline corrected launch. Disjoint views
    retain the small fast kernel and its flag-gated twin; this decision must not
    be cached from warmup pointers. Statically bounded plans still launch once.
    """
    if cutlass.const_expr(cfg.steep_mode == STEEP_INLINE and cfg.safe_gate):
        if may_alias:
            kernel(cfg, *args).launch(grid=grid, block=(cfg.threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl, min_blocks_per_mp=min_blocks_per_mp)
        else:
            for launch_idx in cutlass.range_constexpr(2):
                kernel(steep_launch_cfg(dataclasses.replace(cfg, steep_mode=STEEP_FAST), launch_idx), *args).launch(
                    grid=grid, block=(cfg.threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl, min_blocks_per_mp=min_blocks_per_mp
                )
    else:
        for launch_idx in cutlass.range_constexpr(2 if cfg.steep_mode == STEEP_FAST else 1):
            kernel(steep_launch_cfg(cfg, launch_idx), *args).launch(
                grid=grid, block=(cfg.threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl, min_blocks_per_mp=min_blocks_per_mp
            )


@cute.jit
def steep_exit_unless_flagged(cfg: cutlass.Constexpr, flag_slot):
    """``STEEP_EXACT`` kernel entry, after the grid-dependency wait: leave unless the fast launch raised the flag.

    Do not trigger dependents here: the exact grid must complete before later
    kernels can use its corrected outputs. This also applies to the empty grid."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        # Other CTAs may already be atomically incrementing this same word as
        # their replay ticket. Use a strong load, not a racing ordinary load.
        flag = nvvm.inline_ptx(
            "ld.relaxed.gpu.global.u32 {$w0}, [{$r0}];",
            write_only_types=[cutlass.Int32],
            read_only_args=[flag_slot.iterator + STEEP_FLAG_SLOT],
        )
        if flag == cutlass.Int32(0):
            nvvm.exit()


@cute.jit
def steep_raise_flag(cfg: cutlass.Constexpr, flag_slot, steep_word, lane_idx):
    """``STEEP_FAST`` scan: a warp whose ballot found a steep channel raises the flag (idempotent, never read back)."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_FAST):
        if steep_word != 0:
            if lane_idx == 0:
                nvvm.atomicrmw("or", flag_slot.iterator + STEEP_FLAG_SLOT, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu")


@cute.jit
def steep_ticket(cfg: cutlass.Constexpr, mScheduler):
    """The next ticket off the scheduler slot: the first int32, or under ``STEEP_EXACT`` the flag slot past its 1."""
    fetched = cutlass.Int32(0)
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator + STEEP_FLAG_SLOT, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu")) - 1
    else:
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
    return fetched


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
def channel_needs_exact_decay(cfg: cutlass.Constexpr, prefix, raw_k_ptr, channel):
    """Bound the inverse operand over a channel before selecting its decay path.

    Normalized keys have magnitude at most one. Exact/inline kernels otherwise
    scan the raw tile and reserve one binade for reciprocal rounding. FAST uses
    the prefix bound here and checks its actual inverse registers later.
    The last prefix is the minimum, so the exact bound covers every row.
    """
    threshold = cutlass.Float32(steep_log2_prefix(cfg.io_dtype))
    # FAST checks the actual inverse values while they are already in registers.
    # Only a replay/inline launch needs this channel-wise conservative bound.
    if cutlass.const_expr(not cfg.l2norm and has_exact_path(cfg)):
        max_key = cutlass.Float32(1.0)
        for row in cutlass.range(cfg.b_t, unroll=1):
            key = (raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            max_key = cute.math.max(max_key, cute.math.abs(key))
        limit = cutlass.Float32(15.0 if cfg.io_dtype == cutlass.Float16 else 127.0)
        threshold = cute.math.max(threshold, cute.math.log2(max_key) - limit)
    return prefix < threshold


@cute.jit
def track_inverse_max(cfg: cutlass.Constexpr, maximum, x, y):
    """Track unnormalized FAST operands without loading the raw key tile again."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_FAST and not cfg.l2norm):
        maximum = cute.math.max(maximum, cute.math.max(cute.math.abs(x), cute.math.abs(y)))
    return maximum


@cute.jit
def flag_inverse_max(cfg: cutlass.Constexpr, maximum, flag_slot, lane_idx):
    """Request replay before returning any inverse operand outside its dtype range."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_FAST and not cfg.l2norm):
        limit = cutlass.Float32(65504.0 if cfg.io_dtype == cutlass.Float16 else 3.3895313892515355e38)
        word = cutlass.Int32(nvvm.vote_sync(0xFFFFFFFF, maximum > limit, nvvm.VoteSync.BALLOT))
        steep_raise_flag(cfg, flag_slot, word, lane_idx)


@cute.jit
def classify_steep_channel(cfg: cutlass.Constexpr, prefix, raw_k_ptr, channel, channel_rows: cutlass.Constexpr, flag_slot, lane_idx):
    """Classify a scan lane, mask duplicate lanes, and flag a FAST relaunch.

    Callers retain ownership of stage offsets, scratch stores and barriers.
    """
    channel_steep = cutlass.Boolean(False)
    if cutlass.const_expr(has_exact_path(cfg) and not cfg.safe_gate):
        channel_steep = cutlass.Boolean(True)
    elif cutlass.const_expr(cfg.steep_mode != STEEP_DISABLED):
        channel_steep = channel_needs_exact_decay(cfg, prefix, raw_k_ptr, channel)
    word = cutlass.Int32(0)
    if cutlass.const_expr(cfg.steep_mode != STEEP_DISABLED):
        word = cutlass.Int32(nvvm.vote_sync(0xFFFFFFFF, channel_steep, nvvm.VoteSync.BALLOT))
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        word = word & cutlass.Int32((1 << channel_rows) - 1)
    steep_raise_flag(cfg, flag_slot, word, lane_idx)
    return channel_steep, word


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
    """Classify a channel and encode its exchange registers and scan metadata.

    ``values`` holds log prefixes when ``exponentiate`` is true, otherwise their
    exponentials. Safe-gate replay retains token logs so it can recover
    intervals exactly. The static non-safe path stores each FP32 token decay in
    rows 0 .. BT - 2 and stores the final token decay in ``log_last_ptr``;
    consumers multiply only factors from their causal interval. The last
    exchange row still holds the whole-tile state-decay exponential. Callers own
    exchange stores, stage selection and publication barriers; this helper never
    synchronizes.
    """
    channel_steep, word = classify_steep_channel(cfg, prefix, raw_k_ptr, channel, channel_rows, flag_slot, lane_idx)
    if cutlass.const_expr(has_exact_path(cfg)):
        if lane_idx == 0:
            (words_ptr + scan_warp).store(word)
        if word != 0:
            if channel_steep:
                # DK=64 duplicates scan lanes to transpose two row halves. The
                # final log/prefix has one cell per channel, not per half.
                if channel_active and lane_idx < cutlass.Int32(channel_rows):
                    last = token_logs[cfg.b_t - 1]
                    if cutlass.const_expr(not cfg.safe_gate):
                        last = cute.math.exp2(last, fastmath=True)
                    (log_last_ptr + channel).store(last)
    for row in cutlass.range_constexpr(cfg.b_t):
        if cutlass.const_expr(has_exact_path(cfg) and row < cfg.b_t - 1):
            if channel_steep:
                if cutlass.const_expr(cfg.safe_gate):
                    values[row] = token_logs[row]
                else:
                    values[row] = cute.math.exp2(token_logs[row], fastmath=True)
            elif cutlass.const_expr(exponentiate):
                values[row] = cute.math.exp2(values[row], fastmath=True)
        elif cutlass.const_expr(exponentiate):
            values[row] = cute.math.exp2(values[row], fastmath=True)
    return word


@cute.jit
def steep_tile_mask(cfg: cutlass.Constexpr, words_ptr):
    """Read the tile's four scan-warp ballots; FAST never reads exact-path scratch."""
    mask = cutlass.Int32(0)
    if cutlass.const_expr(has_exact_path(cfg)):
        mask = (words_ptr.load() | (words_ptr + 1).load()) | ((words_ptr + 2).load() | (words_ptr + 3).load())
    return mask


@cute.jit
def lane_steep_bits(cfg: cutlass.Constexpr, words_ptr, lane_in_row_group, channel_rows: cutlass.Constexpr[int]):
    """The steep channels a row-stage lane owns: bit ``h * 8 + j`` is channel ``h * 64 + lane_in_row_group * 8 + j``.

    ``words_ptr`` holds one ballot word per scan warp; channel ``c`` is bit ``c % channel_rows`` of word
    ``c // channel_rows``."""
    bits = cutlass.Int32(0)
    for dim_half in cutlass.range_constexpr(cfg.d_k // 64):
        first = dim_half * 64 + lane_in_row_group * 8
        word = (words_ptr + first // channel_rows).load()
        bits = bits | (((word >> (first % channel_rows)) & 0xFF) << (dim_half * 8))
    return bits


@cute.jit
def steep_channel_bit(words_ptr, channel, channel_rows: cutlass.Constexpr[int]):
    """1 if ``channel`` is steep in the ballot words at ``words_ptr`` (one word per scan warp of ``channel_rows`` channels), else 0."""
    return ((words_ptr + channel // channel_rows).load() >> (channel % channel_rows)) & 1


@cute.jit
def steep_token_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """Read a steep channel's conditioned per-token log2 gate, including zero padding."""
    value = (log_last_ptr + channel).load()
    if row < cfg.b_t - 1:
        value = (exchange_ptr + exchange_offset(cfg, row, channel)).load()
    return value


@cute.jit
def direct_token_decay(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """Read one non-safe channel's conditioned FP32 decay for ``row``."""
    value = (log_last_ptr + channel).load()
    if row < cfg.b_t - 1:
        value = (exchange_ptr + exchange_offset(cfg, row, channel)).load()
    return value


@cute.jit
def direct_interval_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, start, end, channel):
    """Multiply the bounded non-safe token decays in ``[start, end)``."""
    scale = cutlass.Float32(1.0)
    for token in cutlass.range(start, end, unroll=1):
        scale = scale * direct_token_decay(cfg, exchange_ptr, log_last_ptr, token, channel)
    return scale


@cute.jit
def steep_interval_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, start, end, channel):
    """Sum [start, end) directly: prefix subtraction loses mild gates after large spikes."""
    total = cutlass.Float32(0.0)
    for token in cutlass.range(start, end, unroll=1):
        total = total + steep_token_log(cfg, exchange_ptr, log_last_ptr, token, channel)
    return total


@cute.jit
def steep_decay_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K decay / Q decay scale on ``row``, ``exp2(g[row])``, at most 1 (may underflow to zero)."""
    if cutlass.const_expr(not cfg.safe_gate):
        return direct_interval_scale(cfg, exchange_ptr, log_last_ptr, 0, row + 1, channel)
    else:
        log_scale = steep_interval_log(cfg, exchange_ptr, log_last_ptr, 0, row + 1, channel)
    return cute.math.exp2(log_scale, fastmath=True)


@cute.jit
def steep_restore_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K inv / K restore scale on ``row``, ``exp2(g[BT - 1] - g[row])``, at most 1."""
    if cutlass.const_expr(not cfg.safe_gate):
        return direct_interval_scale(cfg, exchange_ptr, log_last_ptr, row + 1, cfg.b_t, channel)
    else:
        log_scale = steep_interval_log(cfg, exchange_ptr, log_last_ptr, row + 1, cfg.b_t, channel)
    return cute.math.exp2(log_scale, fastmath=True)


@cute.jit
def gradient_operand_scales(cfg: cutlass.Constexpr, channel_steep, row, channel, value, exchange_ptr, log_last_ptr):
    """Return decay, inverse-gradient and restore scales before gradient assembly.

    Steep channels use bounded prefix/suffix factors. Their inverse-gradient
    contribution comes from the exact intra-tile correction, so its scale is zero.
    """
    decay = value
    inverse = cutlass.Float32(0.0)
    restore = cutlass.Float32(0.0)
    if channel_steep != 0:
        decay = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
        restore = steep_restore_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
    else:
        inverse = cute.math.rcp(value, approx=True, ftz=True)
        restore = inverse
    return decay, inverse, restore


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
    if cutlass.const_expr(has_exact_path(cfg)):
        if ((bits >> reg_idx) & 1) != 0:
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
    """Form the steep channels' KK and A entries of tile row ``row`` in ``corr_ptr``.

    Called by all lanes of the row stage after its K inv / K decay stores.  ``bits`` is the lane's
    :func:`lane_steep_bits`; ``k_norm`` / ``q_scale`` the row's K inverse norm and Q factor (the inverse norm, times
    the attention scale where the A tile carries it); ``raw_q_ptr`` is ``None`` without an A tile.  Every operand is
    read from the raw and operand tiles. ``corr_ptr`` is ``[2, BT, BT]`` fp32, the KK rows then the A rows.
    Each row group reduces in a fixed order and one lane stores each entry, for
    ``s <= row`` and each steep channel,

        A[row, s]  = q_n[row] k_n[s] 2^(g[row] - g[s])
        KK[row, s] = k_n[row] k_n[s] 2^(g[row] - g[s])   (s < row)

    Non-safe plans mark every channel and consume these complete tiles without
    issuing the factorized Tensor Core products. Safe-gate replay marks only
    hazardous channels, so it subtracts those channels' already-issued MMA
    products and retains the historical additive correction. The row-stage
    barrier must make all operand rows visible before this call."""
    with_a = cutlass.const_expr(raw_q_ptr is not None)
    kk_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    attn_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    for source_row in cutlass.range_constexpr(cfg.b_t):
        kk_regs[source_row] = cutlass.Float32(0.0)
        attn_regs[source_row] = cutlass.Float32(0.0)
    for j in cutlass.range(cfg.d_k // 64 * 8, unroll=1):
        if ((bits >> j) & 1) != 0:
            channel = (j // 8) * 64 + lane_in_row_group * 8 + (j % 8)
            x = (raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * k_norm
            if cutlass.const_expr(cfg.safe_gate):
                x_decay = (k_decay_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            if cutlass.const_expr(with_a):
                z = (raw_q_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * q_scale
                if cutlass.const_expr(cfg.safe_gate):
                    z_decay = (q_decay_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            interval = cutlass.Float32(0.0)
            decay = cutlass.Float32(1.0)
            # Grow the bounded interval backward. Safe replay keeps token logs;
            # static non-safe execution keeps their already-exponentiated FP32
            # factors and needs no per-pair transcendental instruction.
            for step in cutlass.range_constexpr(cfg.b_t):
                source_row = cfg.b_t - 1 - step
                if source_row <= row:
                    if source_row < row:
                        if cutlass.const_expr(cfg.safe_gate):
                            interval = interval + steep_token_log(cfg, exchange_ptr, log_last_ptr, source_row + 1, channel)
                        else:
                            decay = decay * direct_token_decay(cfg, exchange_ptr, log_last_ptr, source_row + 1, channel)
                    k_s = (raw_k_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                    if cutlass.const_expr(k_norm_ptr is not None):
                        k_s = k_s * (k_norm_ptr + source_row).load()
                    interval_decay = decay
                    if cutlass.const_expr(cfg.safe_gate):
                        interval_decay = cute.math.exp2(interval, fastmath=True)
                    k_decayed = k_s * interval_decay
                    if source_row < row:
                        value = x * k_decayed
                        if cutlass.const_expr(cfg.safe_gate):
                            k_inv_s = (k_inv_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                            value = value - x_decay * k_inv_s
                        kk_regs[source_row] = kk_regs[source_row] + value
                    if cutlass.const_expr(with_a):
                        value = z * k_decayed
                        if cutlass.const_expr(cfg.safe_gate):
                            k_inv_s = (k_inv_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                            value = value - z_decay * k_inv_s
                        attn_regs[source_row] = attn_regs[source_row] + value
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
    """Merge a direct/correction tile into an m16n16 accumulator fragment."""
    for accum_idx in cutlass.range_constexpr(8):
        row = lane_idx // 4
        if cutlass.const_expr(accum_idx % 4 >= 2):
            row = row + cutlass.Int32(8)
        col = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col = col + cutlass.Int32(1)
        value = (corr_ptr + row * cfg.b_t + col).load()
        if cutlass.const_expr(cfg.safe_gate):
            acc[accum_idx] = acc[accum_idx] + value
        else:
            acc[accum_idx] = value
    # Callers release this stage through an elected lane. Join every reader
    # before that lane can let the producer overwrite the correction tile.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def steep_grad_corrections(cfg: cutlass.Constexpr, channel, t, scale, exchange_ptr, log_last_ptr, raw_q_ptr, raw_k_ptr, q_norm_ptr, k_norm_ptr, dadm_ptr):
    """The backward's intra-tile terms of steep channel ``channel`` at token ``t`` (bprop compute group 2).

    ``dadm_ptr`` holds the chunk's strict-lower dA and dM tiles, ``[2, BT, BT]`` in the io dtype, exactly as the
    tcgen05 GEMMs read them.  With ``F(t, s) = 2^(g[t] - g[s])`` and ``K inv[s] = k_n[s] * 2^(g[BT - 1] - g[s])``
    rounded to the io dtype (the steep column of the K inv operand), forms

        dq_attn  = sum_{s<t} dA[t, s] k_n[s] F(t, s)          dq_mma  = sum_{s<t} dA[t, s] K inv[s]
        dkd_attn = sum_{s<t} dM[t, s] k_n[s] F(t, s)          dkd_mma = sum_{s<t} dM[t, s] K inv[s]
        dk_inv   = sum_{s>t} (scale q_n[s] dA[s, t] - k_n[s] dM[s, t]) F(s, t)

    every exponent of F at most 0. ``*_mma`` are the steep column's products that the dQ attn and dK decay GEMMs
    already accumulated. Return the dQ correction, the inverse-side dK term and the
    decay-side dK correction (also used for dgate). Callers only map these onto their register layouts;
    they add the two dK terms in their original accumulation order."""
    dq_attn = cutlass.Float32(0.0)
    dq_mma = cutlass.Float32(0.0)
    dkd_attn = cutlass.Float32(0.0)
    dkd_mma = cutlass.Float32(0.0)
    dk_inv = cutlass.Float32(0.0)
    tile = cutlass.const_expr(cfg.b_t * cfg.b_t)
    for s in cutlass.range(cfg.b_t, unroll=1):
        if s != t:
            k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
            if cutlass.const_expr(cfg.l2norm):
                k_s = k_s * (k_norm_ptr + s).load()
            if s < t:
                k_inv_s = (k_s * steep_restore_scale(cfg, exchange_ptr, log_last_ptr, s, channel)).to(cfg.io_dtype).to(cutlass.Float32)
                if cutlass.const_expr(cfg.safe_gate):
                    log_decay = steep_interval_log(cfg, exchange_ptr, log_last_ptr, s + 1, t + 1, channel)
                    decay = cute.math.exp2(log_decay, fastmath=True)
                else:
                    decay = direct_interval_scale(cfg, exchange_ptr, log_last_ptr, s + 1, t + 1, channel)
                da = (dadm_ptr + t * cfg.b_t + s).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + t * cfg.b_t + s).load().to(cutlass.Float32)
                dq_attn = dq_attn + da * (k_s * decay)
                dq_mma = dq_mma + da * k_inv_s
                dkd_attn = dkd_attn + dm * (k_s * decay)
                dkd_mma = dkd_mma + dm * k_inv_s
            else:
                q_s = (raw_q_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    q_s = q_s * (q_norm_ptr + s).load()
                if cutlass.const_expr(cfg.safe_gate):
                    log_decay = steep_interval_log(cfg, exchange_ptr, log_last_ptr, t + 1, s + 1, channel)
                    decay = cute.math.exp2(log_decay, fastmath=True)
                else:
                    decay = direct_interval_scale(cfg, exchange_ptr, log_last_ptr, t + 1, s + 1, channel)
                da = (dadm_ptr + s * cfg.b_t + t).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + s * cfg.b_t + t).load().to(cutlass.Float32)
                dk_inv = dk_inv + ((q_s * scale) * da - (k_s) * dm) * decay
    exp_t = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, t, channel)
    dq_delta = scale * (dq_attn - exp_t * dq_mma)
    dkd_delta = exp_t * dkd_mma - dkd_attn
    return dq_delta, dk_inv, dkd_delta


@cute.jit
def pack_dstate_operand(cfg: cutlass.Constexpr, original, decayed, first_channel, words_ptr, steep_any):
    """Pack the carry once; steep channels use the original because K inv carries their decay.

    Both inputs are already in registers. Leave the decayed FP32 accumulator
    unchanged and let the caller publish the packed operand under its barriers.
    """
    operand = cutlass.Array(cutlass.Float32, 32, alignment=16)
    for channel in cutlass.range_constexpr(32):
        operand[channel] = decayed[channel]
    if cutlass.const_expr(has_exact_path(cfg)):
        if steep_any != 0:
            steep_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_0_warp_ids))
            for channel in cutlass.range_constexpr(32):
                if steep_channel_bit(words_ptr, first_channel + channel, steep_rows) != 0:
                    operand[channel] = original[channel]
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for pc in cutlass.range_constexpr(16):
        packed[pc] = fp32_to_fp16(operand[2 * pc], operand[2 * pc + 1], dtype=cfg.io_dtype)
    return packed
