# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch training composition over the native DSA APIBase wrappers.

This allocating semantic adapter deliberately lives above the kernel execute
contract. Metadata normalization and the SM100 H16/H32-to-H64 / K128 launch
padding are explicit parts of this API; SM90 serves the native H/K extents.
No external forward provider is loaded.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable, Optional

import torch
from cudnn.api_base import TupleDict


def _validate_contract(q, kv, indices, attn_sink, topk_length, softmax_scale):
    """Validate the eager training contract and return scale plus SM90 routing."""
    for name, tensor in (("q", q), ("kv", kv), ("indices", indices), ("attn_sink", attn_sink), ("topk_length", topk_length)):
        if tensor is None and name in ("attn_sink", "topk_length"):
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on Q's device {q.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
        raise ValueError("expected Q [S_q,H,D], KV [S_kv,D], indices [S_q,K]")
    sq, heads, dim = q.shape
    if sq <= 0 or kv.shape[0] <= 0 or indices.shape[1] <= 0:
        raise ValueError("sequence extents and physical Top-K must be positive")
    if kv.shape[1] != dim or indices.shape[0] != sq:
        raise ValueError("KV dimension and index row count must match Q")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or indices.dtype != torch.int32:
        raise TypeError("Q/KV must be BF16 and indices must be INT32")
    if attn_sink is not None and (attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32):
        raise ValueError("attn_sink must be FP32 [H]")
    if topk_length is not None and (topk_length.shape != (sq,) or topk_length.dtype != torch.int32):
        raise ValueError("topk_length must be INT32 [S_q]")
    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, Real)):
        raise TypeError("softmax_scale must be a host real scalar or None")
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else None
    if capability == (9, 0):
        if heads not in (16, 32, 64, 96, 128) or dim not in (512, 576):
            raise NotImplementedError("SM90 native training supports H16/H32/H64/H96/H128 with D512/D576")
    elif capability in ((10, 0), (10, 3)):
        if heads not in (16, 32, 64, 128) or dim not in (512, 576) or (heads, dim) == (128, 576):
            raise NotImplementedError("SM100 native training supports H16/H32/H64 D512/D576 and H128 D512")
    else:
        raise NotImplementedError("native sparse attention training currently requires SM90, SM100, or SM103")
    if capability == (9, 0):
        from .sparse_attention_forward._support_sm90 import sm90_dsa_cutedsl_requirement_error

        error = sm90_dsa_cutedsl_requirement_error()
        if error:
            raise RuntimeError(error)
    else:
        from cudnn.frost.buffers import cutedsl_requirement_error

        error = cutedsl_requirement_error("native DSA training")
        if error:
            raise RuntimeError(error)
    return scale, capability == (9, 0)


def _validate_deterministic_options(deterministic: bool, deterministic_relay_cluster_size: Optional[int]) -> None:
    """Reject invalid deterministic options before metadata staging or launch."""
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if deterministic_relay_cluster_size is not None and (
        type(deterministic_relay_cluster_size) is not int or deterministic_relay_cluster_size not in (4, 8, 16, 32, 64)
    ):
        raise ValueError("deterministic_relay_cluster_size must be one of {4, 8, 16, 32, 64} or None")
    if deterministic_relay_cluster_size is not None and not deterministic:
        raise ValueError("deterministic_relay_cluster_size requires deterministic=True")


def _normalize_sparse_index_slots(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask unsafe slots and bound lengths without changing slot positions."""

    topk = indices.shape[1]
    valid = (indices >= 0) & (indices < s_kv)
    bounded_length = None
    if topk_length is not None:
        bounded_length = topk_length.clamp(min=0, max=topk)
        positions = torch.arange(topk, dtype=torch.int32, device=indices.device).unsqueeze(0)
        valid = valid & (positions < bounded_length.unsqueeze(1))
    return torch.where(valid, indices, -1), bounded_length


def _normalize_cudnn_sparse_metadata(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
    *,
    trusted_compact_metadata: bool = False,
    _compactify: Optional[Callable[[torch.Tensor], Any]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalize metadata for the shared native forward/backward contract.

    The semantic API treats negative or ``>= S_kv`` indices as invalid. Current
    cuDNN DSA backward only guards negative sentinels without lengths, and it
    assumes every entry in a compact active prefix is valid when lengths are
    present.  Normalize high sentinels to ``-1`` in both cases.  For the
    length form, also mask the inactive suffix, compact valid entries to the
    front, and derive a new length.  This is asynchronous GPU work: there is
    no device-to-host validation or synchronization in the adapter.

    Out-of-contract length values are clamped to the physical ``[0, K]``
    envelope before masking.  An asynchronous device assert would poison the
    CUDA context on failure, while a strict host check would synchronize the
    hot path; clamping gives the downstream kernels a memory-safe contract.

    ``trusted_compact_metadata`` is the explicit zero-work normalization
    alternative for a producer that already satisfies the narrower cuDNN
    backward contract.  Later launch preparation can still pad an unaligned
    Top-K or make metadata contiguous where the downstream kernel requires it.
    """

    if trusted_compact_metadata:
        # This is an explicit caller contract, not a property inferred from
        # device data.  Inspecting the values here would either synchronize or
        # launch the same metadata kernels this fast path exists to avoid.
        return indices, topk_length

    normalized, _ = _normalize_sparse_index_slots(indices, topk_length, s_kv)
    if topk_length is None:
        return normalized, None

    if _compactify is None:
        # Lazy import preserves the optional CuTe DSL boundary at module import.
        from .indexer_top_k.api import compactify_wrapper

        _compactify = compactify_wrapper
    compact = _compactify(normalized)
    return compact["indices"], compact["topk_length"]


def _pad_indices(indices):
    topk = indices.shape[1]
    padded = (topk + 127) // 128 * 128
    if padded == topk:
        return indices
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1)


def _native_forward(q, kv, indices, sink, lengths, scale, native_sm90):
    """Launch the native forward, retaining legacy SM100 head padding only."""
    from .sparse_attention_forward.api import sparse_attention_forward_wrapper

    heads = q.shape[1]
    launch_q, launch_sink = q, sink
    if not native_sm90 and heads < 64:
        launch_q = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
        launch_sink = torch.nn.functional.pad(sink, (0, 64 - heads), value=-float("inf"))
    result = sparse_attention_forward_wrapper(
        launch_q,
        kv,
        indices,
        attn_sink=launch_sink,
        topk_length=lengths,
        softmax_scale=scale,
    )
    if not native_sm90 and heads < 64:
        return tuple(result[key][:, :heads].contiguous() for key in ("out", "max_logits", "lse"))
    return result["out"], result["max_logits"], result["lse"]


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, sink, lengths, scale, trusted, native_sm90, deterministic, deterministic_relay_cluster_size):
        """Normalize metadata, launch forward, and save lower-backward inputs."""
        with torch.cuda.device(q.device):
            safe_indices, safe_lengths = _normalize_cudnn_sparse_metadata(
                indices,
                lengths,
                kv.shape[0],
                trusted_compact_metadata=trusted,
            )
            launch_indices = safe_indices if native_sm90 else _pad_indices(safe_indices)
            effective_sink = sink if sink is not None else torch.full((q.shape[1],), -float("inf"), dtype=torch.float32, device=q.device)
            out, max_logits, lse = _native_forward(q, kv, launch_indices, effective_sink, safe_lengths, scale, native_sm90)
        ctx.scale = scale
        ctx.has_lengths = safe_lengths is not None
        ctx.deterministic = deterministic
        ctx.deterministic_relay_cluster_size = deterministic_relay_cluster_size
        if any(ctx.needs_input_grad[i] for i in (0, 1, 3)):
            tensors = [q, kv, out, lse, effective_sink, launch_indices]
            if safe_lengths is not None:
                tensors.append(safe_lengths)
            ctx.save_for_backward(*tensors)
        ctx.mark_non_differentiable(max_logits, lse)
        ctx.set_materialize_grads(False)
        return out, max_logits, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, _dmax, _dlse):
        """Run the selected native backward and return requested gradients."""
        if dout is None:
            return (None,) * 10
        q, kv, out, lse, sink, indices, *rest = ctx.saved_tensors
        from .sparse_attention_backward.api import sparse_attention_backward_wrapper

        with torch.cuda.device(q.device):
            result = sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout.contiguous(),
                lse,
                sink,
                indices,
                softmax_scale=ctx.scale,
                topk_length=rest[0] if ctx.has_lengths else None,
                deterministic=ctx.deterministic,
                deterministic_relay_cluster_size=ctx.deterministic_relay_cluster_size,
            )
        needs = ctx.needs_input_grad
        return (
            result["dq"] if needs[0] else None,
            result["dkv"] if needs[1] else None,
            None,
            result["d_sink"] if needs[3] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    trusted_compact_metadata: bool = False,
    deterministic: bool = False,
    deterministic_relay_cluster_size: Optional[int] = None,
) -> TupleDict:
    """Native BF16 MQA sparse attention with first-order PyTorch autograd.

    Returns ``out``, ``max_logits`` and KV-only natural-log ``lse``. Only
    ``out`` is differentiable, with respect to Q, KV and optional sink.
    Inputs are contiguous on one SM90/SM100 device; launch uses the current
    PyTorch stream. SM90 serves native H/K extents. SM100 H16/H32 forward is
    padded to H64 and its physical Top-K is padded to 128, then sliced back.

    Safe mode masks invalid/OOB indices and inactive suffixes; supplied lengths
    are clamped to [0,K] and valid entries compacted before both passes.
    ``trusted_compact_metadata=True`` skips that work only when the producer
    guarantees bounded valid active prefixes, negative inactive suffixes, and
    lengths in [0,K]. Without lengths, all nonnegative indices must be in range.
    ``deterministic=True`` selects the deterministic backward. On SM90,
    ``deterministic_relay_cluster_size`` may be 4, 8, 16, 32, or 64 and defaults
    to 16; it changes the backward workspace/performance tradeoff. This
    allocating composition does not change the lower-level APIBase execute
    contract.
    """
    if not isinstance(trusted_compact_metadata, bool):
        raise TypeError("trusted_compact_metadata must be a bool")
    _validate_deterministic_options(deterministic, deterministic_relay_cluster_size)
    scale, native_sm90 = _validate_contract(q, kv, indices, attn_sink, topk_length, softmax_scale)
    out, max_logits, lse = _SparseAttention.apply(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        scale,
        trusted_compact_metadata,
        native_sm90,
        deterministic,
        deterministic_relay_cluster_size,
    )
    return TupleDict(out=out, max_logits=max_logits, lse=lse)


@torch.no_grad()
def sparse_attention_score_recompute(
    q: torch.Tensor,
    kv: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Return normalized attention targets aligned to original index slots.

    LSE is the KV-only natural-log statistic from native sparse attention.
    Invalid/inactive slots return zero. No gradients are attached to targets.
    """
    scale, native_sm90 = _validate_contract(q, kv, indices, None, topk_length, softmax_scale)
    if not isinstance(lse, torch.Tensor) or lse.shape != q.shape[:2] or lse.dtype != torch.float32 or lse.device != q.device or not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 [S_q,H] on Q's device")
    from .score_recompute.api import sparse_attn_score_recompute_wrapper

    with torch.cuda.device(q.device):
        safe_indices, _ = _normalize_sparse_index_slots(indices, topk_length, kv.shape[0])
        launch_indices = safe_indices if native_sm90 else _pad_indices(safe_indices)
        result = sparse_attn_score_recompute_wrapper(
            q.unsqueeze(0),
            kv.unsqueeze(0),
            lse.unsqueeze(0),
            launch_indices.unsqueeze(0),
            scale,
            qhead_per_kv_head=q.shape[1],
            topk_length=None,
            topk_indices_global=False,
        )
        target = result["target"].squeeze(0)
        if not native_sm90 and target.shape[1] != indices.shape[1]:
            target = target[:, : indices.shape[1]].contiguous()
    return TupleDict(target=target, indices=safe_indices)


__all__ = ["sparse_attention", "sparse_attention_score_recompute"]
