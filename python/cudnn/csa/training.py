# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd composition for compressed and direct sparse attention training.

The module in this file composes the existing compressor, indexer, Top-K, and
sparse-attention wrappers.  It intentionally remains an allocating PyTorch
convenience layer above their APIBase contracts.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional

import torch
from cudnn.api_base import TupleDict

_SUPPORTED_RATIOS = (1, 2, 4)
_SUPPORTED_DK_PLANES = (96, 128, 148)


def _check_tensor(tensor, name, *, device=None, dtype=None, ndim=None, shape=None, contiguous=True):
    """Validate host-visible tensor metadata without allocating or synchronizing."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(tensor.stride())}")


class _Compressor(torch.autograd.Function):
    """First-order autograd bridge over the public compressor wrappers."""

    @staticmethod
    def forward(ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp, deterministic):
        from .compressor import csa_compressor_forward_wrapper

        result = csa_compressor_forward_wrapper(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            total_comp=total_comp,
        )["out"]
        ctx.ratio = ratio
        ctx.head_dim = head_dim
        ctx.coff = coff
        ctx.deterministic = deterministic
        if any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(kv, score, ape, cu_seqlens, cu_seqlens_comp)
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None or not ctx.saved_tensors:
            return (None,) * 10
        from .compressor import csa_compressor_backward_wrapper

        kv, score, ape, cu_seqlens, cu_seqlens_comp = ctx.saved_tensors
        result = csa_compressor_backward_wrapper(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            grad_out.contiguous(),
            ratio=ctx.ratio,
            head_dim=ctx.head_dim,
            coff=ctx.coff,
            deterministic=ctx.deterministic,
        )
        needs = ctx.needs_input_grad
        return (
            result["grad_kv"] if needs[0] else None,
            result["grad_score"] if needs[1] else None,
            result["grad_ape"] if needs[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SparseAttentionPipeline(torch.autograd.Function):
    """Differentiable attention output plus an explicitly differentiable indexer loss."""

    @staticmethod
    def forward(
        ctx,
        index_q,
        index_k,
        weights,
        attention_q,
        attention_kv,
        attention_sink,
        seq_lens,
        q_causal_offsets,
        token_offsets,
        ratio,
        block_top_k,
        indexer_sm_scale,
        attention_sm_scale,
        indexer_loss_coeff,
        deterministic,
        indexer_backend,
        deterministic_dk_planes,
    ):
        from cudnn import DSA

        scores = DSA.indexer_forward_wrapper(
            index_q,
            index_k.unsqueeze(2),
            weights,
            ratio=ratio,
            sm_scale=indexer_sm_scale,
            q_causal_offsets=q_causal_offsets,
        )["scores"]
        topk = DSA.indexer_top_k_wrapper(
            scores.reshape(-1, index_k.shape[1]),
            seq_lens,
            top_k=block_top_k,
            return_val=True,
            deterministic=deterministic,
        )
        block_indices = topk["indices"].view(1, index_q.shape[1], block_top_k)
        if ratio == 1:
            token_indices = block_indices.squeeze(0)
        else:
            token_indices = (block_indices.squeeze(0).unsqueeze(-1) * ratio + token_offsets).reshape(index_q.shape[1], block_top_k * ratio)

        effective_sink = attention_sink
        if effective_sink is None:
            effective_sink = torch.full((attention_q.shape[1],), -float("inf"), dtype=torch.float32, device=attention_q.device)
        attention = DSA.sparse_attention(
            attention_q,
            attention_kv,
            token_indices,
            effective_sink,
            softmax_scale=attention_sm_scale,
            deterministic=deterministic,
            trusted_compact_metadata=True,
        )
        target = (
            DSA.sparse_attention_score_recompute(
                attention_q,
                attention_kv,
                attention["lse"],
                token_indices,
                softmax_scale=attention_sm_scale,
            )["target"]
            .view(index_q.shape[1], block_top_k, ratio)
            .sum(-1)
            .unsqueeze(0)
        )
        selected_logits = topk["values"].view_as(block_indices)
        log_predict = torch.log_softmax(selected_logits, dim=-1)
        predict = log_predict.exp()
        loss_terms = torch.where(target > 0, torch.special.xlogy(target, target) - target * log_predict, torch.zeros_like(target))
        indexer_loss = loss_terms.sum() * (indexer_loss_coeff / (index_q.shape[0] * index_q.shape[1]))

        ctx.has_sink = attention_sink is not None
        ctx.ratio = ratio
        ctx.indexer_sm_scale = indexer_sm_scale
        ctx.attention_sm_scale = attention_sm_scale
        ctx.indexer_loss_coeff = indexer_loss_coeff
        ctx.deterministic = deterministic
        ctx.indexer_backend = indexer_backend
        ctx.deterministic_dk_planes = deterministic_dk_planes
        ctx.indexer_backward_consumed = False
        ctx.save_for_backward(
            index_q,
            index_k,
            weights,
            attention_q,
            attention_kv,
            attention["out"],
            attention["lse"],
            effective_sink,
            block_indices,
            token_indices,
            target,
            predict,
        )
        ctx.mark_non_differentiable(attention["max_logits"], attention["lse"], block_indices, token_indices)
        ctx.set_materialize_grads(False)
        return attention["out"], attention["max_logits"], attention["lse"], block_indices, token_indices, indexer_loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, _dmax, _dlse, _dblock_indices, _dtoken_indices, dindexer_loss):
        from cudnn import DSA

        index_q, index_k, weights, attention_q, attention_kv, out, lse, sink, block_indices, token_indices, target, predict = ctx.saved_tensors
        needs = ctx.needs_input_grad

        d_attention_q = d_attention_kv = d_sink = None
        if dout is not None and any(needs[i] for i in (3, 4, 5)):
            from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
                sparse_attention_backward_wrapper,
            )

            attention_grads = sparse_attention_backward_wrapper(
                attention_q,
                attention_kv,
                out,
                dout.contiguous(),
                lse,
                sink,
                token_indices,
                softmax_scale=ctx.attention_sm_scale,
                deterministic=ctx.deterministic,
            )
            d_attention_q = attention_grads["dq"] if needs[3] else None
            d_attention_kv = attention_grads["dkv"] if needs[4] else None
            d_sink = attention_grads["d_sink"] if ctx.has_sink and needs[5] else None

        d_index_q = d_index_k = d_weights = None
        if dindexer_loss is not None and any(needs[i] for i in (0, 1, 2)):
            if ctx.indexer_backward_consumed:
                raise RuntimeError(
                    "indexer_loss backward is single-use because the native indexer backward reuses its saved target as scratch; "
                    "run a new forward instead of retaining and reusing this autograd graph"
                )
            ctx.indexer_backward_consumed = True
            indexer_grads = DSA.indexer_backward_wrapper(
                index_q,
                weights,
                index_k,
                target,
                predict,
                block_indices,
                dindexer_loss.contiguous(),
                sm_scale=ctx.indexer_sm_scale,
                loss_coeff=ctx.indexer_loss_coeff,
                backend=ctx.indexer_backend,
                deterministic=ctx.deterministic,
                deterministic_dk_planes=ctx.deterministic_dk_planes,
            )
            d_index_q = indexer_grads["d_index_q"] if needs[0] else None
            d_index_k = indexer_grads["d_index_k"] if needs[1] else None
            d_weights = indexer_grads["d_weights"] if needs[2] else None

        return (
            d_index_q,
            d_index_k,
            d_weights,
            d_attention_q,
            d_attention_kv,
            d_sink,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CompressedSparseAttention(torch.nn.Module):
    """Composable single-batch sparse-attention training module.

    The module owns the index-key LayerNorm and composes index-key pooling,
    dense indexer scoring, Top-K selection, sparse attention, score recompute,
    and their first-order backward paths. This initial API implements NoPE;
    configurations that require positional encoding are not supported.

    ``compress_ratio=1`` is the direct path: ``index_k`` is already
    ``[1, S_k, 128]``, the compressor is not called, and selected indices are
    used directly without a block-to-token expansion. Ratios 2 and 4 enable
    the compressor and accept its packed ``index_k`` input
    ``[T, coff * 128]`` plus ``compressor_score``, ``ape``, and the two
    cumulative-length tensors. This initial composition supports one sequence
    per call; the lower-level kernels continue to expose their broader packed
    contracts.

    The returned ``indexer_loss`` is the mean target-to-predict KL term scaled
    by ``indexer_loss_coeff``.  Differentiating it invokes indexer backward;
    differentiating ``out`` invokes sparse-attention backward.  Gradients from
    the indexer loss flow through LayerNorm and (when enabled) the compressor.
    This is a first-order autograd API. The indexer-loss backward
    is single-use: the native kernel reuses its saved target as scratch, so
    callers must run a new forward instead of applying indexer-loss backward
    twice to a retained graph. Attention-only backward does not consume that
    target.

    All inputs are contiguous CUDA tensors.  Indexer Q/K/weights and attention
    Q/KV are BF16; sequence lengths and indices are INT32; the optional sink is
    FP32. The current native training composition requires SM100 and 32 or 64
    indexer heads. Deterministic indexer backward and
    ``indexer_backend="sm100_v2"`` require 64 indexer heads. Warm every
    fixed-shape specialization before CUDA Graph capture. The convenience layer
    allocates outputs and wrapper workspaces; captured allocations belong to the
    graph's private pool.

    Args:
        index_head_dim: Native indexer width. Currently fixed at 128.
        compress_ratio: 1 for the direct path, or 2 or 4 for fused
            compression.
        coff: Compressor width multiplier, either 1 or 2. The direct path
            requires 1.
        top_k: Selected token count, at most 2048. It must be a multiple of
            ``128 * compress_ratio`` so the corresponding compressed-block
            Top-K satisfies the native indexer contract.
        k_norm_eps: Positive finite epsilon for the owned key LayerNorm.
        k_norm_placement: ``"before_pool"`` or ``"after_pool"``. The direct
            path has no pooling boundary and always normalizes its input key.
        indexer_sm_scale: Positive indexer scale, defaulting to ``128**-0.5``.
        attention_sm_scale: Positive attention scale, defaulting to
            ``attention_q.shape[-1]**-0.5`` in the native attention wrappers.
        indexer_loss_coeff: Finite coefficient applied to the mean KL loss.
        deterministic: Request deterministic compressor, Top-K, attention
            backward, and indexer backward execution without fallback.
        indexer_backend: ``"default"`` or request-or-fail ``"sm100_v2"``.
        deterministic_dk_planes: Deterministic indexer dK workspace policy,
            one of 96, 128, or 148. Non-deterministic execution uses 96.
        device: Initial LayerNorm and offset-buffer device.
        dtype: LayerNorm parameter dtype. Only BF16 or ``None`` is accepted.
    """

    def __init__(
        self,
        *,
        index_head_dim: int = 128,
        compress_ratio: int = 1,
        coff: int = 1,
        top_k: int = 2048,
        k_norm_eps: float = 1e-6,
        k_norm_placement: str = "after_pool",
        indexer_sm_scale: Optional[float] = None,
        attention_sm_scale: Optional[float] = None,
        indexer_loss_coeff: float = 1.0,
        deterministic: bool = False,
        indexer_backend: str = "default",
        deterministic_dk_planes: int = 96,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if index_head_dim != 128:
            raise ValueError(f"index_head_dim must be 128 for the native indexer, got {index_head_dim}")
        if compress_ratio not in _SUPPORTED_RATIOS:
            raise ValueError(f"compress_ratio must be one of {_SUPPORTED_RATIOS}, got {compress_ratio}")
        if coff not in (1, 2):
            raise ValueError(f"coff must be 1 or 2, got {coff}")
        if compress_ratio == 1 and coff != 1:
            raise ValueError("coff must be 1 when compress_ratio=1 because the compressor is bypassed")
        top_k_granularity = 128 * compress_ratio
        if top_k < top_k_granularity or top_k > 2048 or top_k % top_k_granularity:
            raise ValueError(f"top_k must be a multiple of 128 * compress_ratio ({top_k_granularity}) " f"in [{top_k_granularity}, 2048], got {top_k}")
        if isinstance(k_norm_eps, bool) or not isinstance(k_norm_eps, Real) or not math.isfinite(float(k_norm_eps)) or k_norm_eps <= 0:
            raise ValueError(f"k_norm_eps must be finite and positive, got {k_norm_eps}")
        if k_norm_placement not in ("before_pool", "after_pool"):
            raise ValueError(f"k_norm_placement must be 'before_pool' or 'after_pool', got {k_norm_placement!r}")
        if indexer_sm_scale is not None and (
            isinstance(indexer_sm_scale, bool) or not isinstance(indexer_sm_scale, Real) or not math.isfinite(float(indexer_sm_scale)) or indexer_sm_scale <= 0
        ):
            raise ValueError(f"indexer_sm_scale must be finite and positive or None, got {indexer_sm_scale}")
        if attention_sm_scale is not None and (
            isinstance(attention_sm_scale, bool)
            or not isinstance(attention_sm_scale, Real)
            or not math.isfinite(float(attention_sm_scale))
            or attention_sm_scale <= 0
        ):
            raise ValueError(f"attention_sm_scale must be finite and positive or None, got {attention_sm_scale}")
        if isinstance(indexer_loss_coeff, bool) or not isinstance(indexer_loss_coeff, Real) or not math.isfinite(float(indexer_loss_coeff)):
            raise ValueError(f"indexer_loss_coeff must be finite, got {indexer_loss_coeff}")
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool")
        if indexer_backend not in ("default", "sm100_v2"):
            raise ValueError(f"indexer_backend must be 'default' or 'sm100_v2', got {indexer_backend!r}")
        if deterministic_dk_planes not in _SUPPORTED_DK_PLANES:
            raise ValueError(f"deterministic_dk_planes must be one of {_SUPPORTED_DK_PLANES}, got {deterministic_dk_planes}")
        if not deterministic and deterministic_dk_planes != 96:
            raise ValueError("deterministic_dk_planes is configurable only when deterministic=True")
        if dtype is not None and dtype != torch.bfloat16:
            raise TypeError(f"k_norm parameters must use torch.bfloat16 to match native indexer inputs, got {dtype}")

        self.index_head_dim = index_head_dim
        self.compress_ratio = compress_ratio
        self.coff = coff
        self.top_k = top_k
        self.block_top_k = top_k // compress_ratio
        self.k_norm_placement = k_norm_placement
        self.indexer_sm_scale = index_head_dim**-0.5 if indexer_sm_scale is None else float(indexer_sm_scale)
        self.attention_sm_scale = None if attention_sm_scale is None else float(attention_sm_scale)
        self.indexer_loss_coeff = float(indexer_loss_coeff)
        self.deterministic = deterministic
        self.indexer_backend = indexer_backend
        self.deterministic_dk_planes = deterministic_dk_planes
        self.k_norm = torch.nn.LayerNorm(index_head_dim, eps=float(k_norm_eps), device=device, dtype=torch.bfloat16 if dtype is None else dtype)
        offsets = None if compress_ratio == 1 else torch.arange(compress_ratio, dtype=torch.int32, device=device).view(1, 1, compress_ratio)
        self.register_buffer("_token_offsets", offsets, persistent=False)

    def _validate_inputs(
        self,
        index_q,
        index_k,
        weights,
        attention_q,
        attention_kv,
        seq_lens,
        attention_sink,
        compressor_score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        q_causal_offsets,
    ):
        """Validate the complete public contract before launching any operation."""

        _check_tensor(index_q, "index_q", dtype=torch.bfloat16, ndim=4)
        if not index_q.is_cuda:
            raise ValueError("index_q must be a CUDA tensor")
        device = index_q.device
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise NotImplementedError("integrated sparse-attention training currently requires SM100")
        from cudnn.frost.buffers import cutedsl_requirement_error

        requirement_error = cutedsl_requirement_error("integrated sparse-attention training")
        if requirement_error:
            raise RuntimeError(requirement_error)
        if index_q.shape[0] != 1 or index_q.shape[-1] != self.index_head_dim:
            raise ValueError(f"index_q must have shape [1, S_q, H, {self.index_head_dim}], got {tuple(index_q.shape)}")
        _, sq, indexer_heads, _ = index_q.shape
        if sq <= 0:
            raise ValueError("index_q S_q must be positive")
        if indexer_heads not in (32, 64):
            raise NotImplementedError(f"integrated indexer training requires H=32 or H=64, got {indexer_heads}")
        if (self.deterministic or self.indexer_backend == "sm100_v2") and indexer_heads != 64:
            raise ValueError(f"deterministic indexer backward and indexer_backend='sm100_v2' require indexer H=64, got {indexer_heads}")
        _check_tensor(weights, "weights", device=device, dtype=torch.bfloat16, shape=(1, sq, indexer_heads))
        _check_tensor(attention_q, "attention_q", device=device, dtype=torch.bfloat16, ndim=3)
        _check_tensor(attention_kv, "attention_kv", device=device, dtype=torch.bfloat16, ndim=2)
        if attention_q.shape[0] != sq:
            raise ValueError(f"attention_q S_q must match index_q ({sq}), got shape {tuple(attention_q.shape)}")
        attention_heads = attention_q.shape[1]
        if attention_heads not in (16, 32, 64, 128):
            raise NotImplementedError(f"attention H must be one of (16, 32, 64, 128), got {attention_heads}")
        if attention_q.shape[2] not in (512, 576) or attention_kv.shape[1] != attention_q.shape[2]:
            raise ValueError(f"attention Q/KV dimensions must match at 512 or 576, got {attention_q.shape[2]} and {attention_kv.shape[1]}")
        if attention_heads == 128 and attention_q.shape[2] == 576:
            raise NotImplementedError("native sparse-attention training does not support attention H=128 with D=576")
        _check_tensor(seq_lens, "seq_lens", device=device, dtype=torch.int32, shape=(sq,))
        if attention_sink is not None:
            _check_tensor(attention_sink, "attention_sink", device=device, dtype=torch.float32, shape=(attention_heads,))
        if q_causal_offsets is not None:
            _check_tensor(q_causal_offsets, "q_causal_offsets", device=device, dtype=torch.int32, shape=(1,))

        compressor_args = (compressor_score, ape, cu_seqlens, cu_seqlens_comp)
        if self.compress_ratio == 1:
            _check_tensor(index_k, "index_k", device=device, dtype=torch.bfloat16, ndim=3)
            if index_k.shape[0] != 1 or index_k.shape[-1] != self.index_head_dim:
                raise ValueError(f"direct index_k must have shape [1, S_k, {self.index_head_dim}], got {tuple(index_k.shape)}")
            if any(value is not None for value in compressor_args):
                raise ValueError("compressor_score, ape, cu_seqlens, and cu_seqlens_comp must all be None when compress_ratio=1")
            sk = index_k.shape[1]
        else:
            _check_tensor(index_k, "index_k", device=device, dtype=torch.bfloat16, ndim=2)
            expected_width = self.coff * self.index_head_dim
            if index_k.shape[1] != expected_width or index_k.shape[0] % self.compress_ratio:
                raise ValueError(
                    f"pooled index_k input must have shape [T, {expected_width}] with T divisible by {self.compress_ratio}, got {tuple(index_k.shape)}"
                )
            names = ("compressor_score", "ape", "cu_seqlens", "cu_seqlens_comp")
            if any(value is None for value in compressor_args):
                raise ValueError(f"{', '.join(names)} are all required when compress_ratio > 1")
            _check_tensor(compressor_score, "compressor_score", device=device, dtype=torch.bfloat16, shape=index_k.shape)
            _check_tensor(ape, "ape", device=device, dtype=torch.float32, shape=(self.compress_ratio, expected_width))
            _check_tensor(cu_seqlens, "cu_seqlens", device=device, dtype=torch.int32, shape=(2,))
            _check_tensor(cu_seqlens_comp, "cu_seqlens_comp", device=device, dtype=torch.int32, shape=(2,))
            sk = index_k.shape[0] // self.compress_ratio

        if sk < self.block_top_k:
            raise ValueError(f"logical index K length ({sk}) must be at least block Top-K ({self.block_top_k})")
        max_sq = sk * self.compress_ratio + (self.compress_ratio - 1)
        if sq > max_sq:
            raise ValueError(f"index_q S_q ({sq}) must be <= compressed K length * compress_ratio + (compress_ratio - 1) ({max_sq})")
        if attention_kv.shape[0] < sk * self.compress_ratio:
            raise ValueError(f"attention_kv has {attention_kv.shape[0]} tokens but selected index blocks can address {sk * self.compress_ratio} tokens")

        return sk

    def forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        attention_q: torch.Tensor,
        attention_kv: torch.Tensor,
        seq_lens: torch.Tensor,
        attention_sink: Optional[torch.Tensor] = None,
        *,
        compressor_score: Optional[torch.Tensor] = None,
        ape: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_comp: Optional[torch.Tensor] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
    ) -> TupleDict:
        """Run the fixed configuration and return attention outputs and indexer loss.

        Args:
            index_q: Contiguous BF16 ``[1, S_q, H_i, 128]`` index queries,
                where ``H_i`` is 32 or 64.
            index_k: Direct BF16 ``[1, S_k, 128]`` keys for ratio 1, otherwise
                compressor input ``[T, coff * 128]``.
            weights: Contiguous BF16 ``[1, S_q, H_i]`` indexer head weights.
            attention_q: Contiguous BF16 ``[S_q, H_a, D]`` attention queries.
                ``H_a`` is independently one of 16, 32, 64, or 128; H128 with
                D576 is unsupported.
            attention_kv: Contiguous BF16 ``[S_kv, D]`` attention keys/values,
                where ``D`` is 512 or 576.
            seq_lens: Contiguous INT32 ``[S_q]`` valid compressed-key prefix
                lengths. Each device value is caller-owned, lies in
                ``[top_k / compress_ratio, S_k]``, and agrees with the indexer
                causal mask.
            attention_sink: Optional contiguous FP32 ``[H_a]`` learnable sink.
            compressor_score: BF16 compressor scores shaped like compressed
                ``index_k``; required only when the ratio exceeds one.
            ape: FP32 compressor positional embedding
                ``[compress_ratio, coff * 128]``; required only when compressed.
            cu_seqlens: INT32 ``[2]`` token cumulative lengths for the single
                sequence; required only when compressed.
            cu_seqlens_comp: INT32 ``[2]`` compressed cumulative lengths;
                required only when compressed.
            q_causal_offsets: Optional INT32 ``[1]`` global uncompressed token
                position of local query zero.

        Returns:
            A ``TupleDict`` containing BF16 ``out [S_q,H_a,D]``, FP32
            ``max_logits`` and ``lse [S_q,H_a]``, INT32 local compressed
            ``block_indices [1,S_q,top_k/compress_ratio]``, expanded INT32
            ``token_indices [S_q,top_k]``, and scalar FP32
            ``indexer_loss``. Only ``out`` and ``indexer_loss`` differentiate.

        Runtime tensors must remain alive until work on the current stream has
        completed. The module parameters must already reside on the input CUDA
        device, as for any ordinary ``torch.nn.Module``.
        """

        sk = self._validate_inputs(
            index_q,
            index_k,
            weights,
            attention_q,
            attention_kv,
            seq_lens,
            attention_sink,
            compressor_score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            q_causal_offsets,
        )
        for name, parameter in self.k_norm.named_parameters():
            if parameter.device != index_q.device:
                raise ValueError(f"k_norm {name} must be on {index_q.device}, got {parameter.device}")
            if parameter.dtype != torch.bfloat16:
                raise TypeError(f"k_norm {name} must have dtype torch.bfloat16, got {parameter.dtype}")
        if self._token_offsets is not None and self._token_offsets.device != index_q.device:
            raise ValueError(f"token offset buffer must be on {index_q.device}, got {self._token_offsets.device}")

        if self.compress_ratio == 1:
            transformed_k = self.k_norm(index_k)
        else:
            compressor_k = index_k
            if self.k_norm_placement == "before_pool":
                compressor_k = self.k_norm(index_k.view(index_k.shape[0], self.coff, self.index_head_dim)).view_as(index_k)
            transformed_k = _Compressor.apply(
                compressor_k,
                compressor_score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                self.compress_ratio,
                self.index_head_dim,
                self.coff,
                sk,
                self.deterministic,
            ).view(1, sk, self.index_head_dim)
            if self.k_norm_placement == "after_pool":
                transformed_k = self.k_norm(transformed_k)

        out, max_logits, lse, block_indices, token_indices, indexer_loss = _SparseAttentionPipeline.apply(
            index_q,
            transformed_k,
            weights,
            attention_q,
            attention_kv,
            attention_sink,
            seq_lens,
            q_causal_offsets,
            self._token_offsets,
            self.compress_ratio,
            self.block_top_k,
            self.indexer_sm_scale,
            self.attention_sm_scale,
            self.indexer_loss_coeff,
            self.deterministic,
            self.indexer_backend,
            self.deterministic_dk_planes,
        )
        return TupleDict(
            out=out,
            max_logits=max_logits,
            lse=lse,
            block_indices=block_indices,
            token_indices=token_indices,
            indexer_loss=indexer_loss,
        )


__all__ = ["CompressedSparseAttention"]
