# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .h32_pair_topk import H32PairTopkUnion

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


_h32_pair_topk_compile_cache = {}


def _build_h32_pair_topk(
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    seqlen_kv: int,
    current_stream,
):
    """Return the encoded union and length for adjacent H32 query pairs."""
    num_pairs = topk_idxs.shape[0] // 2
    max_topk = topk_idxs.shape[1]
    max_union = min(seqlen_kv, 2 * max_topk)
    union_idxs = torch.empty((num_pairs, max_union), dtype=torch.int32, device=topk_idxs.device)
    union_length = torch.empty((num_pairs,), dtype=torch.int32, device=topk_idxs.device)

    has_length = topk_length is not None
    key = (seqlen_kv, max_topk, has_length)
    if key not in _h32_pair_topk_compile_cache:
        op = H32PairTopkUnion(seqlen_kv=seqlen_kv, max_topk=max_topk)
        _h32_pair_topk_compile_cache[key] = cute.compile(
            op,
            to_cute_tensor(topk_idxs),
            to_cute_tensor(topk_length) if has_length else None,
            to_cute_tensor(union_idxs),
            to_cute_tensor(union_length),
            current_stream,
            options=compile_options(),
        )
    _h32_pair_topk_compile_cache[key](
        topk_idxs,
        topk_length,
        union_idxs,
        union_length,
        current_stream,
    )
    return union_idxs, union_length


def _select_sm100_backend(num_heads: int, head_dim: int) -> Tuple[str, int]:
    """Return the tuned SM100 kernel variant and its sparse-row tile size."""
    if num_heads == 16 and head_dim == 576:
        return "h16_m128", 128
    if num_heads == 32 and head_dim == 576:
        return "h32_m128_m64", 64
    return "generic_m64", 64


def _select_sm100_num_load_kv_warps(backend: str, head_dim: int, num_heads: int) -> int:
    """Use eight gather warps for the tuned H96 specializations."""
    if backend == "generic_m64" and head_dim == 576 and num_heads in (64, 96, 192):
        return 8
    return 16


def flash_attn_bwd_sm100(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    current_stream=None,
    pair_queries: bool = False,
    _split_h96: bool = True,
    _allow_strided_heads: bool = False,
    _workspace_lse_odo: Optional[torch.Tensor] = None,
    _workspace_dkv: Optional[torch.Tensor] = None,
    _accumulate_dkv: bool = False,
    _d_sink: Optional[torch.Tensor] = None,
    _logical_num_head: Optional[int] = None,
    _head_offset: int = 0,
    _workspace_num_heads: Optional[int] = None,
    _preprocess_num_heads: Optional[int] = None,
    _skip_preprocess: bool = False,
    _skip_convert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention (DSA) Backward Pass for Blackwell (SM100), with K=V.

    Accepts flat (unbatched) tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel.

    Args:
        q: (total_S_q, nheads, headdim) float16 or bfloat16
        kv: (total_S_kv, headdim) float16 or bfloat16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v) float16 or bfloat16
        dout: (total_S_q, nheads, headdim_v) float16 or bfloat16
        lse: (total_S_q, nheads) float32, FlashMLA KV-only LSE excluding sink
        attn_sink: (nheads,) float32
        topk_idxs: (total_S_q, topk_max) int32, global indices
        softmax_scale: float (default: 1/sqrt(headdim))
        topk_length: (total_S_q,) int32, per-query valid count, optional
        dq: pre-allocated (total_S_q, nheads, headdim), optional
        dkv: pre-allocated (total_S_kv, headdim), optional
        pair_queries: opt in to adjacent-query union for the high-overlap
            SM100 H96/D576, topk=2048, <=16K-KV specialization

    Returns:
        (dq, dkv, d_sink) -- flat layout gradients
    """
    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    # Mirror the check_support gate: the SM100 kernel is tiled only for
    # head_dim in {512, 576}; any other value indexes shared memory out of
    # bounds and crashes inside the kernel.
    assert head_dim in (512, 576), f"head_dim must be 512 or 576, got {head_dim}"
    head_dim_v = 512 if head_dim == 576 else head_dim
    device = q.device

    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.dtype == kv.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    assert attn_sink.dtype == torch.float32
    assert topk_idxs.dtype == torch.int32
    tensors_to_check = [q, kv, out, dout, lse, attn_sink, topk_idxs]
    if topk_length is not None:
        tensors_to_check.append(topk_length)
    assert all(t.is_cuda and t.device == device for t in tensors_to_check), f"all inputs must be CUDA tensors on {device}"

    # Cross-tensor shape validation: every tensor below is indexed with
    # coordinates derived from q, so a mismatched shape silently reads or
    # writes out of place instead of failing.
    assert kv.ndim == 2 and kv.shape[1] == head_dim, f"kv shape mismatch: expected (total_S_kv, {head_dim}), got {tuple(kv.shape)}"
    expected_o_shape = (total_S_q, num_head, head_dim_v)
    assert out.shape == expected_o_shape, f"out shape mismatch: expected {expected_o_shape}, got {tuple(out.shape)}"
    assert dout.shape == expected_o_shape, f"dout shape mismatch: expected {expected_o_shape}, got {tuple(dout.shape)}"
    assert lse.shape == (total_S_q, num_head), f"lse shape mismatch: expected {(total_S_q, num_head)}, got {tuple(lse.shape)}"
    assert attn_sink.shape == (num_head,), f"attn_sink shape mismatch: expected {(num_head,)}, got {tuple(attn_sink.shape)}"
    assert topk_idxs.ndim == 2 and topk_idxs.shape[0] == total_S_q, f"topk_idxs shape mismatch: expected ({total_S_q}, topk_max), got {tuple(topk_idxs.shape)}"
    if topk_length is not None:
        assert topk_length.dtype == torch.int32, f"topk_length dtype mismatch: expected torch.int32, got {topk_length.dtype}"
        assert topk_length.shape == (total_S_q,), f"topk_length shape mismatch: expected {(total_S_q,)}, got {tuple(topk_length.shape)}"
    if pair_queries:
        assert (
            num_head == 96 and head_dim == 576 and total_S_q % 2 == 0 and topk_idxs.shape[1] == 2048 and total_S_kv <= 16384
        ), "pair_queries requires H96/D576, even total_S_q, topk_max=2048, and total_S_kv<=16384"

    storage_num_head = num_head
    if _logical_num_head is not None:
        assert 0 <= _head_offset < storage_num_head
        assert _head_offset + _logical_num_head <= storage_num_head
        num_head = _logical_num_head

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = 1

    current_stream = resolve_stream(current_stream)
    # The pair path pays for a union build and computes a full H64 tile. It is
    # profitable for the dense sparse-prefill target (topk=2048 in at most 4K
    # KV), where random pair overlap is already 50%. Keep lower-density and
    # smaller-topk shapes on the tuned one-query H32 kernel.
    paired_h32 = (
        _logical_num_head is None and num_head == 32 and head_dim == 576 and total_S_q % 2 == 0 and topk_idxs.shape[1] == 2048 and total_S_kv <= 4096
    ) or (pair_queries and num_head == 96 and head_dim == 576 and total_S_q % 2 == 0 and topk_idxs.shape[1] == 2048 and total_S_kv <= 16384)
    original_q_shape = q.shape
    pair_split_heads = num_head if paired_h32 else None

    # Normalize inputs and allocate outputs/workspaces on the execution stream:
    # the kernel below launches on `current_stream`, so the semantically
    # required zero-initialization of dkv/d_sink and both workspaces (and any
    # contiguity copies) must be stream-ordered with it, not with the ambient
    # torch stream the caller happens to be on.
    with torch_stream_context(current_stream):
        # The internal H96 decomposition uses head slices with the original
        # H96 row stride.  Their CuTe tensors preserve those explicit strides;
        # public calls still normalize arbitrary inputs to contiguous storage.
        if not _allow_strided_heads:
            q, kv, out, dout = [t.contiguous() for t in (q, kv, out, dout)]
            lse = lse.contiguous()
        else:
            kv = kv.contiguous()
        attn_sink = attn_sink.contiguous()
        topk_idxs = topk_idxs.contiguous()
        if topk_length is not None:
            topk_length = topk_length.contiguous()

        # Allocate output tensors
        if dq is None:
            dq = torch.empty_like(q)
        else:
            assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
            assert dq.dtype == q.dtype, f"dq dtype mismatch: expected {q.dtype}, got {dq.dtype}"
            assert dq.device == device, f"dq device mismatch: expected {device}, got {dq.device}"
            # The compile cache is keyed without output strides, so a caller
            # provided output must match the contiguous layout the kernel was
            # compiled for (it is not copied: that would break out-parameter
            # identity).
            assert dq.is_contiguous() or _allow_strided_heads, "dq must be contiguous"
        if dkv is None:
            dkv = torch.empty(total_S_kv, head_dim, dtype=kv.dtype, device=device)
        else:
            expected_dkv_shape = (total_S_kv, head_dim)
            assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
            assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
            assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
            assert dkv.is_contiguous(), "dkv must be contiguous"
        d_sink = torch.zeros_like(attn_sink) if _d_sink is None else _d_sink

        # H96 contains one full H64 head tile plus one H32 tail.  Launch the
        # tuned kernels on strided head views while sharing one FP32 dKV
        # workspace, so the tail no longer pays for a padded second H64 CTA.
        if _split_h96 and not pair_queries and num_head == 96 and head_dim == 576 and topk_idxs.shape[1] == 2048:
            ws_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
                total_S_kv,
                head_dim,
                batch_size,
                cutlass.Float32,
            )
            shared_workspace_dkv = torch.zeros(*ws_dkv_shape, dtype=torch.uint8, device=device)
            ws_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
                total_S_q,
                head_dim,
                num_head,
                batch_size,
                cutlass.Float32,
            )
            shared_workspace_lse_odo = torch.empty(*ws_lse_odo_shape, dtype=torch.uint8, device=device)
            flash_attn_bwd_sm100(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
                dq=dq,
                dkv=dkv,
                current_stream=current_stream,
                _split_h96=False,
                _workspace_lse_odo=shared_workspace_lse_odo,
                _workspace_dkv=shared_workspace_dkv,
                _accumulate_dkv=True,
                _d_sink=d_sink,
                _logical_num_head=64,
                _workspace_num_heads=96,
                _preprocess_num_heads=96,
                _skip_convert=True,
            )
            flash_attn_bwd_sm100(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
                dq=dq,
                dkv=dkv,
                current_stream=current_stream,
                _split_h96=False,
                _workspace_lse_odo=shared_workspace_lse_odo,
                _workspace_dkv=shared_workspace_dkv,
                _accumulate_dkv=True,
                _d_sink=d_sink,
                _logical_num_head=32,
                _head_offset=64,
                _workspace_num_heads=96,
                _skip_preprocess=True,
            )
            return dq, dkv, d_sink

        # Pair adjacent H32 query tokens into one virtual H64 query. The
        # union kernel encodes per-token membership in the index high bits;
        # the H64 main kernel masks the absent half before P/dS are consumed.
        # dQ keeps the original storage and is merely viewed as H64, while
        # dKV is accumulated once for a common KV row inside the CTA.
        dq_return = dq
        if paired_h32:
            topk_idxs, topk_length = _build_h32_pair_topk(
                topk_idxs,
                topk_length,
                total_S_kv,
                current_stream,
            )
            total_S_q //= 2
            num_head *= 2
            q = q.view(total_S_q, num_head, head_dim)
            out = out.view(total_S_q, num_head, head_dim_v)
            dout = dout.view(total_S_q, num_head, head_dim_v)
            lse = lse.view(total_S_q, num_head)
            dq = dq.view(total_S_q, num_head, head_dim)
            attn_sink = torch.cat((attn_sink, attn_sink))
            d_sink = torch.zeros_like(attn_sink)

        # Allocate workspace tensors
        acc_dtype = cutlass.Float32
        ws_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
            total_S_q,
            head_dim,
            num_head,
            batch_size,
            acc_dtype,
        )
        workspace_LSE_OdO = torch.empty(*ws_lse_odo_shape, dtype=torch.uint8, device=device) if _workspace_lse_odo is None else _workspace_lse_odo

        ws_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
            total_S_kv,
            head_dim,
            batch_size,
            acc_dtype,
        )
        workspace_dKV = torch.zeros(*ws_dkv_shape, dtype=torch.uint8, device=device) if _workspace_dkv is None else _workspace_dkv

    backend, block_tile = _select_sm100_backend(num_head, head_dim)
    num_load_kv_warps = _select_sm100_num_load_kv_warps(backend, head_dim, num_head)
    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))

    dtype = torch2cute_dtype_map[q.dtype]

    has_topk_length = topk_length is not None
    max_topk = topk_idxs.shape[1]
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        num_head,
        block_tile,
        max_topk,
        has_topk_length,
        paired_h32,
        num_load_kv_warps,
        tuple(q.stride()),
        tuple(out.stride()),
        tuple(lse.stride()),
        tuple(dq.stride()),
        _head_offset,
        _workspace_num_heads,
        _preprocess_num_heads,
        _skip_preprocess,
        _skip_convert,
    )

    if compile_key not in flash_attn_bwd_sm100.compile_cache:
        q_tensor = to_cute_tensor(q, divisibility=None if _allow_strided_heads else head_dim)
        kv_tensor = to_cute_tensor(kv, divisibility=head_dim)
        out_tensor = to_cute_tensor(out, divisibility=None if _allow_strided_heads else head_dim_v)
        dout_tensor = to_cute_tensor(dout, divisibility=None if _allow_strided_heads else head_dim_v)
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        topk_idxs_tensor = to_cute_tensor(topk_idxs)
        topk_length_tensor = to_cute_tensor(topk_length) if has_topk_length else None
        dq_tensor = to_cute_tensor(dq, divisibility=None if _allow_strided_heads else head_dim)
        dkv_tensor = to_cute_tensor(dkv, divisibility=head_dim)
        d_sink_tensor = to_cute_tensor(d_sink)
        workspace_LSE_OdO_tensor = to_cute_tensor(workspace_LSE_OdO)
        workspace_dKV_tensor = to_cute_tensor(workspace_dKV)

        if backend == "h16_m128":
            from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16

            kernel_obj = FlashAttentionDSABackwardSm100H16(
                element_dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                block_tile=block_tile,
                max_topk=max_topk,
            )
        elif backend == "h32_m128_m64":
            from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32

            kernel_obj = FlashAttentionDSABackwardSm100H32(
                element_dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                block_tile=block_tile,
                max_topk=max_topk,
                head_offset=_head_offset,
                workspace_num_heads=_workspace_num_heads,
                skip_preprocess=_skip_preprocess,
            )
        else:
            # Keep this constructor and class byte-for-byte on the tuned H64
            # path; embedding H16 conditionals in the same CuTe DSL class
            # measurably perturbs H64 code generation.
            kernel_obj = FlashAttentionDSABackwardSm100(
                element_dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                block_tile=block_tile,
                max_topk=max_topk,
                num_load_kv_warps=num_load_kv_warps,
                pair_mask_encoded=paired_h32,
                pair_split_heads=pair_split_heads,
                workspace_num_heads=_workspace_num_heads,
                workspace_head_offset=_head_offset,
                preprocess_num_heads=_preprocess_num_heads,
                skip_convert=_skip_convert,
            )

        with torch.cuda.nvtx.range("flash_attn_bwd_sm100_compile"):
            flash_attn_bwd_sm100.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                problem_shape,
                q_tensor,
                kv_tensor,
                out_tensor,
                dout_tensor,
                lse_tensor,
                attn_sink_tensor,
                topk_idxs_tensor,
                topk_length_tensor,
                dq_tensor,
                dkv_tensor,
                d_sink_tensor,
                workspace_LSE_OdO_tensor,
                workspace_dKV_tensor,
                softmax_scale,
                current_stream,
                options=compile_options(),
            )

    with torch.cuda.nvtx.range(f"flash_attn_bwd_sm100_kernel[{backend}]"):
        flash_attn_bwd_sm100.compile_cache[compile_key](
            problem_shape,
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            topk_length,
            dq,
            dkv,
            d_sink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            current_stream,
        )

    if paired_h32:
        # The virtual head halves correspond to even and odd original query
        # tokens. Fold their independently accumulated sink gradients back to
        # the original per-query head count.
        d_sink = d_sink[:pair_split_heads] + d_sink[pair_split_heads:]
        return dq_return.view(original_q_shape), dkv, d_sink
    return dq, dkv, d_sink


flash_attn_bwd_sm100.compile_cache = {}
