# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by DSA Python wrappers."""

from functools import lru_cache
from typing import Optional

import torch

from cudnn._torch_stream import _raw_current_stream, _reject_per_thread_stream, record_streams, stream_context
import cuda.bindings.driver as cuda


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def device_capability(device=None) -> tuple[int, int]:
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
    return _device_capability(index)


def device_major(device=None) -> int:
    """Return the CUDA compute-capability major for ``device`` or the current device."""
    return device_capability(device)[0]


def maybe_contiguous(
    x: torch.Tensor | None,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if x is None or x.stride(-1) == 1:
        return x
    record_streams((x,), stream, x.device)  # R1 staging: the copy reads x asynchronously on `stream`
    with torch_stream_context(stream, x.device):
        return x.contiguous()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def validate_q_causal_offsets(
    q_causal_offsets: torch.Tensor | None,
    batch: int,
    device: torch.device,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != torch.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}")
    if not q_causal_offsets.is_cuda:
        raise ValueError("q_causal_offsets must be a CUDA tensor")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.is_contiguous():
        return q_causal_offsets
    with torch_stream_context(stream, device):
        return q_causal_offsets.contiguous()


def resolve_stream(
    current_stream: Optional[cuda.CUstream] = None,
    device=None,
) -> cuda.CUstream:
    """Return an explicit driver stream without constructing a torch Stream.

    ``device`` identifies the operand device when it may differ from torch's
    current device.  The raw-current fast path avoids constructing a torch
    Stream wrapper on every frontend launch.  An implicit PTDS sentinel is
    rejected because it is not a process-wide allocator identity.
    """
    if current_stream is not None:
        return current_stream
    raw = _raw_current_stream(torch, device)
    if raw is not None:
        _reject_per_thread_stream(raw)
        return cuda.CUstream(raw)
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def torch_stream_context(current_stream: Optional[cuda.CUstream] = None, device=None):
    """Run torch work on ``current_stream`` (R1). ``device`` is the operand's
    device; it defaults to the current device, which is only right when the
    operands live there -- pass it whenever a tensor is at hand."""
    return stream_context(current_stream, device)
