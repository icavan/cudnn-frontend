# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment gates and explicit graph ownership for NVFP4 tests."""

from contextlib import contextmanager

import pytest
import torch


@pytest.fixture
def nvfp4_gpu():
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state

    installed, _ = cutedsl_state()
    if not installed:
        pytest.skip("NVFP4 GPU tests require nvidia-cutlass-dsl")
    error = cutedsl_requirement_error("NVFP4 GPU tests")
    if error:
        pytest.skip(error)
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 GPU tests require CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("NVFP4 GPU tests require Blackwell SM100/SM103")


@contextmanager
def owned_graph():
    """Destroy the graph before its captured operators or process group."""
    graph = torch.cuda.CUDAGraph()
    try:
        yield graph
    finally:
        graph.reset()
