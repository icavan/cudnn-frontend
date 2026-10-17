# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only contracts for complete KDA numerical comparisons."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("script", ["benchmark_kda_e2e.py"])
def test_benchmark_output_comparisons_reject_missing_slots(script):
    """Do not silently truncate validation if a candidate drops a result slot."""
    tree = ast.parse((ROOT / "benchmark/linear_attention" / script).read_text())
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip"]
    assert comparisons
    for call in comparisons:
        assert any(keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in call.keywords), (
            script,
            call.lineno,
        )
