# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only contracts for prepared KDA launch slots and numerical comparisons."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "schedule, scalars, aliases",
    [
        (
            "uncut",
            ("checkpoint_every_n", "seed_span_chunks", "seed_every_n", "scale"),
            {"scheduler_all_recompute": "scheduler_order_recompute", "scheduler_all_bprop": "scheduler_order_bwd"},
        ),
        (
            "chain",
            ("pieces", "heads_out", "num_seqs", "series_span_chunks", "checkpoint_every_n", "seed_every_n", "scale"),
            {
                "cu_pieces_main": "cu_pieces",
                "state_m_main": "state_m",
                "state_m_chain": "state_m",
                "state_g_chain": "state_g",
                "state_dx_end_chain": "state_dx_end",
            },
        ),
    ],
)
def test_prepared_backward_slots_match_compiled_host_signature(schedule, scalars, aliases):
    """Constexpr slots disappear; repeated tensor views retain their host order."""
    source = ROOT / "python/cudnn/linear_attention/frost/kernel" / f"kda_{schedule}_backward_f16.py"
    tree = ast.parse(source.read_text())
    host = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == f"{schedule}_backward_host")
    binding = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == f"{schedule.upper()}_BACKWARD_BUFFERS" for target in node.targets)
    )
    assert any(ast.unparse(decorator) == "cute.jit" for decorator in host.decorator_list)
    assert not host.args.posonlyargs and not host.args.kwonlyargs and host.args.vararg is None and host.args.kwarg is None
    runtime = []
    for arg in host.args.args:
        annotation = arg.annotation.value if isinstance(arg.annotation, ast.Subscript) else arg.annotation
        if ast.unparse(annotation) != "cutlass.Constexpr":
            runtime.append(aliases.get(arg.arg, arg.arg))
    assert tuple(runtime) == (*scalars, *ast.literal_eval(binding), "stream")


@pytest.mark.parametrize("script", ["benchmark_kda_e2e.py", "benchmark_kda_kernel_ab.py", "audit_kda_precision.py"])
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
