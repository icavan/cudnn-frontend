# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level synchronization invariants for the SM100 KDA backward epilogue."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0


def _epilogue_tree():
    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "linear_attention" / "frost" / "kernel" / "kda_bprop_f16.py"
    tree = ast.parse(source.read_text())
    epilogue = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "epilogue_warp")
    parents = {child: parent for parent in ast.walk(epilogue) for child in ast.iter_child_nodes(parent)}
    return epilogue, parents


def _calls(epilogue, name):
    return [node for node in ast.walk(epilogue) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def _is_inside_elect_one(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "elect_one":
            return True
    return False


def _contains_name(node, name):
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def test_kda_backward_tma_stores_are_elected_but_group_completion_is_warp_wide():
    epilogue, parents = _epilogue_tree()

    stores = _calls(epilogue, "tma_store_tile")
    commits = _calls(epilogue, "tma_store_commit")
    waits = _calls(epilogue, "tma_store_wait")
    done_arrivals = [
        node
        for node in ast.walk(epilogue)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "arrive" and "_tmastg_done" in ast.unparse(node.func.value)
    ]

    assert len(stores) == 4
    assert all(_is_inside_elect_one(call, parents) for call in stores)
    assert len(commits) == 4
    assert not any(_is_inside_elect_one(call, parents) for call in commits)
    assert len(waits) == 4
    assert not any(_is_inside_elect_one(call, parents) for call in waits)
    assert len(done_arrivals) == 4
    assert not any(_is_inside_elect_one(call, parents) for call in done_arrivals)


@pytest.mark.parametrize("kernel", ["kda_bprop_f16.py", "kda_bprop_summary_f16.py"])
def test_kda_backward_decay_scale_stage_waits_for_all_readers(kernel):
    source = (Path(__file__).resolve().parents[4] / "python" / "cudnn" / "linear_attention" / "frost" / "kernel" / kernel).read_text()

    assert "mb_decay_scale_done: MBarrier" in source
    assert "mb_decay_scale_done=MBarrier(alloc(cfg.smem_decay_stages)" in source
    assert "init_count=CG1, producer=Producer.THREAD" in source
    assert "bars.mb_decay_scale_done[decay_stage].wait(operand_done_phase)" in source
    assert "bars.mb_decay_scale_done[seed_stage].arrive()" in source
    assert "bars.mb_decay_scale_done[next_stage].arrive()" in source
    assert "bars.mb_decay_scale_done[stage].init()" in source


@pytest.mark.parametrize("kernel", ["kda_bprop_f16.py", "kda_bprop_summary_f16.py"])
def test_kda_backward_in_place_gate_exchange_has_load_store_barrier(kernel):
    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "linear_attention" / "frost" / "kernel" / kernel
    tree = ast.parse(source.read_text())
    compute0 = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute0_warp_group")

    gate_loads = [
        node
        for node in ast.walk(compute0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load" and _contains_name(node.func.value, "sGate_ptr")
    ]
    prefix_stores = [
        node
        for node in ast.walk(compute0)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and _contains_name(node.func.value, "sGate_exchange_ptr")
    ]
    cg0_barriers = [
        node
        for node in ast.walk(compute0)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "barrier_cta_sync"
        and node.args
        and ast.unparse(node.args[0]) == "cfg.cg0_sync_barrier_id"
    ]

    assert gate_loads and prefix_stores
    assert any(max(load.lineno for load in gate_loads) < barrier.lineno < min(store.lineno for store in prefix_stores) for barrier in cg0_barriers)
