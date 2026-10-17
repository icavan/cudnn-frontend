# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard dO production and both generic/MMA consumers against schedule changes."""

import ast
from pathlib import Path

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement = cutedsl_requirement_error("KDA dO pipeline tests")
if requirement:
    pytest.skip(requirement, allow_module_level=True)

pytestmark = pytest.mark.L0


def functions():
    """Inspect the served kernel, not a copied source fragment."""
    from cudnn.linear_attention.frost.kernel import kda_bprop_f16

    tree = ast.parse(Path(kda_bprop_f16.__file__).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def calls(body, name):
    """Find calls by their complete receiver expression."""
    return [node for node in ast.walk(body) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]


def test_raw_qk_loads_complete_before_slot_release():
    """A new raw-slot writer cannot overtake CG2's asynchronous TMEM reads."""
    body = functions()["compute2_warp_group"]
    loads = calls(body, "nvvm.tcgen05_ld")
    drains = calls(body, "nvvm.tcgen05_wait")
    drains = [node for node in drains if node.args[0].value == "load"]
    release = calls(body, "bars.mb_qk_raw_done[qk_raw_stage].arrive")
    assert loads and drains and len(release) == 1
    assert max(node.lineno for node in loads) < max(node.lineno for node in drains) < release[0].lineno


def test_do_ready_precedes_first_mma_read():
    """The late dQ placement must not move the ready guard past dU/dstate."""
    body = functions()["tcgen05_mma_warp"]
    reads = [node for node in calls(body, "mma_ss") if any(isinstance(arg, ast.Name) and arg.id == "d_do_trans" for arg in node.args)]
    ready = calls(body, "bars.mb_do_ready[raw_stage_idx].wait")
    assert len(reads) == 2 and len(ready) == 1
    assert ready[0].lineno < min(node.lineno for node in reads)


def test_do_ready_precedes_epilogue_reads():
    """Generic ldmatrix readers acquire their own dO completion barrier."""
    body = functions()["epilogue_warp"]
    reads = [node for node in calls(body, "nvvm.ldmatrix") if any(isinstance(arg, ast.Name) and arg.id == "sDo_ptr" for arg in ast.walk(node.args[0]))]
    ready = calls(body, "bars.mb_do_ready[raw_stage].wait")
    assert reads and len(ready) == 1
    assert ready[0].lineno < min(node.lineno for node in reads)
    release = calls(body, "bars.mb_do_done[raw_stage].arrive")
    assert len(release) == 1 and max(node.lineno for node in reads) < release[0].lineno


def test_do_reuse_waits_for_mma_and_epilogue():
    """CG1 acknowledges the committed dstate epoch after its last dO MMA."""
    bodies = functions()
    barrier = next(node.value for node in ast.walk(bodies["make_bars"]) if isinstance(node, ast.keyword) and node.arg == "mb_do_done")
    count = next(node.value for node in barrier.keywords if node.arg == "init_count")
    assert ast.unparse(next(node.value for node in barrier.keywords if node.arg == "producer")) == "Producer.THREAD"
    assert isinstance(count, ast.BinOp) and isinstance(count.op, ast.Add)
    assert {ast.unparse(count.left), ast.unparse(count.right)} == {"WARP", "CG1"}
    cg1 = bodies["compute1_warp_group"]
    ready = calls(cg1, "bars.mb_dstate_acc_ready.wait")
    release = calls(cg1, "bars.mb_do_done[raw_index.idx].arrive")
    assert len(ready) == len(release) == 1 and ready[0].lineno < release[0].lineno
    mma = bodies["tcgen05_mma_warp"]
    reads = [node for node in calls(mma, "mma_ts_step") if isinstance(node.args[0], ast.Name) and node.args[0].id == "bmm_state_do_desc"]
    ss_reads = [node for node in calls(mma, "mma_ss") if any(isinstance(arg, ast.Name) and arg.id == "d_do_trans" for arg in node.args)]
    complete = calls(mma, "bars.mb_dstate_acc_ready.arrive")
    assert len(reads) == len(complete) == 1 and len(ss_reads) == 2
    assert max(node.lineno for node in reads + ss_reads) < complete[0].lineno
