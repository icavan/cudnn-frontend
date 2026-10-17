# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare prepared backward frames with the original cold host ABI."""

from types import SimpleNamespace
import ast
from pathlib import Path

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement = cutedsl_requirement_error("KDA binding tests")
if requirement:
    pytest.skip(requirement, allow_module_level=True)

from cuda.bindings.driver import CUstream
from cudnn.linear_attention.frost.kda_engine import CompiledKdaBwd
from cudnn.linear_attention.frost.kernel.kda_chain_backward_f16 import CHAIN_BACKWARD_BUFFERS, run_chain_backward
from cudnn.linear_attention.frost.kernel.kda_uncut_backward_f16 import UNCUT_BACKWARD_BUFFERS, run_uncut_backward

pytestmark = pytest.mark.L0


def test_raw_qk_loads_complete_before_slot_release():
    """A new raw-slot writer cannot overtake CG2's asynchronous TMEM reads."""
    import cudnn.linear_attention.frost.kernel.kda_bprop_f16 as kernel

    tree = ast.parse(Path(kernel.__file__).read_text())
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute2_warp_group")
    calls = [node for node in ast.walk(body) if isinstance(node, ast.Call)]
    loads = [node.lineno for node in calls if ast.unparse(node.func) == "nvvm.tcgen05_ld"]
    drains = [node.lineno for node in calls if ast.unparse(node.func) == "nvvm.tcgen05_wait" and node.args[0].value == "load"]
    release = next(node.lineno for node in calls if ast.unparse(node.func) == "bars.mb_qk_raw_done[qk_raw_stage].arrive")
    assert max(loads) < max(drains) < release


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("checkpoint", [0, 16, 32])
@pytest.mark.parametrize("gate", ["log", "linear", "safe"])
@pytest.mark.parametrize("optional", [False, True])
def test_warm_backward_frame_matches_cold_with_new_buffers(chain, checkpoint, gate, optional):
    """Optional/coarse slots, folds and repeated views keep their cold semantics."""
    plan = CompiledKdaBwd.__new__(CompiledKdaBwd)
    plan.chain = chain
    plan.has_state_checkpoints = checkpoint > 0
    plan.coarse_checkpoints = checkpoint > 16
    plan.needs_recompute = checkpoint != 16
    plan.has_dstate0 = optional
    plan.fused_h_m = checkpoint == 0
    plan.safe_gate, plan.log_gate = gate == "safe", gate == "log"
    plan.fold_dq, plan.fold_dk, plan.fold_dv = optional, False, optional
    plan.b_t, plan.pieces, plan.n_heads_out, plan.num_seqs = 16, 4, 4, 2
    plan.recompute_span_tokens, plan.checkpoint_cadence, plan.scale = 256, checkpoint, 0.125
    plan.unit_chunks, plan.length_rule, plan.num_sm, plan.device = 1, False, 148, 0
    plan.dim_v, plan.dim_k, plan.gate_lower_bound = 128, 128, -5.0
    plan.use_qk_l2norm, plan.use_beta_sigmoid, plan.allow_neg_eigval = True, True, False
    plan.chain_rows_per_cta = lambda *unused: 16
    plan.stream_type = CUstream
    plan.chain_buffer_names, plan.uncut_buffer_names = CHAIN_BACKWARD_BUFFERS, UNCUT_BACKWARD_BUFFERS
    plan.run_chain_backward, plan.run_uncut_backward = run_chain_backward, run_uncut_backward
    names = ("q", "k", "v", "g", "beta", "cu_seqlens", "dO", "dQ", "dK", "dV", "dG", "dBeta")
    if checkpoint:
        names += ("state_checkpoints",)
    if optional:
        names += ("initial_state", "d_final_state", "d_initial_state")
        if plan.safe_gate:
            names += ("a_log", "dt_bias", "d_a_log", "d_dt_bias")
    plan.carve_names = (
        "work_items",
        "work_count",
        "work_items_summary",
        "work_items_recompute",
        "work_count_recompute",
        "state_checkpoints",
        "scheduler_all",
        "scheduler_recompute",
        "scheduler_bwd",
        "scheduler_m",
        "scheduler_series",
        "scheduler_summary",
        "recompute_tensormaps",
        "cu_pieces",
        "main_rows",
        "summary_rows",
        "main_count",
        "summary_count",
        "fused_tensormaps",
        "recompute_tensormaps_m",
        "recompute_tensormaps_series",
        "summary_tensormaps",
        "bwd_tensormaps",
        "state_h",
        "state_m",
        "state_x",
        "state_g",
        "state_dx_end",
        "dq_ho",
        "dv_ho",
        "gate_part_a",
        "gate_part_dt",
    )
    # Keep only the regions actually carved by the selected scheduling family.
    excluded = set()
    if not plan.needs_recompute:
        excluded |= {"state_checkpoints", "work_items_recompute", "work_count_recompute", "recompute_tensormaps_series"}
    if checkpoint:
        excluded |= {"state_h", "state_x", "fused_tensormaps"}
    if not optional or not plan.safe_gate:
        excluded |= {"gate_part_a", "gate_part_dt"}
    plan.carve_names = tuple(name for name in plan.carve_names if name not in excluded)
    plan.carve = object()
    plan.bind(names)
    calls, finish = [], []
    plan.build_chain_backward = plan.build_uncut_backward = lambda **unused: lambda *args: calls.append(args)
    plan.channel_gate_bwd = lambda *args, **kwargs: finish.append((args, kwargs))
    plan.head_group_reduce = lambda *args, **kwargs: finish.append((args, kwargs))
    plan.chain_launch = plan.uncut_launch = None
    for stream in (None, 0, 17):
        expected_stream = 0 if stream is None else stream
        views = tuple(object() for _ in names)
        scratch = tuple(object() for _ in plan.carve_names)
        workspace = SimpleNamespace(carve=lambda unused: scratch)
        if plan.chain_launch is None and plan.uncut_launch is None:
            plan.run(views, workspace, stream)
        finish.clear()
        plan.run(views, workspace, stream)
        warm, warm_finish = calls[-1], list(finish)
        plan.chain_launch = plan.uncut_launch = None
        finish.clear()
        plan.run(views, workspace, stream)
        cold = calls[-1]
        assert warm[:-1] == cold[:-1]
        assert int(warm[-1]) == int(cold[-1]) == expected_stream
        assert finish == warm_finish
