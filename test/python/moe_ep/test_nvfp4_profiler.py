# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only distributed failure propagation for the benchmark profiler window."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.L0


def profiler_worker():
    from datetime import timedelta

    path = Path(__file__).resolve().parents[3] / "benchmark/bench_nvfp4_moe_ep.py"
    spec = importlib.util.spec_from_file_location("nvfp4_bench", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    try:
        for failing_phase in ("start", "stop", None):
            calls = []
            error = None
            with (
                patch.object(torch.cuda.profiler, "start", side_effect=RuntimeError("injected start failure") if failing_phase == "start" else None) as start,
                patch.object(torch.cuda.profiler, "stop", side_effect=RuntimeError("injected stop failure") if failing_phase == "stop" else None) as stop,
                patch.object(torch.cuda, "synchronize"),
            ):
                try:
                    benchmark.profile_replay(lambda: calls.append("run"), rank)
                except RuntimeError as exc:
                    error = str(exc)
                assert start.call_count == int(rank == 0)
                assert stop.call_count == int(rank == 0 and failing_phase != "start")
            # This collective must be reached by both ranks after either failure.
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            assert all(value == errors[0] for value in errors)
            if failing_phase:
                assert f"profiler {failing_phase} failed" in error
            else:
                assert error is None
            assert calls == ([] if failing_phase == "start" else ["run"])
    finally:
        dist.destroy_process_group()


def test_profiler_failure_reaches_every_rank():
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(Path(__file__).resolve()), "--worker"],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("expected --worker")
    profiler_worker()
