# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare KDA residual revisions using identical CPU-generated inputs.

Run once per source tree with --source-root and a fresh --output. Accuracy is
against an independent CPU FP64 token recurrence; timings are warmed CUDA Graph
replays, excluding compilation, input preparation and reference calculation.
Run baseline/candidate in both orders and include independent baseline controls.
The caller selects an idle GPU with CUDA_VISIBLE_DEVICES; this script does not
probe its model/compute capability or terminate other processes.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch


def inputs(tokens, heads, dim, case, dtype=torch.bfloat16, seed=1731):
    rng = torch.Generator(device="cpu").manual_seed(seed + tokens + dim)
    q = torch.randn(tokens, heads, dim, generator=rng)
    k = torch.randn(tokens, heads, dim, generator=rng)
    q = q / q.square().sum(-1, keepdim=True).sqrt()
    k = k / k.square().sum(-1, keepdim=True).sqrt()
    v = torch.randn(tokens, heads, dim, generator=rng)
    decay, strength = (1e-5, 0.001) if case == "long_memory" else (0.01, 0.5)
    gate = -(torch.rand(tokens, heads, dim, generator=rng) * 0.5 + 0.75) * decay
    beta = (torch.rand(tokens, heads, generator=rng) * 0.5 + 0.75) * strength
    seed = torch.randn(1, heads, dim, dim, generator=rng) * 0.1
    return [q.to(dtype), k.to(dtype), v.to(dtype), gate, beta.to(dtype)], seed


def fp64_reference(cpu_values, seed):
    """Column-state recurrence for H and identity-seeded M together, no chunks."""
    q, k, v, g, beta = [x.double().numpy() for x in cpu_values]
    tokens, heads, dim = q.shape
    state = np.concatenate((seed[0].double().numpy().transpose(0, 2, 1), np.broadcast_to(np.eye(dim), (heads, dim, dim))), axis=-1).copy()
    output = np.empty_like(v)
    for t in range(tokens):
        state *= np.exp(g[t])[:, :, None]
        projection = np.einsum("hk,hkv->hv", k[t], state)
        target = np.concatenate((v[t], np.zeros((heads, dim))), axis=-1)
        state += k[t, :, :, None] * (beta[t, :, None] * (target - projection))[:, None, :]
        output[t] = np.einsum("hk,hkv->hv", q[t], state[:, :, :dim]) / math.sqrt(dim)
    return output, state[:, :, :dim].transpose(0, 2, 1)[None], state[:, :, dim:].transpose(0, 2, 1)[None]


def error(actual, reference):
    a = actual.detach().double().cpu().numpy()
    r = np.asarray(reference, dtype=np.float64)
    return {
        "relative_l2": float(np.linalg.norm((a - r).ravel()) / max(np.linalg.norm(r.ravel()), 1e-30)),
        "max_abs": float(np.max(np.abs(a - r))),
        "reference_max_abs": float(np.max(np.abs(r))),
        "finite": bool(np.isfinite(a).all()),
    }


def graph_time(fn, calls=4, samples=17):
    for _ in range(3):
        warm = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = [fn() for _ in range(calls)]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        timings = []
        for _ in range(samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / calls)
        return {"median_us": statistics.median(timings), "min_us": min(timings), "max_us": max(timings), "samples_us": timings, "calls_per_replay": calls}
    finally:
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    import cudnn

    source = args.source_root.resolve() / "python/cudnn"
    cudnn.__path__.insert(0, str(source))
    from cudnn.linear_attention.frost.common import piece_chain

    choose_pieces = piece_chain.choose_pieces

    def uncut(**kwargs):
        return 0, choose_pieces(**kwargs)[1]

    # batch_invariant alone still cuts sequences longer than 8192 tokens.
    # Isolate the requested LOCAL recurrence: no merge can mask its error/cost.
    piece_chain.choose_pieces = uncut
    piece_chain.is_dv_split = lambda **kwargs: False
    from cudnn.linear_attention.ops.kda import kimi_delta_attention, kimi_delta_attention_summary

    modules = [
        importlib.import_module("cudnn.linear_attention.frost.kernel." + name)
        for name in ("kda_prefill_f16", "kda_summary_f16", "kda_recompute_f16", "kda_bprop_f16", "kda_prep_prefill_f16")
    ]
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"wrong source: {module.__file__}")
    report = {
        "label": args.label,
        "source_root": str(args.source_root),
        "torch": torch.__version__,
        "seed": args.seed,
        "internal_piece_chain": False,
        "dv_split": False,
        "source_sha256": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
        "accuracy": [],
        "performance": [],
    }

    if not args.performance_only:
        for case, tokens, dim in (("standard", 4096, 64), ("long_memory", 32768, 128)):
            cpu, seed_cpu = inputs(tokens, 2, dim, case, seed=args.seed)
            reference_start = time.monotonic()
            oref, href, mref = fp64_reference(cpu, seed_cpu)
            values = [x.cuda() for x in cpu]
            seed = seed_cpu.cuda()
            cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
            with torch.no_grad():
                out, final = kimi_delta_attention(*values, cu, initial_state=seed, output_final_state=True, batch_invariant=True, plan_name="kda_frost")
                h, m = kimi_delta_attention_summary(*values[1:], cu, initial_state=seed, batch_invariant=True, plan_name="kda_summary_frost")
            row = {
                "case": case,
                "tokens": tokens,
                "heads": 2,
                "dim": dim,
                "output": error(out, oref),
                "final_state": error(final, href),
                "summary_h": error(h, href),
                "summary_m": error(m, mref),
                "summary_vs_forward": error(h, final.detach().double().cpu().numpy()),
                "reference_and_compile_seconds_not_throughput": time.monotonic() - reference_start,
            }
            report["accuracy"].append(row)
            print(json.dumps(row), flush=True)

    if not args.accuracy_only:
        for tokens, heads, dim in ((4096, 16, 64), (32768, 16, 128), (262144, 16, 128)):
            cpu, seed_cpu = inputs(tokens, heads, dim, "long_memory", seed=args.seed)
            values = [x.cuda() for x in cpu]
            seed = seed_cpu.cuda()
            cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
            do = torch.ones_like(values[2])
            fwd = lambda: kimi_delta_attention(*values, cu, initial_state=seed, output_final_state=True, batch_invariant=True, plan_name="kda_frost")
            summary = lambda: kimi_delta_attention_summary(*values[1:], cu, initial_state=seed, batch_invariant=True, plan_name="kda_summary_frost")
            bwd = lambda: torch.ops.cudnn.kimi_delta_attention_bwd(
                do, *values, cu, 1 / math.sqrt(dim), initial_state=seed, batch_invariant=True, plan_name="kda_frost"
            )
            with torch.no_grad():
                for operation, fn in (("forward_uncut", fwd), ("summary_hm_uncut", summary), ("backward_with_recompute", bwd)):
                    if tokens == 262144 and operation == "backward_with_recompute":
                        continue
                    timing = graph_time(fn)
                    row = {"operation": operation, "tokens": tokens, "heads": heads, "dim": dim, **timing}
                    report["performance"].append(row)
                    print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
