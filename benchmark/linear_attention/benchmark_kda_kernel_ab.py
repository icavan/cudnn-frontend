# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-input alternating KDA backward control after the host migration.

The baseline directory must contain the exact exported bprop source and config
snapshot. Both variants use the candidate host path, isolating GPU changes.
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import cudnn.linear_attention.ops.kda as ops
from cudnn.linear_attention.frost.kernel import kda_bprop_config, kda_bprop_f16, kda_bprop_summary_f16, kda_chain_backward_f16, kda_uncut_backward_f16
from benchmark_kda_e2e import measure


def load_baseline(directory):
    """Load the exact exported source under a distinct module identity."""
    package = "cudnn.linear_attention.frost.kernel"
    modules = []
    current_cfg = kda_bprop_config.CFG
    try:
        for stem in ("kda_bprop_config", "kda_bprop_f16", "kda_bprop_summary_f16"):
            spec = importlib.util.spec_from_file_location(f"{package}.{stem}_ab_baseline", directory / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
            if stem == "kda_bprop_config":
                # Derived config dataclasses stamp CFG defaults at import time.
                kda_bprop_config.CFG = module.CFG
        return modules[1:]
    finally:
        kda_bprop_config.CFG = current_cfg


def run(args):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    generator = torch.Generator().manual_seed(1234)
    shape = (args.tokens, args.heads, args.dim)
    dtype = getattr(torch, args.dtype)
    q, k, v, do = [torch.randn(shape, dtype=dtype, generator=generator).cuda() for _ in range(4)]
    g = (torch.randn(shape, generator=generator) - 4).cuda()
    beta = torch.randn(shape[:2], generator=generator).cuda()
    inputs = (q, k, v, g, beta)
    if args.e2e:
        for tensor in inputs:
            tensor.requires_grad_(True)
    cu = torch.tensor([0, args.tokens], dtype=torch.int32, device="cuda")
    opts = dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True, plan_name="kda_frost")
    out = ops.kimi_delta_attention(q, k, v, g, beta, cu, checkpoint_every_n_tokens=args.checkpoint, **opts)
    checkpoints = out[2] if args.checkpoint else None
    baseline, baseline_summary = load_baseline(args.baseline)
    import cudnn.linear_attention.frost.kernel as package

    graphs, results, samples = {}, {}, {"baseline": [], "candidate": []}
    try:
        for label, module, summary in (("baseline", baseline, baseline_summary), ("candidate", kda_bprop_f16, kda_bprop_summary_f16)):
            ops.bprop_cache.clear()
            kda_uncut_backward_f16.uncut_backward_cache.clear()
            kda_chain_backward_f16.chain_backward_cache.clear()
            kda_uncut_backward_f16.kda_bprop_f16 = module
            kda_chain_backward_f16.kda_bprop_f16 = module
            kda_chain_backward_f16.kda_bprop_summary_f16 = summary
            # build_kda resolves the module from the package at plan construction.
            package.kda_bprop_f16 = module
            package.kda_bprop_summary_f16 = summary

            def call():
                if args.e2e:
                    output = ops.kimi_delta_attention(*inputs, cu, checkpoint_every_n_tokens=args.checkpoint, **opts)[0]
                    return (output,) + torch.autograd.grad(output, inputs, do)
                return ops.kda_bwd(do, q, k, v, g, beta, cu, args.dim**-0.5, state_checkpoints=checkpoints, checkpoint_every_n_tokens=args.checkpoint, **opts)

            for _ in range(5):
                call()
            graph = torch.cuda.CUDAGraph()
            graphs[label] = graph
            with torch.cuda.graph(graph, stream=stream):
                result = call()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            results[label] = tuple(t.detach().clone() for t in result)
        metrics = []
        for actual, expected in zip(results["candidate"], results["baseline"], strict=True):
            if not expected.numel():
                continue
            rms = expected.double().square().mean().sqrt().item()
            difference = actual.double() - expected.double()
            relative = difference.square().mean().sqrt().item() / max(rms, 1e-30)
            metrics.append(
                {
                    "max_abs": difference.abs().max().item(),
                    "rms_relative": relative,
                    "bitwise_identical": bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))),
                }
            )
            assert torch.isfinite(actual).all() and relative < 0.01, metrics
        for round_index in range(args.rounds):
            order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            for label in order:
                samples[label].append(measure(graphs[label].replay, args.repeats, 1)["event_us"])
        medians = {label: statistics.median(values) for label, values in samples.items()}
        report = vars(args) | {
            "baseline": str(args.baseline),
            "json": str(args.json),
            "samples_us": samples,
            "median_us": medians,
            "speedup": medians["baseline"] / medians["candidate"],
            "latency_reduction": 1 - medians["candidate"] / medians["baseline"],
            "accuracy": metrics,
        }
        print(json.dumps(report), flush=True)
        args.json.write_text(json.dumps(report, indent=2))
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        package.kda_bprop_summary_f16 = kda_chain_backward_f16.kda_bprop_summary_f16 = kda_bprop_summary_f16
        package.kda_bprop_f16 = kda_uncut_backward_f16.kda_bprop_f16 = kda_chain_backward_f16.kda_bprop_f16 = kda_bprop_f16


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--checkpoint", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--e2e", action="store_true", help="capture forward plus autograd backward instead of direct backward")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json", type=Path, required=True)
    run(parser.parse_args())
