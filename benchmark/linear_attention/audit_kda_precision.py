# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare safety-fixed and optimized KDA gradients against one FP64 oracle.

Inputs are generated on CPU and rounded before the oracle sees them. The oracle
uses this source tree's normalization contract (additive squared-norm eps=1e-6).
The constant is checked against the served kernels before GPU auditing. Reference work is never
included in performance timing. This script does not change production routing.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


def normalize(x):
    """The source contract, evaluated independently in double precision."""
    return x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()


def load_reference():
    """Load the repository's independent FP64 recurrence without pytest setup."""
    path = Path(__file__).resolve().parents[2] / "test/python/linear_attention/reference_kda.py"
    spec = importlib.util.spec_from_file_location("kda_precision_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kda_reference


def oracle_block(reference, tensors, lengths, upstream, *, safe_gate, l2norm, parameters):
    """Evaluate one independent group of heads using double autograd math."""
    leaves = tuple(t.detach().double().requires_grad_(True) for t in tensors)
    q, k, v, g, beta = leaves[:5]
    a_log, dt_bias = (
        leaves[5:]
        if parameters
        else (torch.zeros(g.shape[1], dtype=torch.float64, device=g.device), torch.zeros(g.shape[1:], dtype=torch.float64, device=g.device))
    )
    if l2norm:
        q, k = normalize(q), normalize(k)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    out, _ = reference(q[None], k[None], v[None], g[None], beta[None], cu_seqlens=cu, safe_gate=safe_gate, a_log=a_log, dt_bias=dt_bias, use_beta_sigmoid=True)
    out = out[0]
    gradients = torch.autograd.grad(out, leaves, upstream.double())
    return out.detach(), tuple(t.detach() for t in gradients)


def oracle(reference, tensors, lengths, upstream, *, safe_gate, l2norm, parameters, reference_heads=1):
    """Bound reference memory by evaluating independent heads in small groups.

    Each group still differentiates the full sequence in FP64. No checkpoint
    quantization or approximation is introduced into the reference.
    """
    outputs, gradients = [], []
    for start in range(0, tensors[0].shape[1], reference_heads):
        stop = start + reference_heads
        group = tuple(t[:, start:stop] if i < 5 else t[start:stop] for i, t in enumerate(tensors))
        out, grads = oracle_block(reference, group, lengths, upstream[:, start:stop], safe_gate=safe_gate, l2norm=l2norm, parameters=parameters)
        outputs.append(out)
        gradients.append(grads)
    return torch.cat(outputs, dim=1), tuple(torch.cat(group, dim=1 if i < 5 else 0) for i, group in enumerate(zip(*gradients)))


def draw_case(lengths, heads, dim, dtype, distribution, seed, *, safe_gate, parameters, device):
    """Create reproducible normal and cancellation-sensitive rounded inputs."""
    generator = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    shape = (total, heads, dim)

    def rand(shape):
        return torch.randn(shape, generator=generator)

    q, k, v = (rand(shape) for _ in range(3))
    if distribution == "small_norm":
        q[::3] *= 1e-6
        k[1::3] *= 1e-6
    elif distribution == "correlated":
        k = rand((1, heads, dim)).expand(shape).clone() + k * 1e-3
    elif distribution == "cancellation":
        v[1::2] = -v[::2][: v[1::2].shape[0]]
        k = rand((1, heads, dim)).expand(shape).clone() + k * 1e-3
    elif distribution == "radial":
        q = rand((1, heads, dim)).expand(shape).clone()
        k = q.clone()
    g = rand(shape) - 4 if safe_gate else -(rand(shape).abs() * 0.02 + 0.01)
    if distribution == "strong_decay":
        g = rand(shape) + 4 if safe_gate else -(rand(shape).abs() + 0.5)
    elif distribution == "radial":
        g.fill_(-50 if safe_gate else 0)
    tensors = [q.to(dtype), k.to(dtype), v.to(dtype), g, rand(shape[:2])]
    if parameters:
        tensors += [rand((heads,)) * 0.2, rand((heads, dim)) * 0.1]
        if distribution == "radial":
            tensors[-1].zero_()
    return tuple(t.to(device) for t in tensors), rand(shape).to(dtype).to(device)


def error(actual, expected):
    """Report absolute/RMS error without masking small reference gradients."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(expected).all(), "FP64 reference must be finite"
    difference = actual.double() - expected.double()
    rms = difference.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    return {
        "rms_error": rms,
        "rms_relative": rms / max(reference_rms, 1e-30),
        "max_abs": difference.abs().max().item(),
        "reference_rms": reference_rms,
        "finite": bool(torch.isfinite(actual).all()),
    }


def paired_error(candidate, baseline, expected):
    """Expose coordinate-level regressions; an aggregate can hide cancellation."""
    candidate_error = (candidate.double() - expected).abs()
    baseline_error = (baseline.double() - expected).abs()
    increase = candidate_error - baseline_error
    return {
        "bitwise_identical": bool(torch.equal(candidate.contiguous().view(torch.uint8), baseline.contiguous().view(torch.uint8))),
        "worse_fraction": (increase > 0).double().mean().item(),
        "better_fraction": (increase < 0).double().mean().item(),
        "max_abs_error_increase": increase.clamp_min(0).max().item(),
        "candidate_baseline_max_abs": (candidate.double() - baseline.double()).abs().max().item(),
    }


def self_test():
    """A one-token closed form checks the oracle's normalization and gradients."""
    reference = load_reference()
    tensors, do = draw_case([1], 2, 8, torch.float64, "normal", 7, safe_gate=True, parameters=True, device="cpu")
    out, gradients = oracle(reference, tensors, [1], do, safe_gate=True, l2norm=True, parameters=True)
    q, k, v, g, beta, a_log, dt_bias = (t.detach().double().clone().requires_grad_(True) for t in tensors)
    # A zero incoming state makes the first-token output independent of decay.
    analytic = ((normalize(q) * normalize(k)).sum(-1) * beta.sigmoid())[..., None] * v / math.sqrt(8)
    wanted = torch.autograd.grad(analytic, (q, k, v, beta), do)
    torch.testing.assert_close(out, analytic.detach(), rtol=1e-13, atol=1e-13)
    for actual, expected in zip((gradients[0], gradients[1], gradients[2], gradients[4]), wanted):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    assert all(torch.count_nonzero(gradients[i]) == 0 for i in (3, 5, 6))
    for safe_gate in (False, True):
        for l2norm in (False, True):
            tensors, do = draw_case([0, 3, 0, 5], 3, 8, torch.bfloat16, "small_norm", 11, safe_gate=safe_gate, parameters=safe_gate, device="cpu")
            options = dict(safe_gate=safe_gate, l2norm=l2norm, parameters=safe_gate)
            whole = oracle_block(reference, tensors, [0, 3, 0, 5], do, **options)
            grouped = oracle(reference, tensors, [0, 3, 0, 5], do, reference_heads=2, **options)
            for actual, expected in zip((grouped[0], *grouped[1]), (whole[0], *whole[1]), strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    expected = torch.tensor([1.0, 1.0], dtype=torch.float64)
    metrics = paired_error(torch.tensor([1.0, 1.5]), torch.tensor([1.25, 1.25]), expected)
    assert metrics["worse_fraction"] == metrics["better_fraction"] == 0.5
    assert metrics["max_abs_error_increase"] == metrics["candidate_baseline_max_abs"] == 0.25
    signed_zero = paired_error(torch.tensor([-0.0]), torch.tensor([0.0]), torch.tensor([0.0], dtype=torch.float64))
    assert signed_zero["candidate_baseline_max_abs"] == 0 and not signed_zero["bitwise_identical"]
    print("FP64 oracle closed-form, independent-head and paired-error self-tests passed", flush=True)


def run(args):
    import cudnn
    import cudnn.linear_attention.ops.kda as ops
    import cudnn.linear_attention.frost.kernel as package
    from cudnn.linear_attention.frost.kernel import kda_bprop_f16, kda_bprop_summary_f16, kda_chain_backward_f16, kda_uncut_backward_f16
    from benchmark_kda_kernel_ab import load_baseline

    from cudnn.linear_attention.frost.kernel import kda_prefill_f16, kda_prep_f16, kda_recompute_f16, kda_summary_f16

    for module in (kda_bprop_f16, kda_bprop_summary_f16, kda_prefill_f16, kda_prep_f16, kda_recompute_f16, kda_summary_f16):
        assert module.L2_NORM_EPS == 1e-6, f"oracle contract needs updating for {module.__name__}"

    baseline, baseline_summary = load_baseline(args.baseline)
    reference = load_reference()
    stems = ("kda_bprop_config", "kda_bprop_f16", "kda_bprop_summary_f16")
    snapshot = {
        label: {stem: hashlib.sha256((directory / f"{stem}.py").read_bytes()).hexdigest() for stem in stems}
        for label, directory in (("baseline", args.baseline), ("candidate", Path(kda_bprop_f16.__file__).parent))
    }
    report = {
        "source": cudnn.__file__,
        "torch": torch.__version__,
        "baseline": str(args.baseline),
        "args": vars(args) | {"baseline": str(args.baseline), "json": str(args.json)},
        "kernel_sha256": snapshot,
        "cases": [],
    }
    previous_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(previous_stream)
    torch.cuda.set_stream(stream)
    try:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for distribution in args.distributions:
                tensors, do = draw_case(
                    args.lengths, args.heads, args.dim, dtype, distribution, args.seed, safe_gate=args.safe_gate, parameters=args.parameters, device="cuda"
                )
                q, k, v, g, beta = tensors[:5]
                gate_parameters = dict(a_log=tensors[5], dt_bias=tensors[6]) if args.parameters else {}
                opts = dict(
                    use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    safe_gate=args.safe_gate,
                    batch_invariant=args.batch_invariant,
                    plan_name="kda_frost",
                    **gate_parameters,
                )
                cu = torch.tensor([0, *torch.tensor(args.lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
                forward = ops.kimi_delta_attention(q, k, v, g, beta, cu, checkpoint_every_n_tokens=args.checkpoint, **opts)
                checkpoints = forward[2] if args.checkpoint else None
                out_ref, grads_ref = oracle(
                    reference,
                    tensors,
                    args.lengths,
                    do,
                    safe_gate=args.safe_gate,
                    l2norm=True,
                    parameters=args.parameters,
                    reference_heads=args.reference_heads,
                )
                case = {
                    "dtype": dtype_name,
                    "distribution": distribution,
                    "lengths": args.lengths,
                    "heads": args.heads,
                    "dim": args.dim,
                    "checkpoint": args.checkpoint,
                    "safe_gate": args.safe_gate,
                    "parameters": args.parameters,
                    "forward": error(forward[0], out_ref),
                    "gradients": {},
                    "schedules": {},
                }
                names = ("dq", "dk", "dv", "dg", "dbeta") + (("da_log", "ddt_bias") if args.parameters else ())
                results = {}
                for label, module, summary in (("baseline", baseline, baseline_summary), ("candidate", kda_bprop_f16, kda_bprop_summary_f16)):
                    ops.bprop_cache.clear()
                    kda_uncut_backward_f16.uncut_backward_cache.clear()
                    kda_chain_backward_f16.chain_backward_cache.clear()
                    package.kda_bprop_f16 = kda_uncut_backward_f16.kda_bprop_f16 = kda_chain_backward_f16.kda_bprop_f16 = module
                    package.kda_bprop_summary_f16 = kda_chain_backward_f16.kda_bprop_summary_f16 = summary
                    actual = ops.kda_bwd(
                        do, q, k, v, g, beta, cu, args.dim**-0.5, state_checkpoints=checkpoints, checkpoint_every_n_tokens=args.checkpoint, **opts
                    )
                    graph, _ = next(reversed(ops.bprop_cache.values()))
                    plan = graph._compiled_plans[graph._plan_index].compiled
                    case["schedules"][label] = {"chain": plan.chain, "pieces": plan.pieces}
                    selected = actual[:5] + (actual[6:] if args.parameters else ())
                    results[label] = tuple(t.detach().clone() for t in selected)
                    case["gradients"][label] = {name: error(got, want) for name, got, want in zip(names, selected, grads_ref, strict=True)}
                case["relative_error_change"] = {
                    name: (case["gradients"]["candidate"][name]["rms_error"] - case["gradients"]["baseline"][name]["rms_error"])
                    / max(case["gradients"]["baseline"][name]["rms_error"], 1e-30)
                    for name in names
                }
                case["paired_error"] = {
                    name: paired_error(candidate, baseline, expected)
                    for name, candidate, baseline, expected in zip(names, results["candidate"], results["baseline"], grads_ref, strict=True)
                }
                case["finite"] = case["forward"]["finite"] and all(metric["finite"] for metrics in case["gradients"].values() for metric in metrics.values())
                report["cases"].append(case)
                args.json.write_text(json.dumps(report, indent=2))
                print(json.dumps(case), flush=True)
                assert case["finite"], "non-finite result; failing case saved to JSON"
                if args.require_identical:
                    assert all(metric["bitwise_identical"] for metric in case["paired_error"].values()), "gradient changed; failing case saved to JSON"
    finally:
        package.kda_bprop_f16 = kda_uncut_backward_f16.kda_bprop_f16 = kda_chain_backward_f16.kda_bprop_f16 = kda_bprop_f16
        package.kda_bprop_summary_f16 = kda_chain_backward_f16.kda_bprop_summary_f16 = kda_bprop_summary_f16
        torch.cuda.set_stream(previous_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[0, 17, 31, 256])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--reference-heads", type=int, default=1, help="independent heads per FP64 autograd group; bounds peak reference memory")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--checkpoint", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["bfloat16", "float16"])
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["normal", "small_norm", "correlated", "cancellation", "radial", "strong_decay"],
        choices=["normal", "small_norm", "correlated", "cancellation", "radial", "strong_decay"],
    )
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true", help="use the length-based policy (uncut for one sequence up to 8192 tokens)")
    parser.add_argument("--parameters", action="store_true")
    parser.add_argument("--require-identical", action="store_true", help="require all candidate gradients to match the baseline bitwise")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.baseline or not args.json or (args.parameters and not args.safe_gate):
            parser.error("--baseline and --json are required; --parameters requires --safe-gate")
        if min(args.lengths) < 0 or sum(args.lengths) == 0 or args.heads < 1 or args.reference_heads < 1:
            parser.error("lengths must be nonnegative with a positive total; head counts must be positive")
        run(args)
