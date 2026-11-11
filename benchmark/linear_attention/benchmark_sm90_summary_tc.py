# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare explicit BF16 Tensor Core and FP32 SM90 summaries, including errors.

Timings include both H/G and direct M, but exclude attention apply, communication
and training. Inputs are CPU-seeded before transfer. No training-loss claim is
made. --fp64 checks the exact BF16 payload against an independent recurrence;
otherwise numerical differences use the native FP32 summary as the reference.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"

import torch
import cudnn

from benchmark_sm90_summary import candidate, make_case, measure


def errors(got, expected):
    a, b = got.double(), expected.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("nonfinite summary/reference")
    norm, diff = b.norm().item(), (a - b).norm().item()
    return dict(rel_l2=diff / norm if norm else (0.0 if diff == 0 else None), max_abs=(a - b).abs().max().item(), reference_norm=norm)


def compose(outputs, backward):
    """FP64 composition isolates summary error; this is not GPU communication."""
    h, m = (value.double() for value in outputs)
    state = torch.zeros_like(h[0])
    for n in range(h.shape[0] - 1, -1, -1) if backward else range(h.shape[0]):
        state = state @ m[n] + h[n]
    return state


def correlated_reference(x, backward):
    """FP64 closed form for repeated K, constant scalar g and beta=0.5.

    A = exp(g)*(I-beta*k*k^T) has two eigenvalues: decay off k and
    decay*(1-beta*||k||^2) along k. Sum value/gradient forcing with those
    exact powers, avoiding a million serial FP32 state updates. This helper
    is only valid for the harness's --correlated-keys construction.
    """
    k = x["k"][0].double()
    norm2 = k.square().sum(-1)
    decay = x["g"][0, :, 0].double().exp()
    beta = x["beta"][0].double()
    parallel = decay * (1 - beta * norm2)
    projector = k[:, :, None] * k[:, None, :] / norm2[:, None, None]
    eye = torch.eye(128, device=k.device, dtype=torch.float64)
    bounds = x["cu_seqlens"].tolist()
    states, transitions = [], []
    for begin, end in zip(bounds, bounds[1:]):
        state = torch.zeros_like(projector)
        for start in range(begin, end, 4096):
            stop = min(start + 4096, end)
            t = torch.arange(start - begin, stop - begin, device=k.device, dtype=torch.float64)[None, :]
            if backward:
                d = decay[:, None].pow(t + 1)
                p = parallel[:, None].pow(t + 1)
                q = x["q"][start:stop].double().transpose(0, 1)
                do = x["dO"][start:stop].double().transpose(0, 1).transpose(1, 2)
                along = (q * k[:, None, :]).sum(-1) / norm2[:, None]
                correction = do @ ((p - d) * along)[:, :, None]
                state += (do @ (q * d[:, :, None]) + correction * k[:, None, :]) * 128**-0.5
            else:
                weight = beta[:, None] * parallel[:, None].pow(end - begin - 1 - t)
                v = x["v"][start:stop].double().transpose(0, 1).transpose(1, 2)
                state += (v @ weight[:, :, None]) * k[:, None, :]
        d, p = decay.pow(end - begin), parallel.pow(end - begin)
        states.append(state)
        transitions.append(d[:, None, None] * eye + (p - d)[:, None, None] * projector)
    return torch.stack(states), torch.stack(transitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gate", type=float, default=-0.001)
    parser.add_argument("--thd", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--closed-form", action="store_true", help="Independent FP64 repeated-key reference, including long spans")
    parser.add_argument("--cp64", action="store_true", help="64 logical spans per case, composed in FP64; no communication simulation")
    parser.add_argument("--correlated-keys", action="store_true", help="Repeat the first key to retain a large transition on its orthogonal complement")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--profile-one", action="store_true")
    args = parser.parse_args()
    if min(args.tokens) <= 0 or min(args.heads, args.repeats, args.groups) <= 0:
        parser.error("tokens, heads, repeats and groups must be positive")
    if args.cp64 and args.thd:
        parser.error("choose --cp64 or --thd")
    if args.closed_form and (not args.correlated_keys or args.fp64):
        parser.error("--closed-form requires --correlated-keys and excludes --fp64")
    if args.fp64:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/python/linear_attention/hopper"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/python"))
        from test_kda_summary_sm90 import reference
    reference_name = "fp64_closed_form" if args.closed_form else ("fp64" if args.fp64 else "fp32")
    print(json.dumps(dict(cudnn=cudnn.__file__, torch=torch.__version__, cuda=torch.version.cuda, reference=reference_name)), flush=True)
    layouts = [(t,) * (64 if args.cp64 else 1) for t in args.tokens]
    if args.thd:
        layouts += [(33,) * 32, (17,) * 20 + (1024, 4096)]
    handle = cudnn.create_handle()
    try:
        for lengths in layouts:
            x = make_case(lengths, args.heads, args.gate)
            if args.correlated_keys:
                x["k"].copy_(x["k"][:1].clone().expand_as(x["k"]))
            for backward in (False, True):
                fp_run, fp_out, _ = candidate(x, backward, handle)
                tc_run, tc_out, workspace = candidate(x, backward, handle, "kda_summary_hopper_tc")
                fp_run()
                tc_run()
                torch.cuda.synchronize()
                expected = fp_out
                if args.fp64:
                    ref_case = dict(x, cu=x["cu_seqlens"], state=torch.zeros_like(fp_out[0]))
                    ref_case.pop("cu_seqlens")
                    expected = reference(ref_case, backward)
                elif args.closed_form:
                    expected = correlated_reference(x, backward)
                row = dict(
                    lengths=lengths, heads=args.heads, gate=args.gate, backward=backward, workspace_bytes=workspace, correlated_keys=args.correlated_keys
                )
                row["tc_error"] = [errors(a, b) for a, b in zip(tc_out, expected)]
                row["fp32_error"] = [errors(a, b) for a, b in zip(fp_out, expected)]
                if args.cp64:
                    whole = dict(x, cu_seqlens=torch.tensor([0, sum(lengths)], dtype=torch.int32, device="cuda"))
                    if args.fp64:
                        whole_case = dict(ref_case, cu=whole["cu_seqlens"], state=ref_case["state"][:1])
                        monolithic = reference(whole_case, backward)[0][0]
                    elif args.closed_form:
                        monolithic = correlated_reference(whole, backward)[0][0]
                    else:
                        whole_run, whole_out, _ = candidate(whole, backward, handle)
                        whole_run()
                        monolithic = whole_out[0][0]
                    row["tc_cp64_error"] = errors(compose(tc_out, backward), monolithic)
                    row["fp32_cp64_error"] = errors(compose(fp_out, backward), monolithic)
                if not args.profile_one:
                    row["tc_us"] = measure(tc_run, args.repeats, args.groups)
                    row["fp32_us"] = measure(fp_run, args.repeats, args.groups)
                    if not (row["tc_us"] > 0 and row["fp32_us"] > 0):
                        raise AssertionError("invalid timing")
                    row["speedup"] = row["fp32_us"] / row["tc_us"]
                print(json.dumps(row), flush=True)
                if args.profile_one:
                    return
    finally:
        cudnn.destroy_handle(handle)


if __name__ == "__main__":
    main()
