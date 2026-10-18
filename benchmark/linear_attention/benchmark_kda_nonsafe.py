# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event-timed KDA non-safe forward/backward microbenchmark."""

import argparse
import statistics

import torch

from cudnn.linear_attention.ops import kimi_delta_attention


def _elapsed_us(fn, iterations):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--checkpoint", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--dangerous", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1644)
    dtype = torch.bfloat16
    total = args.batch * args.tokens
    q = torch.randn(total, args.heads, args.dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(total, args.heads, args.dim, device="cuda"), dim=-1).to(dtype).requires_grad_()
    v = torch.randn(total, args.heads, args.dim, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.full((total, args.heads, args.dim), -0.01, device="cuda", dtype=torch.float32)
    if args.dangerous:
        g.view(args.batch, args.tokens, args.heads, args.dim)[:, 5::16].fill_(-200.0)
    g.requires_grad_()
    beta = torch.full((total, args.heads), 0.25, device="cuda", dtype=dtype, requires_grad=True)
    cu = torch.arange(args.batch + 1, device="cuda", dtype=torch.int32) * args.tokens

    kwargs = dict(
        scale=args.dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        safe_gate=False,
        checkpoint_every_n_tokens=args.checkpoint,
        batch_invariant=args.batch_invariant,
        plan_name="kda_frost",
    )
    leaves = (q, k, v, g, beta)

    def forward():
        return kimi_delta_attention(q, k, v, g, beta, cu, **kwargs)[0]

    output = forward()
    grad = torch.randn_like(output)

    def backward():
        return torch.autograd.grad(output, leaves, grad, retain_graph=True)

    for _ in range(5):
        forward()
        backward()
    torch.cuda.synchronize()
    forward_us = _elapsed_us(forward, args.iterations)
    backward_us = _elapsed_us(backward, args.iterations)
    grads = backward()
    finite = bool(torch.isfinite(output).all() and all(torch.isfinite(x).all() for x in grads))
    print(
        f"batch={args.batch} tokens={args.tokens} heads={args.heads} dim={args.dim} "
        f"checkpoint={args.checkpoint} dangerous={args.dangerous} batch_invariant={args.batch_invariant} "
        f"fwd_median_us={statistics.median(forward_us):.2f} "
        f"bwd_median_us={statistics.median(backward_us):.2f} finite={finite}",
        flush=True,
    )
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            backward()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)


if __name__ == "__main__":
    main()
