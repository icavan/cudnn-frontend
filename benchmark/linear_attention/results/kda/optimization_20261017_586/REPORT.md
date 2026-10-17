# KDA rolled L2 projection — 586 GPU 1, 2026-10-17

## Scope and provenance

This adds one microtuning to the existing performance branch at
`18b3281017aeb334fb90ac8ee79519196cf4569f`. The PR base is the complete upstream
checkout at `bd4ebdddeab3cb384e15d29871e652aa0a032fcb`; no newer upstream rebase
is part of this update. Historical reports remain historical, not fresh evidence.

Only the D128 normalized backward projection changes: its two token passes use
`cutlass.range(2, unroll=1)` instead of `cutlass.range_constexpr(2)`. Five local
names are separated from the channel-owned branch to avoid a CuTeDSL loop-scope
join. AST canonicalization confirms that arithmetic, indices, rounding, reduction
order, synchronization and every other function are unchanged from the prior PR.
There is no shape-specific dispatch, new knob, signature or compile-key axis.

The integrated main-kernel SHA256 is
`37c2a1d53c800954d94295cb5a3eb6afdf9ba2341ab4eff9e1d5ed6d84f181d8`,
identical to the candidate used for precision, sanitizer and NCU validation.
Python 3.12, Torch 2.11.0+cu130, `nvidia-cutlass-dsl[cu13]==4.7.0`, cuDNN 9.19;
the shared Python extension SHA256 is
`0b9ee82097c29ef4dc9f23337b8215c10d8e5c940d0f96cc7ad20f82a11d4e6b`.
No dependency, GPU clock or power configuration was changed.

## Full-source performance

The final matrix uses canonical imports of the entire baseline or optimized
package, not backward-module substitution under a shared host. Inputs are BF16,
S=8192, D=128, checkpoint=16, Q/K normalization, beta sigmoid and default safe
gate, without initial state or learnable gate parameters. H64 uses chain; the
other measured heads use uncut. E2E means public KDA forward plus autograd
backward, not the whole model or optimizer.

Independent-process ABBA order is baseline A, optimized A, optimized B,
baseline B. Each process measures all five shapes with the unchanged
`benchmark_kda_e2e.py::run_case`: nine rounds, 25 replays, four complete calls
per CUDA graph. Compilation, allocations, verification and profiling are outside
the measured region. Results are the arithmetic mean of two process medians;
latency reduction is `100 * (1 - optimized / baseline)`, not throughput increase.
Both eager and graph timing samples and six output/gradient fingerprints are
retained. The fixed-host ablation is a separate experiment and is not used as
the whole-PR baseline here.

CUDA-graph latency, microseconds (complete backward and forward+backward):

| B x H | Baseline bwd | Optimized bwd | Reduction | Baseline E2E | Optimized E2E | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 x 64 | 2033.18 | 1924.37 | 5.35% | 2518.10 | 2408.67 | 4.35% |
| 1 x 96 | 2331.24 | 1884.21 | 19.18% | 2965.41 | 2516.85 | 15.13% |
| 1 x 128 | 2579.89 | 2291.18 | 11.19% | 3294.10 | 3006.39 | 8.73% |
| 2 x 64 | 2588.71 | 2300.76 | 11.12% | 3296.39 | 3008.91 | 8.72% |
| 2 x 128 | 5166.76 | 4678.83 | 9.44% | 6610.32 | 6119.76 | 7.42% |

Eager latency, microseconds (same sources, inputs and independent processes):

| B x H | Baseline bwd | Optimized bwd | Reduction | Baseline E2E | Optimized E2E | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 x 64 | 2039.52 | 1931.59 | 5.29% | 2529.36 | 2423.38 | 4.19% |
| 1 x 96 | 2341.14 | 1893.07 | 19.14% | 2984.03 | 2529.06 | 15.25% |
| 1 x 128 | 2583.42 | 2295.32 | 11.15% | 3311.97 | 3017.82 | 8.88% |
| 2 x 64 | 2597.73 | 2306.03 | 11.23% | 3308.89 | 3021.88 | 8.67% |
| 2 x 128 | 5168.26 | 4720.70 | 8.66% | 6618.94 | 6145.74 | 7.15% |

All six output/gradient fingerprints agree across all four full-source
processes for every shape. Every eager/graph backward/E2E cell improves over
the pinned baseline; this establishes the measured five cells, not all shapes.
The former 2x128 regression is gone. In an additional six-process target check,
the rolled microdelta alone reduces 2x128 graph bwd by 10.27% and E2E by 7.94%
relative to the previous PR, with identical outputs/gradients.

`evidence.json` records process medians, fingerprints, source hashes,
validation summaries, selected NCU counters and raw-artifact hashes. Raw
round samples, logs, NCU reports/SASS and the external orchestration scripts
are retained under `tune-bh256-20261017` on the test host and in the local
`tune-bh256-586` evidence directory; bulky profiler binaries are not added to
the PR. Host enqueue samples are retained, not used to claim a universal
CPU-overhead reduction.

## Precision and sanitizer evidence

The exact integrated kernel passed 22 paired FP64-reference cases, covering
150 gradient tensors, against the unchanged prior PR: every tensor is
byte-identical and every reported FP64-error delta is zero. Coverage includes
FP16/BF16 safe gate, ordinary BF16 non-safe gate, normal/small-norm/correlated/
cancellation/radial/strong-decay inputs, checkpoint 0/16/32, empty sequences,
tails, chain execution and optional learnable gate parameters. This is no
claim of fixing the separately tracked extreme-gate precision problem.

Backward-only changed-input replay tests passed under both tools:

- memcheck: 8 passed, `ERROR SUMMARY: 0 errors`.
- racecheck: 8 passed, `0 hazards displayed (0 errors, 0 warnings)`.

These use caller-supplied checkpoints, D64/D128, FP16/BF16 and optional state/
gate arguments. **Full E2E racecheck is not clean**: previously documented
upstream-identical forward/recompute hazards remain outside this microtuning.
Existing nanobind shutdown reference-leak diagnostics are retained in raw logs.
No process was signaled or forcibly stopped.

An earlier post-timing pytest run reported 89 passing native/execution,
binding, pipeline, ABI and resource cases. Follow-up review found that its
borrowed linear-attention conftest could redirect submodules to the original
PR despite the printed top-level package path. That run is not evidence of
testing the integrated rolled kernel; see `REVIEW_COMMENTS.md` for the corrected
source-pinned verification. The standalone performance, precision and NCU
harnesses do not load this pytest conftest. Ten source-only ABI/store-ownership
checks also passed locally. Both pinned baseline/previous-PR exports were reverified
against all 1062 production/build Git blobs. Changed-file pre-commit, Python
syntax and whitespace checks passed; no unrelated worktree was modified.

## Generated-code explanation

Fresh NCU 2025.3.0 reports compare the prior PR with this microtuning for the
2x128 main backward kernel. Kernel replay, `--cache-control all`,
`--clock-control none`, `--kill no`; profiler durations are not performance data.

| Metric | Prior PR | Rolled projection |
| --- | ---: | ---: |
| Static instruction span, bytes | 75,264 | 73,344 |
| Static instruction sites | 4,704 | 4,584 |
| Executed warp instructions | 1,514,500,287 | 1,537,842,393 |
| Issue active, % | 35.40 | 38.71 |
| Eligible warps / active cycle | 0.48 | 0.55 |
| No-instruction stall ratio | 2.71 | 1.85 |
| Local spilling requests | 0 | 0 |
| Entry registers / thread | 128 | 128 |
| Shared memory / CTA, bytes | 198,656 | 198,656 |
| BAR.SYNC warp executions | 2,625,848 | 2,625,848 |
| LDTM warp executions | 14,139,392 | 14,139,392 |

The reduced code footprint and improved issue behavior support an instruction
supply benefit. Dynamic instruction count actually increases by 1.54%; neither
less dynamic work nor fewer barriers explains the gain. Stall ratios are not
percentages of elapsed time and do not prove a particular I-cache miss cause.
CG register budgets remain 144/152/160 and other warps 56, with no LDL/STL in
either main-kernel report. Latest reports do not measure HBM byte traffic.
This is a 2x128 uncut profile, not a claim that every shape is spill-free:
the earlier H64 chain profile recorded local spilling. That path's current
performance is assessed by the full-source timing matrix, not this NCU cell.

A rename-only control stays effectively unchanged (5207.92 vs 5206.93 us),
while the rolled loop reaches 4687.69 us in the same fixed-host screen. Register
budget alternatives and the small early-dQ gain were discarded.

## Final source and fence review

Compatibility audit: passed for the complete PR's public ordered-buffer/map
fallback contract, fresh per-call workspace/stream binding, native cold/warm
frames, chain/uncut, checkpoint 0/16/32, grouped heads, optional gate/state
inputs, strided outputs and capture/replay. The microdelta changes none of
these contracts and introduces no host allocation or synchronization.
Docstring audit: passed for changed runtime entry points and binding helpers;
the token-loop implementation has no new public contract.

Fence audit: passed for changed backward edges. Reviewed actual `MBarrier`,
TMA and MMA helpers, the complete main/summary diff and retained generated-code
evidence against the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
Ordinary arrivals are release/CTA and successful waits acquire/CTA; they are
not relaxed counter-only handoffs. Execution/completion and visibility were
checked separately:

| Resource / scope | Completion and visibility | Reuse protection |
| --- | --- | --- |
| Raw Q/K/V/gate SMEM, CTA | TMA transaction completion and acquire wait before generic/MMA readers | CG0/CG1/CG2 acknowledgements; CG2 retains Q/K through both rolled passes |
| dO SMEM, CTA | Independent acquire before epilogue and MMA reads | Epilogue plus CG1 acknowledgement after the last dO-reading committed MMA epoch |
| Gate exchange SMEM, CTA | All input loads precede the CG0 named barrier and in-place stores | Existing raw-slot consumer acknowledgements |
| Decay/inverse/scale SMEM, CTA | Generic stores, async-shared proxy bridge where consumed asynchronously, release/acquire ready handoff | Separate MMA/CG2 and scale-reader done phases; absent seeds still acknowledge |
| Raw Q/K TMEM, CTA | Store completion before publication and acquire before reads | Explicit `tcgen05.wait::ld` before reader-done allows another warp to overwrite |
| Projection SMEM/registers, CTA | 128-thread named barrier after gradient staging; exact four-part rounded reduction | Uniform two-pass loop; each lane has distinct channel/token output addresses |
| Output staging SMEM to GMEM, CTA | Per-writer async-shared proxy fence, release/acquire ready handoff, elected TMA issue | Prior-slot completion wait before staging; warp-wide bulk commit/read-wait and done arrivals |
| Descriptors / workspace, device and stream | Existing descriptor publication and dependent-launch ordering retained | Plan retains positions, not operand storage or streams; fresh call binding |

TMA completion includes its documented async-to-generic visibility bridge;
generic-to-async SMEM stores retain explicit proxy fences. No fence moved into
the rolled loop or away from its corresponding ready/done signal. Same dynamic
BAR.SYNC/LDTM counts support, but do not replace, the source ownership proof.
The two passes cover the same disjoint tokens 0..15 and channels 0..127; each
warp executes both passes uniformly with the original shuffle masks.
