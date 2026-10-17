# KDA performance rebase against upstream — 2026-10-16

## Scope and provenance

This supersedes the performance conclusions in the October 12 and October 15
reports. The performance branch is rebased onto upstream
`bd4ebdddeab3cb384e15d29871e652aa0a032fcb` (develop, #1679).
The old pushed performance tip was `721c44ca64bc8202325056bec4a8a0145f9ea020`.
The steep-decay/NaN precision branch is independent and is **not** included here.

The baseline production sources are an unmodified `git archive` of that upstream
commit. Candidate and baseline imports were checked in every process; the old
editable-install finder was bypassed inside these processes only. Both use the
same extension built from this upstream revision. Per-file source hashes,
output/gradient fingerprints, round samples and FP64 error metrics are recorded
in `evidence.json`.

Environment: `aistudio-61550139-ssctl`, Python 3.10, Torch 2.11.0+cu130,
cuDNN 9.19, `nvidia-cutlass-dsl[cu13]==4.7.0`, CUDA runtime 13 selected explicitly.
The existing environment's SM100-family record was reused without an extra
hardware/model probe. GPU 1 was reserved for the independent-process ABBA
matrix; other validation ran on separate idle GPUs. No process was killed,
including by wrapper cleanup or pytest timeouts.

Remote evidence is retained under
`/root/kda-opt-20261012/rebase-perf-20261016/results`.

## Retained and removed changes

- Keep the backward launch binding recipe: cache immutable indices/scalars,
  rebind inputs, outputs, workspace and stream on every execution. Use the
  public ordered-buffer graph API for both backend and Python plans.
- Keep the backward MMA rescheduling and normalized Q/K gradient ownership
  change. Retain SMEM values in registers through each row projection instead
  of repeating TMEM/shared-memory load/reduction passes.
- Keep CG1/CG2 register redistribution (152/160 instead of upstream's 168/144),
  with unchanged aggregate CTA register budget.
- Keep the associated correctness fixes: elected TMA store issue, in-place
  gate load/store synchronization, independent decay-reader acknowledgements,
  dO readiness/reuse protection and TMEM load completion before slot release.
- Remove the historical three-to-two raw-stage specialization and its private
  override. Upstream already defaults to two; this cannot be counted as a new
  gain. All schedules retain upstream's raw-ring depth of two.
- Reject the packed diagonal-gradient fusion and reassociated L2 formula.
  They were fast but changed FP32 rounding, and small-norm inputs also exposed
  half/bfloat16 output differences. The retained implementation preserves the
  original rounded products and four-warp reduction order. No precision
  tolerance was enlarged.

The branch changes no public signature, gate semantics, checkpoint dtype,
approximation, history truncation, or dispatch eligibility.

## Performance method and results

The measured region is raw Q/K/V/gate/beta through O and autograd gradients.
Projection, model loss, optimizer and communication are excluded. Compilation,
input construction, validation and profiling are outside timing. All timing
cases use BF16, B=1, normalized Q/K, beta sigmoid, default safe gate, and no
initial state or trainable gate parameters. Those optional inputs are checked
separately below.

Independent processes run baseline-A, candidate-A, candidate-B, baseline-B on
the same GPU. Each measurement has nine rounds of 25 replays. A CUDA graph
contains four complete calls, and reported GPU times are per call. The table
uses the mean of the two process medians. All round samples, including eager
outliers, are retained; no favorable rounds are selected. Host enqueue is
reported separately and must not be added to GPU time, since they overlap.

| T / H / D / checkpoint | Backward us (upstream -> candidate) | Bwd reduction | Graph E2E us | E2E reduction |
| --- | ---: | ---: | ---: | ---: |
| 8192 / 96 / 128 / 16 | 2458.01 -> 2061.39 | 16.14% | 3126.56 -> 2729.30 | 12.71% |
| 8192 / 96 / 128 / 0 | 3089.68 -> 2692.46 | 12.86% | 3570.56 -> 3173.29 | 11.13% |
| 16384 / 96 / 128 / 16 | 4910.32 -> 4116.52 | 16.17% | 6233.37 -> 5438.46 | 12.75% |
| 8192 / 96 / 64 / 16 | 1652.69 -> 1496.69 | 9.44% | 2090.10 -> 1941.40 | 7.11% |
| 2048 / 32 / 128 / 16 | 302.64 -> 290.72 | 3.94% | 431.14 -> 419.18 | 2.77% |
| 2048 / 32 / 128 / 0 | 385.96 -> 373.80 | 3.15% | 499.30 -> 487.12 | 2.44% |

All six fingerprints (O and five input gradients) match across all four
independent processes. The two process medians for captured GPU time are very
stable (each pair differs by less than 0.05%); host-bound eager measurements
are noisier. At the primary 8K checkpoint=16 shape, eager E2E is
3136.20 -> 2736.86 us (-12.73%), while host enqueue is 649.19 -> 567.74 us
(-12.55%). At checkpoint=0 the enqueue improvement is only 2.16%; no universal
host-overhead speedup is claimed. The short-sequence improvement is a control
result of the shared implementation, not a new shape specialization.

An additional same-address, alternating AB/BA GPU-only control runs ten rounds
of 30 replays with baseline kernels loaded under the candidate host. At
8K/H96/D128/checkpoint=16 it measures 2463.30 -> 2065.75 us, a **16.14% latency
reduction**, with all five gradients byte-identical. This isolates a kernel
gain from the host binding optimization. It ran on GPU 2; its absolute times
are not mixed into the GPU-1 matrix.

## Precision and functional validation

**29 FP64-audited cases** pass with byte-identical paired gradients and
identical FP64 error metrics, without enlarging any tolerance:

- Ten dense-checkpoint cases: FP16/BF16, D128, safe gate with trainable
  a_log/dt_bias, lengths [0, 17, 31, 128]; normal, small-norm, correlated,
  cancellation and radial inputs.
- Ten recompute cases: FP16/BF16, D64, mild non-safe log gates, the same ragged
  lengths and five input distributions.
- Two coarse-checkpoint cases: BF16/D128, cadence=32, safe-gate parameters,
  small-norm and strong-decay inputs.
- One 8192-token BF16/D128 dense-checkpoint case with trainable gate
  parameters and batch-invariant uncut scheduling.
- Six 1024-token, two-head chain cases (8 or 16 pieces): FP16/BF16 normal and
  small-norm dense checkpoints, plus BF16 recompute and coarse checkpoints,
  all with safe-gate parameters.

The independent FP64 recurrence consumes the same rounded input values and
uses the source contract's additive normalization epsilon of 1e-6. The audit
checks forward errors, each gradient's FP64 RMS/max error, paired elementwise
error increases, and byte identity. Identical baseline/candidate errors mean
no regression, not exact agreement with FP64. In particular, a nearly zero
radial reference gradient can have a large relative error in both versions.

These are bounded numerical-regression cases, not certification of arbitrary
FP16 or extreme non-safe gates. The separately tracked steep-decay precision
problem remains outside this performance branch.

Functional regression: **104 pytest cases passed** (84 native/binding/order/
multistream cases plus 20 public KDA API cases). Coverage includes fresh operand
and workspace storage, default/non-default streams, CUDA-graph changed-input
replay, grouped-head reductions, output strides/padding, log/linear/safe gates,
optional states/adjoints, trainable gate parameters, dense/coarse/recomputed
checkpoints, and uncut/chain host bindings. The warm/cold launch ABI checks
cover 36 combinations without substituting a second operand frame.

## Sanitizer: distinguish backward from the complete pipeline

Compute Sanitizer 2025.3.1 instruments all launched kernels; there is no kernel
filter. Its process launch timeout is disabled. Pytest isolation and watchdogs
are disabled. Memcheck uses `--report-api-errors no` only to avoid intentional
unsupported-driver/VMM capability probes in the Python runtime; device-memory
instrumentation remains enabled. Thus this is a device-memcheck result, not a
claim that every CUDA API probe succeeded.

The tool's `--log-file` output and pytest stdout/stderr are separate artifacts.
`evidence.json`'s `raw_log_sha256` refers only to the tool log. Both clean
memcheck logs are the same 62 bytes (banner plus zero-error summary), so their
equal hashes are valid; the different pytest durations are in the separate
`final-{bwd,e2e}-mem-pytest.log` files. Their file names and distinct SHA256
hashes are recorded alongside the tool-log identities after checking the
original locally archived bytes, not reconstructing them from summaries.

- Final backward-only memcheck: 8 cases passed, **0 errors**.
- Final backward-only racecheck: 8 cases passed, **0 errors, 0 warnings**.
- The same backward-only cases against exact upstream: pytest also passes,
  but racecheck reports **80 warnings** and exits 99. This is a RED/GREEN
  control, not an inference from numerical agreement alone.
- Final full forward+autograd-backward memcheck: 8 cases passed, **0 errors**.
- Final full E2E racecheck: pytest passes, but **1272 hazard groups
  (406 errors, 866 warnings)** are reported in prefill/recompute, not backward.
  Exact upstream's same full E2E control reports 1368 groups (408 errors,
  960 warnings), including backward as well as those prefill/recompute kernels.
  Counts vary with scheduling; their difference is not a quantitative
  optimization or correctness metric.

The full E2E racecheck is therefore **not clean**. Its prefill/recompute sources
are byte-identical to upstream. Those handoffs are already handled by the
independent precision branch; this task does not silently merge that branch.
The newly added caller-checkpoint regression isolates backward from forward
and recompute so that their existing reports cannot obscure backward ownership.
It covers D64/D128, FP16/BF16, optional states/adjoints/parameters, five chunks,
tail handling, phase wraps, eager execution and changed-input graph replay.

## Final ordering and compatibility audit

Fence audit: passed for the changed backward edges. Execution completion and
memory visibility were checked independently:

| Resource | Handoff and reuse protection (CTA scope) |
| --- | --- |
| Raw Q/K/V/gate | TMA transaction completion/acquire; CG0+CG1+CG2 done arrivals, with CG2 retaining ownership through projection |
| dO | MMA and epilogue acquire before their first reads; epilogue plus CG1 acknowledgement after the last dO-reading MMA's committed epoch |
| K decay/inverse | Generic stores, async-shared proxy bridge, release/acquire; independent MMA and CG2 reuse acknowledgements |
| Decay scale | CG0 ready release/CG1 acquire; CG1 reader-done phase, including absent seeds, first/steady/tail transitions |
| Raw Q/K TMEM | Store completion before publication; explicit `tcgen05.wait::ld` before reader-done permits overwrite |
| Q/K gradient transpose | 128-thread named barrier after stores, exact register reduction, proxy fence and release/acquire before TMA reads |
| Output staging | Elected issue in PTX; warp-wide commit/wait and completion arrivals protect the one-behind TMA ladder |
| Descriptors and workspace | Existing prologue/PDL ordering retained; plan holds indices, never per-call storage or stream |

The actual CuTeDSL 4.7 lowering was inspected: successful waits use
`mbarrier.try_wait.parity.acquire.cta`, ordinary arrivals use
`mbarrier.arrive.shared.b64`, and generic/async bridges use
`fence.proxy.async.shared::cta`. The omitted arrival qualifiers imply release
and CTA scope in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive).
Generated PTX contains both TMEM load and store drains. Native SASS shows the
expected 144/152/160 register allocations and no LDL/STL in the main backward
kernel (the unchanged descriptor prologue does use local storage). PTX's
elected store regions lower to ten native UTMASTG tile instructions, split
across the four output tensors; this static count alone is not a dynamic
instruction-issue measurement.

Compatibility audit: passed for the native/public, cold/warm, stream/capture,
optional-input, output-layout and checkpoint/schedule intersections above.
No new allocation, D2H read, device synchronization or retained operand frame
was added to steady-state execution. Docstring audit: passed for changed
entry points and helpers; historical staging claims are explicitly marked.
Changed-file formatting, license checks, syntax checks and `git diff --check`
passed. The cleaned baseline-loader benchmark also passed a final GPU smoke
comparison after removing its obsolete staging-override adapter. No PR is
created by this task.
