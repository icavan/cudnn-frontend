# KDA correctness and profiling continuation — 2026-10-15/16

> Historical experiment: upstream now already uses two raw stages. Its rebase
> validation is in `../optimization_20261016_upstream/REPORT.md`; the three-to-two
> stage gain below must not be claimed against that newer baseline.

## Outcome and scope

The final audit found and fixed a dO ownership gap in the existing optimized
backward. A small, independent resource optimization then selects two raw stages
only for normalized D128 uncut backward. Other schedules retain three stages.

Final validation is complete: 120 pytest cases, twelve byte-identical FP64
audit cases, zero reported racecheck hazards, and zero device-memcheck errors
(with the API-reporting qualification below).

On BF16 B1/S8192/H96/K=V=128 with Q/K normalization, beta sigmoid, safe gate
and checkpoint=16, the two-stage candidate reduces backward latency by **3.72%**
and forward+autograd backward latency by **2.70%**, compared with a **dO-corrected
three-stage baseline**. The correctness fix is present in both variants and is
not counted as an optimization gain. The additional 20% goal was not achieved.

The measured region is raw Q/K/V/gate/beta through O and their gradients.
Projection, optimizer, distributed communication and model loss are excluded.
Timing inputs have no initial state or trainable gate parameters; those paths
are covered separately by correctness checks.

## Provenance and environment

- Starting optimized source: `220d5e9a0345c1f3c23320802f489484dd45e6fb`.
  It already includes the October 12 adaptation of `57e2f85a`; that earlier
  optimization is not claimed again as new work.
- Corrected three-stage baseline: `45b11933a284143b383b0d6dca11e16541f17174`. Its main-kernel
  source SHA256 is
  `37abae3a9eadd562e3b1b6c105b58cf2649c9b1d3ba2e496529a4aaa98fc5d87`.
  It is exactly the starting source plus the dO handshake fix, with the
  original three-stage builder and unchanged config/summary modules.
- Host: `aistudio-61550139-ssctl`; source:
  `/root/kda-opt-20261012/continuation-20261015/source`.
  `cudnn.__file__` was verified there; the other editable checkout was
  bypassed only inside these test processes, without changing its installation.
- The reused extension's 108 in-repository build inputs match this checkout;
  aggregate SHA256:
  `de2b2d8276f9ecea99ace85ce065c206a7dda70eb1b5a1c4f49e69174ba3b754`.
- Python 3.10.20, Torch 2.11.0+cu130, cuDNN 9.19, driver 580.95.05,
  `nvidia-cutlass-dsl[cu13]==4.7.0`. Final runs explicitly select
  `CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13`.
- Runtime capability is 10.0; no commercial SKU is inferred from the product
  string. Final A/B and validation use GPU 1, verified idle. Initial exploration
  used GPU 0; different GPUs' absolute times are not used as an A/B comparison.
- Remote helpers disable automatic cleanup. Pytest isolation/watchdogs are off.
  No process was terminated.

## Correctness fix: dO readiness and reuse

In `6ce21b69`, moving dQ MMA later also moved its dO-ready wait past the
earlier dU/dstate dO reads. The epilogue did not directly acquire dO readiness,
and its 32-thread done signal did not independently cover the late dQ MMA.
TMA issue order alone does not prove completion of a different copy.

The minimal fix:

1. Move the existing MMA dO-ready wait before its first dO-consuming MMA.
2. Have the epilogue acquire dO readiness before its generic ldmatrix reads.
3. Combine its 32 done arrivals with 128 CG1 arrivals. CG1 acknowledges after
   the existing committed dstate epoch, which follows **all** dO-reading MMAs,
   including the late dQ inter term.

This adds no new MMA commit, allocation, broad fence or arithmetic change.
The summary kernel already guards its first dO MMA and acknowledges reuse from
CG1 after the committed dstate epoch; it does not need this change.

Without the fix, adverse asynchronous timing could expose unready/overwritten
dO or a missed barrier phase. No nominal numerical failure is claimed to have
been reproduced. All three source-ordering regression detectors are RED on
`220d5e9a` and GREEN on the fix. Numerical agreement and sanitizer silence
alone were not accepted as an ownership proof.

## Profiling and resource optimization

Initial NCU on the pre-fix three-stage main kernel, 8K/H96/D128:

- 96 CTAs on 148 SMs (0.65 waves), 512 threads/CTA, 222.21 KB dynamic SMEM.
- 128 reported registers/thread; no local spill requests.
- Achieved occupancy 24.98%; SM throughput 27.97%, DRAM 27.50%, L1/TEX 68.26%.
- No-eligible-warp cycles 54.26%; long-scoreboard about 34.79% of warp time.

These counters are not attainable-speedup estimates. The initial NCU capture
used the default libcudart 12 loader; subsequent timing/validation explicitly
uses libcudart 13. Both runtimes are installed.

Before the dO fix, eager E2E was 2.667 ms versus graph E2E 2.652 ms, despite
about 0.616 ms of host enqueue work: most host work overlaps GPU execution.
The enqueue duration is not an additional 0.616 ms on the critical path.

Exploratory same-input eight-round AB/BA at 8K/H96/D128:

| Variant | Backward us | Change |
|---|---:|---:|
| Current three-stage control | 1991.78 | control |
| Independent unchanged-source compilation | 1990.11 | -0.08% |
| CG2 register allocation 168 | 2009.47 | +0.89% |
| CG1 register allocation 160 | 1994.75 | +0.15% |
| Two raw stages | 1922.28 | -3.49% |
| Force three pieces | 2856.04 | +43.39% |

Register changes and extra pieces are discarded. Three pieces grow workspace
from 5504 bytes to about 56.65 MB. A global two-stage default regressed the
2K/H32 chain case by 1.58%, so chain and summary stay unchanged. D64's gain was
only 0.87%; D64, rectangular and non-normalized schedules retain their defaults.

The retained uncut D128-normalized override is applied **before** deriving SMEM
cosizes, descriptors and TMA geometry. It reduces raw-buffer footprint, but does
not create a second resident CTA. Public APIs, eligibility, compile-key axes,
checkpoint dtypes and reduction ordering are unchanged.

## Final corrected performance

Both variants include the dO fix. GPU 1, same process, shared input addresses,
eight alternating AB/BA rounds of thirty replays; microseconds:

| Workload | Corrected three-stage baseline | Two-stage candidate | Latency reduction |
|---|---:|---:|---:|
| 8K/H96/D128, checkpoint=16, backward | 1995.71 | 1921.38 | 3.72% |
| 8K/H96/D128, checkpoint=16, forward+backward | 2656.87 | 2585.03 | 2.70% |
| 8K/H96/D128, checkpoint=0, backward | 2623.01 | 2550.74 | 2.76% |
| 8K/H96/D128, checkpoint=0, forward+backward | 3102.28 | 3037.68 | 2.08% |
| 16K/H96/D128, checkpoint=16, backward | 3987.90 | 3835.62 | 3.82% |

All measured outputs/gradients are bitwise identical, including signed zeros.
Compilation, data generation, capture and validation are outside timing.

## Final precision, boundary and sanitizer checks

The dO-fixed candidate passed final re-validation. Earlier stage-only results
are retained as historical evidence, not attributed to the new final kernel:

- All 85 metadata, ABI, native cache/stride, source-ordering and replay cases,
  plus 35 selected KDA reference/gate/checkpoint/state/two-stream cases:
  **120 passed**. The three strengthened ordering detectors also pass in a
  separate run and remain RED on the pre-fix source.
- FP16 and BF16 FP64-oracle audits on the exact rounded inputs: five normal,
  tiny-norm, correlated, cancellation and radial distributions with lengths
  `[0,17,31,256]`, plus length 8192/H2 using `batch_invariant=True` so both
  baseline and candidate run uncut. Safe gate and trainable A_log/dt_bias are
  included; all seven gradients in all twelve cases are finite and bitwise
  identical to the corrected baseline, with zero FP64 error increase.
- The independent recurrence uses the served additive squared-norm epsilon
  `1e-6`. Closed-form, independent-head and signed-zero auditor self-tests pass.
  Near-zero radial gradients can have large relative FP64 error in both
  variants; identity is not a claim of exact FP64 accuracy or model convergence.
- Focused racecheck and device memcheck on eight committed replay cases:
  T=65/H1, D64/D128, FP16/BF16, checkpoint 0/16, optional FP32 state and gate
  parameters, tails, ring wraps and changed-input CUDA Graph replay.
  Both runs pass all eight cases. Racecheck reports 0 hazards displayed,
  0 errors and 0 warnings; device memcheck reports 0 errors.
- Memcheck uses `--report-api-errors no`: only device-memory checking is
  claimed, not an unqualified host-API sanitizer pass. Existing pytest
  marker/DSL-deprecation warnings are separate from sanitizer warnings.

An earlier large racecheck (T=257/H96, pre-dO-fix global-uncut two-stage trial)
has not completed and is **not** counted as a passing result. At the reporting
cutoff PID 346949 was still running on GPU 0 while explicit termination approval
was pending. Neither a tool defect nor a kernel deadlock is asserted as its
cause; the final candidate excludes its D64 two-stage configuration.

## Ordering review and stopping point

Fence audit: passed after the dO fix for these ownership edges:

| Edge | Protection |
|---|---|
| TMA raw Q/K/gate/V → compute → reuse | Transaction completion/acquire waits; all 384 raw-reader arrivals |
| TMA dO → MMA and epilogue | Each consumer acquires dO readiness before its first read |
| All dO reads → TMA overwrite | Epilogue + CG1 done credits; CG1 waits the MMA epoch after the last dO MMA |
| In-place gate/decay → readers → reuse | Existing named barrier, async-proxy fences and independent reader acknowledgements |
| TMEM raw Q/K → CG2 → next writer | Existing explicit tcgen05 load drain before reader-done |
| Generic output SMEM → TMA store → reuse | Proxy fence, ready barrier, elected issue, commit/read waits and warp-wide done |

Installed DSL 4.7 helpers were inspected: plain arrivals default to release,
parity waits to acquire; MMA completion delegates to CTA-group-1 tcgen05 commit.
The [PTX async-proxy rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-operations)
and [tcgen05 commit contract](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.2.pdf)
underpin the completion/visibility reasoning. First free-slot phase 1, ready
phase 0, stage wraps, empty work items, persistent-item transitions and tail
drains must all retain matching counts and phases. The fix adds no new
asynchronous arrival, so it needs no new asynchronous-commit tail drain.

Compatibility audit: passed for the tested dtype, dimensions, gate, checkpoint,
state, layout and stream intersections. Public API/eligibility/dependency floors
are unchanged. Docstring audit: passed for the changed private builders,
schedule description and audit/benchmark entry points. Changed-file pre-commit,
syntax checks and `git diff --check` pass.

This is an engineering stopping point, not proof that KDA has no further
optimization space. The extra 20% goal was not met. Other measured candidates
lose, and remaining host work is mostly off the critical path. Gate-postpass
fusion or a larger kernel redesign was not prototyped; no negative measured
result is claimed for those ideas. No further speculative change is retained.

## Reproduction and artifacts

Export `kda_bprop_config.py`, `kda_bprop_f16.py` and
`kda_bprop_summary_f16.py` from the corrected baseline commit into one directory.
The updated `benchmark_kda_kernel_ab.py` preserves the old builder's own raw
depth rather than applying the candidate host override to it. Run with
`--baseline <directory> --tokens 8192 --heads 96 --checkpoint 16
--rounds 8 --repeats 30 --json <output>`; add `--e2e` for forward+autograd
backward. Repeat with checkpoint=0 or tokens=16384.

For `audit_kda_precision.py`, use `--safe-gate --parameters --require-identical
--distributions normal small_norm correlated cancellation radial`. For long
uncut coverage add `--lengths 8192 --heads 2 --batch-invariant
--distributions normal`. Reference work is outside performance timing.

The committed replay test is
`linear_attention/frost/test_kda_execution.py::test_uncut_bwd_raw_ring_wraps_and_changed_input_replay`.
From `test/python`, disable pytest isolation/watchdogs, then prefix Python/pytest
with `compute-sanitizer --tool racecheck --kernel-name kns=cutlass_frost_kda_bprop_
--racecheck-report analysis --error-exitcode 86`. For device memcheck use
`--tool memcheck --report-api-errors no` with the same filter/error exit code.
No automatic process termination is enabled.

The final round samples, source hashes and per-gradient FP64 errors are in
`do_fixed_evidence.json`; `racecheck.log` and `memcheck.log` retain the actual
pytest/tool output. `evidence.json` is explicitly marked pre-dO-fix exploration.
Full profiler reports and raw logs remain under
`/root/kda-opt-20261012/continuation-20261015/results`. Pre-dO-fix exploration
is preserved separately from final corrected evidence. This run establishes
neither a universal speedup nor SM103/SM107 behavior.
