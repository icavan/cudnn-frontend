# KDA forward/backward E2E optimization — 2026-10-12

## Outcome and scope

On `aistudio-61550139-ssctl`, SM100, BF16 THD, B=1/S=8192/H=96/K=V=128,
in-kernel Q/K normalization, beta sigmoid and safe gate, the final backward
GPU replay latency is **20.38% lower** with dense checkpoint reuse (`checkpoint=16`).
The corresponding forward+autograd backward replay latency is **16.15% lower**.
The result is shape/mode-specific, not a promise of 20% for every configuration.

The measured region starts at Q/K/V, raw gate and beta and ends at O and their
gradients. Projection, optimizer, distributed communication and whole-model loss
are excluded. The timing inputs have no initial/final state or trainable A_log /
dt_bias; those optional paths are covered separately by correctness tests.

## Provenance and environment

- Branch: `icavan/perf-kda-e2e`, destination `code.alipay.com/ling/cudnn-frontend`.
- Starting checkout: `a3f3513ad57b686aa085201255ff379be9a5ba1c`.
- Safety-fixed baseline: `51c73656f2361b02d9007293d163967b2e39d8c2`, after the
  existing elected-lane TMA-store, decay-scale-reuse and in-place gate-exchange fixes.
  Their performance effects are **not** counted as new optimization gains.
- GPU pipeline changes adapt `57e2f85ac9bbd61fefdd55b971301035f64a1ce1`
  (`perf: optimize KDA backward pipeline`) onto the current safety-fixed source.
  The new raw-slot TMEM drain is retained in the final candidate.
- Source checkout: `/root/kda-opt-20261012`; freshly built C++ extension plus this
  checkout's Python package, not an overlay of the wheel's graph implementation.
- Python 3.10, PyTorch 2.11.0+cu130, CUDA toolkit 13.0.88, driver 580.95.05,
  cuDNN 9.19.0.56, `nvidia-cutlass-dsl[cu13]==4.7.0`, cuda-bindings 13.4.3.
- Benchmark device: physical ordinal 0. Runtime capability 10.0; product name
  `NVIDIA L20C` is not used to infer a commercial SKU. Tests also used ordinals 1/2.
- Compute Sanitizer 2025.3.1.0. No process termination or automatic tmux cleanup
  was used; remote calls set `REMOTE_EXEC_AUTO_CLEANUP=0` and pytest watchdogs
  were disabled (`CUDNN_TEST_TIMEOUT=0`, `CUDNN_TEST_NO_ISOLATION=1`).

## Changes

1. The shared LA op helper retains only descriptor/UID order and uses the public
   ordered-binding API on warm calls. Fresh buffers, workspace and stream are
   validated per invocation; changed order still uses the mapping API. This
   does not bypass graph routing or specialize callers by engine.
2. Workspace allocation uses the canonical current-device/current-stream fast
   contexts, retaining per-call scratch and thread-private handles.
3. Warm KDA backward prepares an immutable positional binding recipe once,
   rebinding current operands and carved scratch without rebuilding launch
   dictionaries/kwargs. Chain/uncut, dense/coarse checkpoints, optional states,
   gate parameters and grouped-head folds preserve the cold ABI.
4. GPU backward uses three raw TMA stages, rebalances CG1/CG2 registers, schedules
   dQ MMA after the state-gradient critical path, and performs the D128 L2 row
   projection using cooperative eight-lane reductions and retained raw vectors.
   CG2 acknowledges decay consumption independently; raw slots are released
   only after their remaining Q/K consumers finish.

No public signatures, input layouts, checkpoint dtypes or compile-key data-value
contracts were changed. The DSL floor remains 4.7.0 (Rule 7). Per-call workspace
ownership and the common backend/FROST graph contract remain intact (Rule 9).

## Final timings (microseconds)

`baseline.json` and `final.json`: warm enqueue/eager CUDA-event E2E and CUDA
graph replay measured separately. A graph contains four calls, normalized per
call. Input generation, compilation, validation and profiling are outside timing;
capture/replay and autograd use one explicit nondefault stream.

| B1/S8192/H96/D128 | Baseline | Final | Latency reduction |
|---|---:|---:|---:|
| checkpoint=16, graph forward+backward | 3162.09 | 2651.55 | 16.15% |
| checkpoint=16, graph backward | 2498.57 | 1989.26 | 20.38% |
| checkpoint=16, eager forward+backward | 3173.82 | 2660.62 | 16.17% |
| checkpoint=16, host enqueue forward+backward | 631.21 | 587.48 | 6.93% |
| checkpoint=16, host enqueue backward | 142.23 | 120.39 | 15.36% |
| checkpoint=0, graph forward+backward | 3608.53 | 3097.25 | 14.17% |
| checkpoint=0, graph backward | 3130.07 | 2618.14 | 16.36% |

Ordinary calls still pay Torch custom-op/autograd overhead; this is not eliminated
by the host migration. Graph replay submission was below 1 us per normalized call
in the final run; CUDA Graph use is measured, not automatically enabled by the op.

The stronger GPU control in `kernel_ab.json` uses both variants in **one process**,
the same input addresses/values, the candidate host path for both, and eight
alternating AB/BA rounds of thirty replays. Derived config dataclass defaults are
loaded under each snapshot's own CFG, rather than changing CFG after import.

- Baseline: **2500.740 us** (round range 2500.368–2501.579).
- Final: **1991.013 us** (round range 1990.230–1991.526).
- Speedup: **1.256×**; latency reduction: **20.383%**.
- Candidate-minus-baseline gradient RMS relative differences: dQ `4.89e-6`,
  dK `5.75e-6`, dV `0`, dGate `1.46e-7`, dBeta `0`.
- Max absolute differences: dQ `1.22e-4`, dK `2.44e-4`, dGate `2.69e-8`.
  Reordered FP32 reductions/FMA can change rounding; bitwise baseline identity
  is not claimed. Existing precision tolerances were not increased.

The initial 2K/H96 baseline had unstable isolated backward timing. It is retained
in raw data for transparency but is not used for a performance claim. Low-head
chain modes showed smaller gains than the 96-head uncut target.

## Correctness and sanitizer evidence

Final ordinary test runs: 37 warm ABI/drain cases, 5 shared-resource cases,
35 reference/shape/checkpoint cases, 8 backward parity cases and 9 safe-gate /
multi-stream cases: **94 passed**. Ten parametrizations were intentionally
skipped by the existing parity sweep's sampling policy, not by a support failure.

- Prepared warm/cold ABI checks cover chain/uncut, checkpoints 0/16/32, log/linear/
  safe gates, optional states/parameter gradients and grouped-head folds.
- Existing GPU tests cover FP16/BF16 backward, K/V 64/128, ragged/empty sequences,
  output strides, capture/rebinding, dense/coarse checkpoints, safe-gate parameter
  gradients and same-graph two-stream backward with fresh caller-owned scratch.
- The shared helper's native backend ADD control checks warm ordered execution,
  changed-input capture replay and fresh operand bindings on a side stream.
- The TMEM release detector was observed RED without the final load drain and
  GREEN with it. Its concrete technique is recorded in `python/cudnn/AGENTS.md`.
- Final **racecheck: 7 passed; 0 errors, 0 warnings, 0 hazards**.
- Final **device-memory memcheck: 7 passed; ERROR SUMMARY: 0 errors**, using
  `--report-api-errors no`; device memory checks and `--error-exitcode 86` remain
  enabled. Filter `kns=cutlass_frost_kda_bprop_` covers bprop and bprop-summary.

The unsuppressed memcheck run returned 86 with **42 CUDA API diagnostics**:
41 `cuGetProcAddress_v2` invalid-value reports and one `cuMemCreate` not-permitted
report. A separate control doing only FE device/handle initialization and a Torch
allocation, **with no KDA**, reproduced the same 42 diagnostics. Device-memory
checking was therefore repeated separately. This is not an unqualified all-API
sanitizer pass. Full diagnostic logs remain in the remote source directory:

- `memcheck_final.log`, SHA256 `a38a73135f96be8679216b60545b5fe66d94059f9b825add24c0bdb8e66375d9`.
- `api_frontend_control.log`, SHA256 `9e90560dadeef38a368b0014860db8d72206501cd4ced3814a05c736d1f7a7ef`.

## Final fence and compatibility review

Fence audit: passed for the changed edges:

| Edge | Ordering / reuse protection |
|---|---|
| Gate input → in-place prefix | All CG0 input reads precede its named barrier and swizzled writes |
| CG0 decay → MMA/CG2 → reuse | Acquire ready waits; independent MMA, CG1 scale and CG2 reader acknowledgements |
| Raw Q/K TMA → CG2 row projection → overwrite | TMA completion, retained raw vectors, deferred CG2 raw-done arrival |
| CG2 projection scratch → readers → reuse | CG2 named barrier; the next one-stage dQ/dK store-done waits follow all CG2 ready arrivals and TMA completion |
| TMEM raw Q/K reads → CG0 next store | Explicit `tcgen05.wait::ld` before the reader-done release/acquire handoff |
| SMEM dQ/dK → TMA store → reuse | Async-proxy fence, all-producer ready barrier, elected issue, commit/wait and warp-wide done barrier |

The installed DSL's `mbarrier_arrive` defaults to release and its parity wait to
acquire. Its `tcgen05_ld` does not add a wait. The explicit drain follows the
[PTX tcgen05 memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-memory-consistency-model): register dependencies alone do not establish anti-dependency memory ordering.
Sanitizer silence supports, but does not replace, this audit.

Compatibility audit: passed for the tested dtype/layout/checkpoint/state/gate/
stream intersections, with provider-independent binding kept inside FE.
Docstring audit: passed for the shared execution helper, warm backward binding
helpers and updated GPU schedule/stage descriptions. Changed-file pre-commit,
syntax checks and `git diff --check` passed.

## Reproduction

The two scripts one directory above `results/` are `benchmark_kda_e2e.py` and
`benchmark_kda_kernel_ab.py`. First build/install this checkout and verify
`cudnn.__file__`. Run E2E with `--tokens 8192 --heads 96 --checkpoint 16 0
--repeats 30 --rounds 8 --json <output>`.

For the AB script, export `kda_bprop_config.py`, `kda_bprop_f16.py` and
`kda_bprop_summary_f16.py` from baseline commit `51c73656…`, then pass their common
directory as `--baseline`, with `--tokens 8192 --heads 96 --checkpoint 16
--rounds 8 --repeats 30 --json <output>`. The script performs controlled module
swaps solely inside its benchmark process; it is not a production dispatch path.

This run does not establish full-model convergence, every pretraining shape, or
SM103/SM107 performance. Production validation should use the actual model's
local packed capacity, head counts, checkpoint cadence and gate parameters.
