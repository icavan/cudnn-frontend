# Optional SM90 Tensor Core KDA summaries

`plan_name="kda_summary_hopper_tc"` adds a direct-M BF16 Tensor Core provider
for the existing `kimi_delta_attention_summary` and
`kimi_delta_attention_summary_bwd` APIs. It builds on the native FP32 summary
provider in [PR #2473](https://github.com/NVIDIA/cudnn-frontend/pull/2473).
The default remains `kda_summary_hopper_cuda`: the new provider trades some
precision and workspace for faster summaries, and requires explicit selection.

## Arithmetic

Each 16-token tile is represented as a small triangular solve and a rank-16
state update. In forward, with per-channel inclusive log prefix `G_i`, prepare
`A_i = k_i * exp(G_i)`, `B_i = k_i * exp(-G_i)` and
`R_i = k_i * exp(G_last - G_i)`. BF16 Tensor Core products compute
`L_ij = A_i @ B_j` and the incoming-state projection `P_i = S @ A_i`.
Solve in FP32:

```text
y_i = beta_i * (v_i - P_i - sum_{j<i} L_ij * y_j)
S_next = S * exp(G_last) + sum_i y_i * R_i
```

Projection and update matrix operands are BF16; their accumulators, residuals,
beta multiplication, triangular solve, and recurrent state carry are FP32.
The near-one carry decay uses `expm1` plus FMA. NVRTC fast math is disabled.
The reverse recurrence uses exclusive reverse prefixes and includes the
scaled Q/dO forcing term; it returns G and the reverse transition, not the
ordinary attention dQ/dK/dV/dg/dbeta gradients.

Separate CTAs evolve M from identity without any V/Q/dO/seed contribution.
There is no subtractive `F(I,V)-F(0,V)` probe. This removes the probe's
value-dependent cancellation but **does not remove BF16 operand rounding**.
Forward and reverse factorizations need not produce bitwise transposes.
For a tile with a nonfinite log prefix or absolute prefix above 20, the device
takes the direct per-token FP32 recurrence, avoiding unsafe reciprocal gate
factorization. This range fallback is not a promise of FP32 parity on other tiles.
It also has a performance cost: at T=1024/H=8/g=-5 on H20, the reverse
summary took 2290.82 us versus 694.08 us for the default FP32 engine (3.30x
slower). Prefer the FP32 provider for workloads dominated by these tiles;
the weak-decay speedups do not apply to all gate distributions.

A parallel producer prepares each K/gate coefficient tile once. The recurrent
H/G and M CTAs share those coefficients instead of repeating their preparation
for every state-row tile. The kernels use synchronous warp MMA with BF16 inputs
and FP32 accumulation, not TF32 or asynchronous WGMMA. Shared-memory pitches
are padded to reduce bank conflicts; each CTA owns 16 state rows.

## Binding, workspace and compilation contract

- SM90 only; BF16 THD Q/K/V/dO, FP32 log gates/beta/optional seed and outputs,
  equal head counts, K=V=128. Normalize Q/K and activate gates/beta before
  calling. Safe gate, in-kernel normalization/sigmoid, context/checkpoints and
  indexed/in-place state are unsupported by this provider.
- Enable `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`. A complete CUDA toolkit with
  NVRTC and MMA headers is required; `CUDA_HOME` can select it. No new
  CuTeDSL version requirement is introduced. Missing support is declined,
  not silently routed to another precision when this plan is pinned.
- Nonnegative outer strides with unit innermost stride and INT32/INT64 packed
  offsets are supported. Offsets must be valid, nondecreasing and bounded by
  the declared token capacity. Empty spans return their seed (or zero) and
  identity; zero sequences produce empty outputs.
- Graph execution writes caller-owned outputs and scratch on the handle's
  stream. Outputs must be mutually disjoint and not overlap inputs. Scratch
  must be contiguous, 32-byte aligned, on the same CUDA device, and disjoint
  from every nonempty operand. Retain buffers until stream completion; use
  separate outputs and scratch for concurrent executions. Bare pointers retain
  the graph API's caller responsibility for device and accessible capacity.
- For token capacity T, sequence count N and head count H, workspace is
  `(T//16 + N)*H*10400` bytes forward or `(T//16 + N)*H*15456` reverse; it is
  zero if T or N is zero. Query `graph.get_workspace_size()` rather than
  hard-coding the formula. The public convenience wrappers allocate outputs
  and scratch; graph execution allocates neither.
- Sequence i starts at coefficient slot `floor(cu[i]/16)+i`. Its
  `ceil(length_i/16)` tiles fit before the next sequence base, so the fixed
  capacity covers any THD distribution. Gaps are never read. Device offsets,
  gates and values can change between CUDA Graph replays without a host
  length read, rebucketing or compilation. Warm convenience wrappers before
  capture; graph plans compile at build time.
- One compiled module per device contains both directions and both stages.
  Lengths, strides, offsets and pointer values are launch parameters, not
  compile-cache specializations. No plan owns device scratch.

Workspace grows with local token capacity. For one sequence with T=16,384
and H=8 it is 81.33 MiB forward / 120.87 MiB reverse. A single GPU evaluating
64 such spans at H=1 uses 650.63 / 966.94 MiB. Real CP ranks allocate for their
local inputs, not the global sequence length. The FP32 provider uses zero
global workspace and is preferable when its precision or memory cost matters.

## Validation and measurements

The reproducible harness is
`benchmark/linear_attention/benchmark_sm90_summary_tc.py`. It reports both
state and transition relative L2, maximum absolute error, reference norm,
workspace, and CUDA Graph latency. `--fp64` uses an independent recurrence
over the quantized inputs; otherwise the reference is the native FP32 engine.
`--cp64` composes 64 logical spans in FP64 to isolate summary error. It does
not simulate communication, the production FP32 merge, or training loss.
For very long repeated-key cases, use `--correlated-keys --closed-form`:
an FP64 sum using the recurrence's parallel/orthogonal eigenvalues avoids
treating a million-step FP32 recurrence as exact. The closed form is checked
against the independent FP64 recurrence in both directions on small THD cases.

H20 measurements, exact commands and test outcomes are recorded in the
[validation report](../../../test/python/determinism_goldens/README.md).
