# KDA FP32 residual evaluation

## Change and scope

BF16 KDA now retains its FP32 projection accumulator through residual subtraction and beta multiplication, rounding only the resulting Y operand. The same helper serves fused H/M summary, normal prefill, checkpoint recompute and backward Y regeneration. Prep-fed prefill retains FP32 subtraction; beta is already folded into its records. Backward dBeta uses the unrounded BF16-path residual. FP16 preserves the legacy packed arithmetic, including its beta rounding.

This is separate from selectable cross-rank merge precision. MMA operands, inverse precision, checkpoint storage, collectives and merge are unchanged. No production routing or public signatures change.

## Reproduction

The benchmark takes a complete source tree (including `python/cudnn`) and overlays that tree's Python implementation on an installed package providing the compiled extension. It verifies each measured kernel's resolved path and records source SHA256 hashes. It is not a fresh full-package build.

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  python benchmark/linear_attention/kda_fp32_residual.py \
  --source-root /path/to/baseline --output baseline.json --label baseline
CUDA_VISIBLE_DEVICES=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  python benchmark/linear_attention/kda_fp32_residual.py \
  --source-root /path/to/candidate --output candidate.json --label candidate
```

Use fresh output filenames; existing reports are never overwritten. Repeat in B/A order with `--performance-only`; add independent seeds with `--accuracy-only --seed 42` and `--seed 2026`.

Both revisions use identical CPU-generated BF16 Q/K/V/beta, FP32 gates and FP32 initial state. Accuracy uses two heads against an independent CPU FP64 per-token recurrence for H and identity-seeded M. The harness explicitly disables internal piece chains and d_v splitting on BOTH revisions: `batch_invariant=True` alone still cuts long sequences. The reference also compares each output token; it is not a comparison against the revision's own serial kernel.

Performance uses 16 heads on the same GPU, three warm calls, capture of four calls, three warm replays and 17 timed replays. Compilation, inputs and FP64 reference time are excluded. Two fresh-process runs per revision were ordered A/B/B/A, with persistent FE compilation reuse disabled. These are operator-stage GPU replay times, not host latency, model training throughput or distributed CP8 times. No GPU clocks were changed.

## Measured target and baseline

User-specified B300, CUDA 13 environment, Torch 2.12.1+cu130. Baseline commit: `473a2c8f767a4681915c7dc16b874f1877534413`. Candidate kernel hashes in the reports match the PR source. Resource inspection of the representative dumped forward/summary/recompute/backward cubins found unchanged REG/STACK/LOCAL sizes (main entry REG=128, LOCAL=0); this does not assert zero cost for the extra FP32 arithmetic.

## H/M accuracy

The tables focus on the H/M stage changed by this PR. Downstream output error also depends on query projection, gating/decay and the chosen case, so it is not used here as a proxy for H/M accuracy. Original JSON reports retain all measured diagnostics; no observations are deleted.

| Case | Seed | H relative L2: old → new | M relative L2: old → new |
|---|---:|---:|---:|
| standard (4096, D=64) | 1731 | 0.36566% → 0.32073% | negligible history; see absolute errors |
| long_memory (32768, D=128) | 1731 | 0.05266% → 0.03901% | 0.02023% → 0.01859% |
| standard (4096, D=64) | 42 | 0.37357% → 0.31895% | negligible history; see absolute errors |
| long_memory (32768, D=128) | 42 | 0.05281% → 0.03922% | 0.02022% → 0.01867% |
| standard (4096, D=64) | 2026 | 0.36216% → 0.31990% | negligible history; see absolute errors |
| long_memory (32768, D=128) | 2026 | 0.05322% → 0.03865% | 0.02019% → 0.01861% |

H summary matched ordinary uncut forward final state bitwise in all six random cases. Across the three seeds:

- 4K standard: H/final-state relative L2 falls 11.7–14.6%.
- 32K long-memory: H relative L2 falls 25.7–27.4%; M relative L2 falls 7.6–8.1%.
- Standard-case M has reference max magnitude around 1e-28 and absolute errors around 1e-30. Its relative error is not evidence about material history preservation.
- L2 improvement is not a pointwise/maximum-error guarantee. For long-memory seed 1731, H maximum absolute error changes from 1.32396e-4 to 9.21678e-5 and M from 1.19114e-4 to 1.11295e-4.

## Performance

| Tokens / D | Operation | Baseline mean of run medians (µs) | Candidate (µs) | Change |
|---|---|---:|---:|---:|
| 4096 / 64 | forward_uncut | 157.48 | 158.36 | +0.56% |
| 4096 / 64 | summary_hm_uncut | 188.23 | 189.77 | +0.82% |
| 4096 / 64 | backward_with_recompute | 749.42 | 749.59 | +0.02% |
| 32768 / 128 | forward_uncut | 1684.72 | 1718.53 | +2.01% |
| 32768 / 128 | summary_hm_uncut | 2079.78 | 2100.36 | +0.99% |
| 32768 / 128 | backward_with_recompute | 7898.74 | 7924.12 | +0.32% |
| 262144 / 128 | forward_uncut | 13375.22 | 13640.43 | +1.98% |
| 262144 / 128 | summary_hm_uncut | 16540.56 | 16696.64 | +0.94% |

The 32K/256K forward cost is about 2%, summary about 1%, and the 32K backward including recompute about 0.32%. Baseline repeat drift was within 0.18% for these cells. Treat the sub-microsecond 4K differences as small observations, not a resolved instruction-level performance claim: the runs use separate contexts and do not provide shared-address/code-placement controls. No throughput claim includes first compilation.

## Tests and retained evidence

- Original focused suite: **56 passed**. Includes all four K/V=64/128 combinations, first/steady chunks, BF16/FP32 beta inputs, explicit prep-fed execution, H/M operand packing, FP16 legacy arithmetic and backward saved-checkpoint/recompute parity with/without safe gate. Review follow-up: **58 passed**, with two representative cancellation cells at L0 and the full 56-cell matrix at L1; GPU-exclusive/group markers are preserved.
- Review RED/GREEN controls: the old files collect 56 L0 / 0 L1 cases and fail the intended selection contract; the updated files collect 2 L0 / 56 L1 cases. A host-only import tripwire verifies device eligibility before GPU DSL imports: old code fails for modeled absent/SM80/SM90/SM120 devices, while the fix skips those and admits SM100–SM103/SM107 (nine modeled conditions total, without querying real hardware).
- Existing selected KDA tests: **28 passed, 7 skipped** (head dimensions, ragged/initial-state forward, backward gradients against FP64, affine summary, prep and checkpoint paths); not a full repository test run.
- RED observed against the untouched baseline: the analytic correction was lost, with max absolute state error 0.001953125. New H forward and summary paths pass the 1e-7 absolute check on the correction itself.
- An initial FP16 compiler failure in the candidate was fixed by representing absent projections as compile-time None, preserving the old zero-seed packed path instead of converting constant zero through inline PTX. The corresponding baseline and corrected-candidate FP16 test both pass.
- Changed-file pre-commit and diff whitespace checks pass.

Remote evidence: `/root/kda-cp256k-20261019/fp32-residual-20261022`. Small JSON reports and logs were retrieved through Unison. Pilot `baseline-a1` did not disable internal chains and is deliberately excluded from the tables; its files remain preserved.

These synthetic local-stage results do not establish FLA parity, end-to-end CP8 accuracy, or a training-convergence benefit.
