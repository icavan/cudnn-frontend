# SM100 H96 DSA backward optimization on GB200

This branch targets BF16 DSA backward with `Hq=96`, `Dqk=576`, `Dv=512`,
`Sq=Skv=16384`, and `topk=2048`.  It keeps BF16 input/output and FP32 dKV
accumulation unchanged.

## Changes

- Use four 8-lane-subwarp `cp.async` KV-loader warps for the H96 generic
  specialization.  H96 launches two H64 head blocks per query, and the smaller
  CTA removes loader scheduling and barrier overhead without starving the
  single-stage KV pipeline.
- Add an explicit `pair_queries=True` path for workloads whose adjacent query
  tokens have high top-k overlap.  A preprocessing kernel builds the encoded
  union of each adjacent pair.  The main kernel views the pair as 192 virtual
  heads and launches three full H64 CTAs per pair instead of four CTAs.  The
  membership bits mask token-private KV rows before P/dS are consumed.
- Keep pairing opt-in.  At low overlap the larger union increases sparse-tile
  work and is slower than the one-query path.

The pair workspace is an `int32[Sq/2, min(Skv, 2*topk)]` union tensor plus one
`int32` length per pair.  For the 16K target it occupies 128 MiB + 32 KiB.

## Performance

Measurements used one NVIDIA GB200, CUDA 13.0, CuTe DSL 4.5.2, identical
synthetic tensors, 10 warmups, and 40 timed calls.  The NVIDIA baseline is
`origin/develop@73d8feb4`.  Its five repeated measurements span
90.790--90.810 ms.

| Backend | Adjacent top-k overlap | Time (ms) | Speedup vs NVIDIA develop |
|---|---:|---:|---:|
| NVIDIA develop | N/A | 90.800 | 1.000x |
| H96, 4 loader warps | N/A | 45.912 | 1.978x |
| H96 paired union | 80% | 42.035 | 2.160x |
| H96 paired union | 100% | 34.664 | 2.619x |

The four-loader ablation on the optimized one-query kernel was 47.499 ms with
16 loader warps, 46.548 ms with eight, and 45.913 ms with four.  The structural
paired path remains overlap-sensitive: prior sweeps put its break-even point at
about 66--67% adjacent-query overlap.

The benchmark intentionally avoids a dense reference forward so that a 16K
shape is inexpensive to construct.  It preserves the production tensor shapes,
dtypes, top-k contract, and softmax range.  Mathematical correctness is covered
separately by the PyTorch-reference regression test.

## Correctness and sanitizer results

The H96 pair regression uses two queries with common, first-only, and
second-only KV rows.  It checks all 96 original heads, including virtual heads
128:191 in the third H64 CTA, against the PyTorch reference.

- Targeted L0 tests: 7 passed.
- Pair-vs-one-query numerical comparison (`Sq=256`, `Skv=16384`, 80% overlap):
  - dQ max absolute difference: `9.536743e-7`
  - dKV max absolute difference: `6.103516e-5`
  - dSink max absolute difference: `6.984919e-10`
- Compute Sanitizer memcheck: `ERROR SUMMARY: 0 errors`.
- Compute Sanitizer racecheck: `0 hazards displayed (0 errors, 0 warnings)`.

The paired path changes FP32 accumulation order, so dKV and dSink are not
promised bitwise identical.  No lower-precision accumulator or approximate
output conversion is introduced.
