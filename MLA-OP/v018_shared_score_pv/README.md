# MLA Decode v018: Shared Score PV

## Objective

Remove a redundant shared-to-fragment score copy from the dominant
materialized QK plus output-part-0 kernel.

## Change

- retain the FP16 `Score_shared` tile already produced for the global
  prefix-relative score tensor;
- pass `Score_shared` directly to the output-part-0 PV GEMM;
- remove the separate FP16 `acc_s_cast` fragment and the
  `Score_shared -> acc_s_cast` copy;
- retain v017's adaptive reduce-max clear, cached split LSE, QK clear-accum,
  fused 576-D QK, shared KV part 0, split policy, and final reduction.

MACA supports the shared-score/shared-value GEMM layout used here. The same
score tile now serves both the global materialization copy and the part-0 PV
GEMM without a second staging step.

## Correctness

- All 31 official outputs are bitwise identical to v017.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v017 total | v018 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Materialized path | 24 | 24.29153 ms | 23.92361 ms | **+1.51%** |
| Batch 4 | 5 | 1.74065 ms | 1.70020 ms | **+2.32%** |
| Batch 8 | 5 | 3.17171 ms | 3.11324 ms | **+1.84%** |
| Batch 16 | 5 | 6.02924 ms | 5.93943 ms | **+1.49%** |
| Batch 32 | 5 | 12.38623 ms | 12.23434 ms | **+1.23%** |
| 31-case total | 31 | 25.72111 ms | 25.33056 ms | **+1.52%** |

All five aggregate samples improved: `+1.1831%, +1.5672%, +1.6467%,
+1.4799%, +1.5186%`. Every per-shape median improved. Targeted B4/2K and
B32/64K runs measured `+5.08%` and `+1.14%`, respectively.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`458d6cd8680be1d55e099bec1f651e437aca2abae492a7644f057bae18a5fe51`.
