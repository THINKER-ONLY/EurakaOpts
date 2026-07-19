# MLA Decode v015: QK Clear-Accum Scheduling

## Objective

Remove unnecessary scheduling overhead between score-fragment initialization
and the QK GEMM in both split-K attention paths.

## Change

- replace the standalone `T.clear(acc_s)` in the materialized QK plus output
  part 0 kernel with QK GEMM `clear_accum=True`;
- apply the same transformation to the direct split-K path before its latent
  QK GEMM, while retaining the accumulating RoPE QK GEMM;
- leave the no-split kernel unchanged because it already used
  `clear_accum=True`;
- retain v014's fused 576-D materialized QK, shared KV part 0, prefix-scaled
  weights, output partitioning, split policy, and final reduction.

On MACA this lets the accumulator initialization be scheduled with the GEMM
after the shared-memory input synchronization instead of as a separate
operation before that synchronization.

## Correctness

- All 31 official outputs are bitwise identical to v014.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v014 total | v015 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 6 | 1.37024 ms | 1.34677 ms | **+1.71%** |
| Batch 2 | 5 | 1.03634 ms | 1.01325 ms | **+2.23%** |
| Batch 4 | 5 | 1.74597 ms | 1.71971 ms | **+1.50%** |
| Batch 8 | 5 | 3.17007 ms | 3.14076 ms | **+0.93%** |
| Batch 16 | 5 | 6.03131 ms | 6.00059 ms | **+0.51%** |
| Batch 32 | 5 | 12.45028 ms | 12.42058 ms | **+0.24%** |
| 31-case total | 31 | 25.80421 ms | 25.64165 ms | **+0.66%** |

All five aggregate samples improved: `+0.2539%, +0.6001%, +0.6561%,
+0.6777%, +0.7051%`. Every per-shape median improved. The smallest gain was
`+0.091%` on B32/64K, while short and low-batch shapes improved by several
percent.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`f45a4febc31482c2129d9f639512a7f46415b612ae94e025efaccb259981ce20`.
