# MLA Decode v011: Adaptive Normalization Threads

## Objective

Reduce launch and wave overhead in v010's logits-normalization kernel when each
split contains only a small number of tokens.

## Change

- use 64 threads for normalization when `seqlen_kv / num_split <= 256`;
- retain 128 threads for larger per-split chunks;
- retain v010's QK+part0 kernel, normalized-weight materialization, remaining
  PV partitions, direct-path threshold, split policy, and final reduction.

The selected normalization kernel has no GEMM and processes one 16x32 score
tile per loop iteration. A 64-thread CTA uses one C500 wave for short chunks;
larger chunks retain two waves for throughput.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- All 31 official outputs are bitwise identical to v010.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes logits and weights, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted repeat used five warmups, ten
alternating measured calls per sample, and five samples over all 31 official
shapes:

| Selected shape | Median paired improvement over v010 |
| --- | ---: |
| B2, context 8K / 16K | **+1.86% / +0.86%** |
| B4, context 2K / 8K | **+3.00% / +0.97%** |
| B8, context 2K | **+2.16%** |
| B16, context 2K | **+0.78%** |
| 31-case total | **29.5205 ms -> 29.3917 ms (+0.446%)** |

All five aggregate samples improved: `+0.2620%, +0.4655%, +0.4013%,
+0.4456%, +0.4961%`. Every per-shape median improved. An earlier full run
measured `+0.375%` with all five aggregate samples positive, confirming the
same direction.

A global 64-thread policy was rejected: it improved B4/2K by about 5.5% but
regressed B32/64K by about 2.5%. The accepted per-split threshold preserves
long-chunk throughput.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`24758e2ac9c25d11cde197ea0c4c862f63d466302e11bf9f7caf9ceae8fb8f79`.
