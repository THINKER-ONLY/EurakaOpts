# NSA v004: Single-Block Softmax

## Objective

Specialize the dominant `selected_blocks == 1` shape family in the published
109-case race set without changing the multi-block algorithm.

## Change

- allocate the previous-maximum, rescale, and accumulated-logsum fragments only
  when more than one sparse block is selected;
- for one selected block, compute the maximum, exponentials, and denominator
  directly instead of running the online cross-block softmax recurrence;
- initialize the single PV accumulation with `clear_accum=True`, removing a
  separate output-fragment clear and rescale;
- retain the v003 serial K/V shared-tile reuse and thread policy.

Every call reads its current Q, K, V, block indices, and output buffer. Only
compiled kernels are cached.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00159550` and `0.00134659`.
- All ten local performance proxies match v003 with `atol=1e-2, rtol=1e-2`;
  the observed v003/v004 output difference is exactly zero.
- The `S > 1` path retains the v003 online-softmax equations and passes the S4,
  S8, and 64K/S16 regression proxies.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`.

Five shape proxies taken from the published 109-case `origin/race` set used
five warmups, twenty alternating measured calls per sample, and five samples:

| Case | v003 median | v004 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| B8, L2048, D32, S1, BS16 | 0.0696 ms | 0.0665 ms | **+4.61%** |
| B4, L1024, D128, S1, BS16 | 0.0961 ms | 0.0948 ms | **+1.40%** |
| B1, L8192, D128, S1, BS16 | 0.1607 ms | 0.1598 ms | **+0.45%** |
| B4, L128, D64, S1, BS64 | 0.0390 ms | 0.0377 ms | **+3.45%** |
| B4, L1024, D64, S8, BS16 | 0.1215 ms | 0.1206 ms | +0.85% |
| proxy total | 0.4870 ms | 0.4794 ms | **+1.67%** |

All five aggregate samples improved: `+1.6672%, +1.4049%, +1.8477%,
+1.6872%, +1.4079%`.

A separate seven-sample regression run measured S1 improvements of `+3.22%`
at B1/L1024/D64/BS16 and `+2.43%` at B1/L16384/D64/BS16. The non-official
64K/S16 stress proxy was unchanged (`+0.02%` median), confirming no material
regression outside the specialized shape family.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`aed76e012a89e2555a5c6afd7211662d02313ae238701e71dd91cc60bd180d67`.
