# MLA Decode v004: Batch16 Split8

## Objective

Cover the batch 2, 4, and 16 families in the published 31-case race set and
retune context splitting where the v003 proxy set had no measurements.

## Change

- batch 1 continues to use split 16;
- batch 2 through 16 now use split 8;
- batch 32 and larger continue to use split 4.

The only behavior change from v003 is batch 16, which moves from split 4 to
split 8. The output-quarter kernel and all numerical operations are unchanged.
In the rejected search points, split16 was neutral for batch2, split4 regressed
batch8 by about 31%, and split16 was slower than split8 for batch16 at 8K.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00038418` and `0.00028522`.
- All eleven batch/context proxies match v003 with `atol=1e-2, rtol=1e-2`.
- The largest observed v003/v004 difference is `0.000244140625`; unchanged
  batch families are bitwise identical.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples across eleven proxies:

| Case | v003 median | v004 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| batch 16, context 8192 | 0.9331 ms | 0.7964 ms | **+14.68%** |
| batch 16, context 65536 | 7.1099 ms | 5.9927 ms | **+15.65%** |
| eleven-case proxy total | 30.6753 ms | 29.3825 ms | **+4.19%** |

All five aggregate samples improved: `+4.1789%, +4.1783%, +4.1851%,
+4.2360%, +4.2633%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`8c50da4850359b6c63f1ab2f4d657c6aba29848485597101c9d93a288419369e`.
