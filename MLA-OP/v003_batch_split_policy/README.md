# MLA Decode v003: Batch Split Policy

## Objective

Choose the context-split count from the amount of batch parallelism instead of
using split 8 for every shape.

## Change

- batch 1 uses split 16;
- batch 2 through 8 use split 8;
- larger batches use split 4.

The output-quarter kernel and all numerical operations are unchanged from
v002. Split 32 was slower than split 16 for batch 1, while split 2 regressed
batch 32 by about 16%, so neither is selected.

## Correctness

- Two changed-input checks at the same shape match the FP32 reference with
  maximum absolute errors `0.00038418` and `0.00028522`.
- All five proxies match v002 with `atol=1e-2, rtol=1e-2`.
- The largest observed v002/v003 difference is `0.000244140625`.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples:

| Case | v002 median | v003 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| batch 1, context 8192 | 0.1773 ms | 0.1040 ms | **+41.16%** |
| batch 8, context 8192 | 0.4852 ms | 0.4819 ms | +0.68% |
| batch 32, context 8192 | 1.5724 ms | 1.5558 ms | **+1.02%** |
| batch 1, context 65536 | 1.1785 ms | 0.6097 ms | **+48.26%** |
| batch 32, context 65536 | 12.3255 ms | 12.2177 ms | +0.88% |
| proxy total | 15.7389 ms | 14.9690 ms | **+4.87%** |

All five aggregate samples improved: `+4.8457%, +4.8552%, +4.9777%,
+4.8665%, +4.9810%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`7fffba657efb1815a727dc9d10428cf8792f2c59a8ba687425ef2464d7bd9065`.
