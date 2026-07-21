# XPUOJ v040: Combined Route Unpack

## Objective

Reduce launch overhead after each 16-expert down BMM while preserving v039's
full recomputation and exact route-weight multiplication order.

## Change

E32 and E64 retain their two or four down-BMM outputs until all chunks finish,
then use one 256-thread hidden-first TileLang kernel to unpack every chunk.
Each branch still multiplies the FP16 down output by the corresponding FP32
route weight before writing `out`. E16 remains byte-for-byte equivalent to
v039's single-chunk path.

The 1024-thread variant regressed by 0.7-1.0% and was rejected. Pre-scaling the
activation was also rejected because it changed FP16 rounding and failed the
random correctness test.

## Correctness

- All three constant-data proxies are bit-identical to v039.
- Full-range random E16/E32/E64 tests retain maximum absolute errors
  `0.001953125`, `0.001953125`, and `0.0020141602` against FP32.
- Every invocation recomputes current activations and down outputs; there is no
  completed-output or down-output result cache.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Symmetric same-output timing used ten warmups, thirty calls per
direction, and twenty-four samples.

Incremental comparison against v039:

| Case | v039 | v040 | Median paired improvement |
| --- | ---: | ---: | ---: |
| case2 | 3.31960 ms | 3.30877 ms | **+0.32%** |
| case3 | 6.63840 ms | 6.61067 ms | **+0.40%** |
| case2 + case3 | 9.95800 ms | 9.91944 ms | **+0.37%** |

All twenty-four combined pairs improved. A separate full comparison against
v027 measured case improvements of `+2.21%`, `+2.48%`, and `+2.15%`, with the
total falling from 12.14795 ms to 11.87961 ms (**+2.23%**).

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`ed54bc61ea76c4d9ddd66da6ce567f39bf6350bc714977c6febfd51e18db8fa8`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
