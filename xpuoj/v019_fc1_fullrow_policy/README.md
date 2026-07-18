# XPUOJ v019: FC1 FullRow Warp Policy

## Objective

Tune the FC1 gate and up GEMM warp mapping on the three exact XPUOJ shapes
published by the SPJ.

## Single Change

Set `T.GemmWarpPolicy.FullRow` on both FC1 GEMMs. FC2 remains on TileLang's
default Square policy. Tile sizes, thread count, pipeline stages, swizzles,
routing, arithmetic, and workspace management are unchanged from v008.

## Scout Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each case used three warmups, five measured calls per sample,
and two alternating paired samples.

| Case | v008 | v019 | Paired improvement |
| --- | ---: | ---: | ---: |
| case1: 2048x8192, 16 experts, 2272 rows | 9.3821 ms | 7.3367 ms | +21.80% |
| case2: 7168x2048, 32 experts, 4544 rows | 15.4981 ms | 11.7632 ms | +24.10% |
| case3: 7168x2048, 64 experts, 9088 rows | 29.6729 ms | 22.7168 ms | +23.44% |

All candidate outputs were bit-identical to v008. The explicit Square control
was neutral, while FullCol regressed by 50.46%, confirming that the result is
specific to the FC1 warp orientation.

Decision: **accepted for full local verification and XPUOJ submission**.

## Full Local Verification

The final paired run used five warmups, ten measured calls per sample, and five
alternating samples on every official case:

| Case | v008 median | v019 median | Median improvement |
| --- | ---: | ---: | ---: |
| case1 | 9.3704 ms | 7.3380 ms | +21.68% |
| case2 | 15.5013 ms | 11.7722 ms | +24.08% |
| case3 | 29.6804 ms | 22.6998 ms | +23.54% |
| total | 54.5521 ms | 41.8100 ms | **+23.35%** |

All 15 per-case paired samples improved. Candidate and baseline outputs were
bit-identical on every benchmark case, and both FP32 oracle checks matched the
same tolerance as v008.

Final local decision: **accepted**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
