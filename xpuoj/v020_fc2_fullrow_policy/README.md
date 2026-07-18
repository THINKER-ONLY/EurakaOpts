# XPUOJ v020: FC2 FullRow Warp Policy

## Objective

Tune FC2's GEMM warp mapping after v019 established FullRow for both FC1
GEMMs on the exact XPUOJ shapes.

## Single Change

Set `T.GemmWarpPolicy.FullRow` on the FC2 down projection GEMM. All three GEMMs
now use FullRow. Tile sizes, thread count, pipeline stages, swizzles, routing,
arithmetic, and workspace management are unchanged from v019.

## Scout Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each case used three warmups, five measured calls per sample,
and two alternating paired samples against v019.

| Case | v019 | v020 | Paired improvement |
| --- | ---: | ---: | ---: |
| case1: 2048x8192, 16 experts, 2272 rows | 7.3591 ms | 6.3004 ms | +14.39% |
| case2: 7168x2048, 32 experts, 4544 rows | 11.7749 ms | 10.4292 ms | +11.43% |
| case3: 7168x2048, 64 experts, 9088 rows | 22.6983 ms | 20.0615 ms | +11.62% |

All candidate outputs were bit-identical to v019. The explicit Square control
was neutral, while FullCol regressed by 33.49%, confirming that FullRow is the
correct orientation for the FC2 tile as well.

Decision: **accepted for full local verification**.

## Full Local Verification

The final paired run used five warmups, ten measured calls per sample, and five
alternating samples on every official case:

| Case | v019 median | v020 median | Median improvement |
| --- | ---: | ---: | ---: |
| case1 | 7.3396 ms | 6.3005 ms | +14.18% |
| case2 | 11.7729 ms | 10.4087 ms | +11.52% |
| case3 | 22.6932 ms | 20.0662 ms | +11.58% |
| total | 41.8057 ms | 36.7754 ms | **+12.03%** |

All 15 per-case paired samples improved. Candidate and baseline outputs were
bit-identical on every benchmark case, and both FP32 oracle checks matched the
same tolerance as v019. Relative to v008's 54.5521 ms local total, v020 is
32.59% faster.

Final local decision: **accepted**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
