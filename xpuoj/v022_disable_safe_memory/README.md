# XPUOJ v022: Disable Safe Memory Legalization

## Objective

Remove compiler-generated bounds guards that are redundant for the three
fixed, fully divisible XPUOJ shapes.

## Single Change

Add `TL_DISABLE_SAFE_MEMORY_ACCESS=True` to v021's TileLang pass configuration.
The official hidden and intermediate dimensions are divisible by every active
BK/BN tile, expert IDs come from the valid routing map, and token accesses use
the provided padded storage. Kernel math and scheduling are otherwise
unchanged.

## Correctness

- The standalone randomized check passes for counts `[142, 65, 128]`.
- All three official-shape proxies are bit-identical to v021 on valid and
  padded rows.
- Both FP32 oracle checks match the same `atol=1e-2, rtol=1e-2` tolerance as
  v021.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The final run started with physical GPU utilization at 0% and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v021 median | v022 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 6.2464 ms | 6.2508 ms | -0.07% |
| case2 | 10.3384 ms | 10.0894 ms | +2.42% |
| case3 | 19.9339 ms | 19.5949 ms | +1.68% |
| total | 36.5186 ms | 35.9351 ms | **+1.61%** |

All five aggregate pairs improved. Additional isolated runs measured +2.40%
for case2 and +1.64% for case3, each with all five pairs positive.

Decision: **accepted as the local baseline, not yet submitted online**. The
gain is real but below the standalone large-improvement threshold.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
