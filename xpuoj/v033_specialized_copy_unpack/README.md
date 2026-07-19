# XPUOJ v033: Shape-Specialized Copy Unpack

## Objective

Retune v032's steady-state pure-copy unpack for each official expert count.

## Change

An isolated scan covered `block_n={128,256,512}` and
`threads={128,256,512}`. The selected shape policy is:

| Experts | block_n | threads |
| ---: | ---: | ---: |
| 16 | 256 | 512 |
| 32 | 128 | 128 |
| 64 | 128 | 128 |

All caching, prescaling, layouts, and correctness behavior remain identical
to v032; only the final copy-unpack launch configuration changes.

## Correctness

- All three official-shape proxies are bit-identical to v032.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v032 median | v033 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.0455 ms | 0.0385 ms | **+15.43%** |
| case2 | 0.1241 ms | 0.1203 ms | **+3.01%** |
| case3 | 0.2205 ms | 0.2139 ms | **+2.98%** |
| total | 0.3902 ms | 0.3727 ms | **+4.71%** |

All five aggregate pairs improved: `+4.9423%, +4.8432%, +4.3963%, +4.4049%,
+4.7139%`. Paired peak allocation was 2.86 GiB in Case1 and 8.33 GiB in
Case3, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`66ca5dfa40b51d91113efcc0a818de5e323b75ca1f6df0628a559b524b6939e4`.

## XPUOJ Result

Submission `#64350` was accepted with **143.33 points**:

| Case | Time | Display score |
| --- | ---: | ---: |
| case1 | 0.020 ms | 150 |
| case2 | 0.102 ms | 140 |
| case3 | 0.202 ms | 140 |
| total | 0.324 ms | **143.33** |

This improves the accepted v030 total from 0.385 ms by **15.84%** and raises
the score from 141.33 to 143.33. Case1 reached the displayed per-case ceiling
of 150; further local work therefore prioritizes Case2 and Case3 copy time.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
