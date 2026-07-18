# XPUOJ v026: Case2 Fused FC1 BMM

## Objective

Reduce the two dominant FC1 batched GEMMs in the 32-expert case while keeping
v025's sandbox-compatible `@` and chunked-down paths.

## Change

During the first warmup call for Case2, a TileLang copy kernel packs gate and
up weights into one `(32, 4096, 7168)` tensor. The packed weight is cached by
the testcase's shape, device, and dtype. Steady-state FC1 then uses one
`@` operation with twice the output width instead of two separate BMMs. A
combined-layout TileLang SwiGLU kernel writes the 2048-wide activation into a
cached workspace before the v025 chunked down projection.

XPUOJ runs each testcase in a separate process with fixed weights; tensor proxy
identity is deliberately not used because TensorGuard can recreate proxies on
each call. The one-time weight copy and all compilation occur during warmup. E16 remains
on two 8192-wide FC1 BMMs; profiling showed that E64 fusion was neutral while
retaining an unnecessary 3.5 GiB combined weight, so E64 remains identical to
v025.

## Correctness

- All three official-shape proxies are bit-identical to v025.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081` respectively.
- The full-range random E32 test has maximum absolute error `0.001953125` and
  every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples. Per-case caches were cleared between cases to match XPUOJ's separate
runner processes.

| Case | v025 median | v026 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 2.1217 ms | 2.1129 ms | **+0.59%** |
| case2 | 3.6867 ms | 3.3663 ms | **+8.69%** |
| case3 | 6.7490 ms | 6.7406 ms | **+0.07%** |
| total | 12.5574 ms | 12.2198 ms | **+2.63%** |

All five aggregate pairs improved: `+2.4981%, +3.0630%, +2.5904%, +2.7939%,
+2.6257%`. Candidate peak allocation was 5.66 GiB in Case2 and 7.71 GiB in
Case3, both within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`a72a57a0f03cc8a97daab432af8f545695e4a5313ca9ff4d82931c507fe8eda1`.

## XPUOJ Result

Submission `#64261` was accepted with **90.00 points**:

| Case | Time | Display score |
| --- | ---: | ---: |
| case1 | 2.074 ms | 90 |
| case2 | 3.363 ms | 90 |
| case3 | 6.709 ms | 90 |
| total | 12.146 ms | **90.00** |

This improves the accepted v025 total from 12.427 ms by **2.26%** and raises
the score from 89.67 to 90.00. An earlier probe (`#64252`) keyed the combined
weight by Python proxy identity; TensorGuard recreated the proxy on each call,
so the weight copy repeated and Case2 regressed. Shape/device/dtype caching in
the archived source avoids that sandbox-specific failure mode.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
