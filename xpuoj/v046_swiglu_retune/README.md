# XPUOJ v046: SwiGLU Retune

## Objective

Retune the two SwiGLU kernels in v045 for the MetaX C500 while preserving its
complete-recomputation and full stream-pipeline semantics.

## Change

The standalone E16 SwiGLU kernel changes from `BN256,T256` to `BN128,T512`.
The 16-expert chunk kernel used by the E32/E64 pipelines changes from
`BN256,T256` to `BN64,T256`. Isolated kernel measurements showed approximately
4.8% and 25.5% lower latency, respectively.

No operation is removed: every invocation still repacks the current input and
recomputes FC1, SwiGLU, down projection, route multiplication, and output.

## Correctness

Random tests using all three disclosed expert distributions passed against the
FP32 oracle:

| Case | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| E16 | 0.001953125 | 0.0001468057 |
| E32 | 0.001953125 | 0.0001410794 |
| E64 | 0.002197265625 | 0.0001427663 |

After CUDA Graph capture, activation and route weights were modified in place
for E32 and E64. Graph replay and a fresh eager execution were bit-identical
for both distributions (`max_abs=0`). The three constant-data proxies are also
bit-identical to v045. Peak paired allocation was 15.27 GiB.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each load direction used ten warmups, thirty calls per timing
slot, and twenty-four symmetric same-output samples in each direction.

Multi-stream module initialization creates a load-order bias. The raw medians
were therefore measured in two fresh processes and the two v046/v045 runtime
ratios were combined with their geometric mean:

| Case | v045-first / v046-second | v046-first / v045-second | Corrected improvement |
| --- | ---: | ---: | ---: |
| case1 | 1.94045 / 1.93516 ms | 1.93890 / 1.94117 ms | **+0.195%** |
| case2 | 3.18244 / 3.33346 ms | 3.17531 / 3.34929 ms | **+0.348%** |
| case3 | 6.36981 / 6.34137 ms | 6.34878 / 6.35321 ms | **+0.258%** |
| total | 11.49270 / 11.61000 ms | 11.46299 / 11.64367 ms | **+0.273%** |

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`fe9d410123e2f21537db9ee508d416d77f2161f0ad1db2d0cfe49bce472eff04`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
