# XPUOJ v045: E32/E64 Full Stream Pipeline

## Objective

Remove the global FC1-to-down barrier for E32 and E64, and trim redundant
packing work without changing the complete-recomputation semantics of v044.

## Change

E32 and E64 now split their experts evenly between two cached CUDA streams.
Each stream owns its complete dependency chain:

```text
32/16-expert fused FC1 -> 16-expert SwiGLU -> 16-expert down BMM
```

E32 executes one chain per stream. E64 executes two consecutive SwiGLU/down
chunks after each stream's 32-expert FC1. This lets one half begin activation
and down projection without waiting for the other half's FC1 to finish.

The pack kernel changes from `BN256,T256` to `BN128,T512` and writes only valid
expert rows. Packed tail rows are never read by unpack and cannot contribute to
valid rows, so initializing them to zero was redundant. Every valid input row
is still copied on every invocation. E16 retains v044's GEMM path and receives
only this pack-kernel improvement.

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
bit-identical to v044. Peak paired allocation was 15.33 GiB.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each load direction used ten warmups, thirty calls per timing
slot, and twenty-four symmetric same-output samples.

Multi-stream module initialization creates a load-order bias. The raw medians
were therefore measured in two fresh processes and the two v045/v044 runtime
ratios were combined with their geometric mean:

| Case | v044-first / v045-second | v045-first / v044-second | Corrected improvement |
| --- | ---: | ---: | ---: |
| case1 | 1.94674 / 1.94004 ms | 1.95779 / 1.95985 ms | **+0.225%** |
| case2 | 3.25344 / 3.33981 ms | 3.17969 / 3.32313 ms | **+0.892%** |
| case3 | 6.41138 / 6.36995 ms | 6.35855 / 6.41595 ms | **+0.771%** |
| total | 11.61157 / 11.64980 ms | 11.49603 / 11.69892 ms | **+0.708%** |

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`15c4a608941edf2c2f283d17f4f68fcff6ac7fe1ebbaab5ef29d23ba6a5da92f`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
