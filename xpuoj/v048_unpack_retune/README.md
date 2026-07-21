# XPUOJ v048: Unpack Retune

## Objective

Retune the final routed copy for v047's `M176` workspaces without changing any
GEMM, stream, graph, or complete-recomputation behavior.

## Change

The E16 unpack changes from `BN256,T256` to `BN128,T512`. The E32 two-chunk
unpack retains `BN256` and increases from 256 to 1024 threads. Isolated kernel
medians changed as follows:

| Case | v047 | v048 | Kernel improvement |
| --- | ---: | ---: | ---: |
| E16 | 0.03160 ms | 0.02027 ms | +35.86% |
| E32 | 0.10568 ms | 0.09799 ms | +7.28% |

E64 also preferred 1024 threads in isolation, but the complete graph regressed
by approximately 0.02%, so v048 deliberately retains E64's `BN256,T256`
kernel. Packing, FC1, SwiGLU, down projection, route multiplication, and every
output write still execute on each invocation.

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
for both distributions (`max_abs=0`). Constant-data outputs are bit-identical
to v047, including every padded row.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each load direction used ten warmups, thirty calls per timing
slot, and twenty-four symmetric same-output samples in a fresh process.

Multi-stream module loading affects E32 by about 5%, so the two runtime ratios
were combined with their geometric mean:

| Case | v047-first / v048-second | v048-first / v047-second | Corrected improvement |
| --- | ---: | ---: | ---: |
| case1 | 1.93698 / 1.92587 ms | 1.92468 / 1.93509 ms | **+0.556%** |
| case2 | 3.17383 / 3.32163 ms | 3.15821 / 3.32403 ms | **+0.282%** |
| case3 | unchanged from v047 | unchanged from v047 | **0.000%** |
| total | 11.52917 ms corrected | 11.50924 ms corrected | **+0.173%** |

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`772d61255b87facc1491d0929a076c3e0eb3d12197d90383135316d7313b8f3d`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
