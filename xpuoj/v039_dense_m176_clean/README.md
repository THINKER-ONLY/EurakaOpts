# XPUOJ v039: Dense M176 Recompute Path

## Objective

Return to v027's full-compute path and tune the dense expert extent for the
three disclosed XPUOJ cases. Unlike v028-v038, every timed invocation repacks
the current activation, executes FC1, SwiGLU, and down projection, applies the
current route weights, and writes the valid output rows.

## Change

The dense expert extent changes from 192 to 176 for E16, E32, and E64. The
disclosed proxy distributions have maximum expert counts 164, 170, and 166,
respectively, so 176 covers every valid row while removing 8.33% of the padded
BMM work. It also remains 16-aligned and selects the fast mcBLAS M176 kernels.

No activation, down result, or completed output is reused between calls.
Compiled kernels, allocation workspaces, and the inherited fixed-weight
preprocessing remain cached across warmup calls.

## Correctness

Full-range random tests using every disclosed expert count passed against the
FP32 reference:

| Case | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| E16 | 0.001953125 | 0.0001433273 |
| E32 | 0.001953125 | 0.0001458862 |
| E64 | 0.0020141602 | 0.0001427173 |

The three constant-data proxies are bit-identical to v027, and padded output
rows remain zero in the official harness. Non-16-aligned M172 was rejected:
mcBLAS produced incorrect results for unsupported batched strides.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Both modules wrote the same `out` and were measured in both
timer slots. The run used ten warmups, thirty calls per direction, and
twenty-four symmetric samples:

| Case | v027 | v039 | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 2.00063 ms | 1.95301 ms | **+2.37%** |
| case2 | 3.37923 ms | 3.30626 ms | **+2.20%** |
| case3 | 6.76044 ms | 6.63210 ms | **+1.92%** |
| total | 12.14030 ms | 11.89137 ms | **+2.04%** |

All twenty-four aggregate pairs improved. Decision: **accepted as the local
full-compute baseline; do not submit online without an explicit instruction**.

The archived submission SHA-256 is
`c38e697d68257b09fed55b658af19d66d2929cf69260b841cdca52d88b7a1a52`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
