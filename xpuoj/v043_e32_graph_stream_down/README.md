# XPUOJ v043: E32 Graph Stream Down

## Objective

Extend v042's graph-stabilized down-projection overlap to E32 without changing
the E16 or E64 mathematical paths.

## Change

E32 now runs its two independent 16-expert down BMMs on two CUDA streams. The
complete E32 invocation is captured on the second call and replayed after
that, so per-call stream context and join overhead does not erase the BMM
overlap. CUDA streams are cached separately by device and expert count; E32
and E64 therefore do not share mcBLAS stream state.

The graph key still covers every argument's data pointer, shape, stride, and
dtype. Each replay executes pack, fused FC1, SwiGLU, both down BMMs, route
multiplication, and unpack against the current tensor contents. No activation,
down result, or completed output is reused.

## Correctness

- The official E32-distribution random test passed against the FP32 oracle.
- Maximum absolute error was `0.001953125`; mean absolute error was
  `0.0001458862`, identical to v042.
- All three constant-data proxies are bit-identical to v042.
- Peak paired allocation was 15.21 GiB, below the 32 GiB device quota.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Symmetric same-output timing used ten warmups and thirty calls
per direction.

Two multi-stream modules in one process exhibit an approximately 1% module
initialization-order bias that timer-slot swapping cannot remove. The final
comparison therefore ran in two fresh processes with v042/v043 load order
reversed and used the geometric mean of the two runtime ratios:

| Case | Samples per direction | Bidirectional improvement |
| --- | ---: | ---: |
| case1 | 12 + 12 | -0.01% (neutral) |
| case2 | 12 + 12 | **+1.21%** |
| case3 | 12 + 12 | +0.02% (neutral) |
| total | 12 + 12 | **+0.35%** |

A focused Case2 confirmation used twenty-four samples in each load direction
and measured a bidirectionally corrected improvement of **+1.44%**. Every
paired Case2 sample favored v043 in both directional runs.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`5c512627b7a7c734461d8f015bb448fb0b330cb0b6ee08518da7e3cb6b3aa640`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
