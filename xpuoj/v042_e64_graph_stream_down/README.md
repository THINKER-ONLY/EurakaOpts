# XPUOJ v042: E64 Graph Stream Down

## Objective

Overlap the four independent E64 down-projection chunks and remove their
per-invocation host scheduling overhead while preserving v041's full
recomputation semantics.

## Change

E64 uses two cached CUDA streams. Each stream executes two consecutive
16-expert down BMMs, after which the default stream joins both workers and
runs the combined route-weight unpack.

The first call for a tensor set runs eagerly and initializes JIT and mcBLAS.
The second call captures the complete E64 invocation, including pack, fused
FC1, SwiGLU, all four down BMMs, route multiplication, and output unpack.
Later calls replay that graph. The key includes every argument's data pointer,
shape, stride, and dtype; a different input or output tensor gets an
independent eager/capture path. E16 and E32 remain on v041's eager path.

Graph replay is launch scheduling, not result reuse: every replay reads the
current activation and route weights and executes all mathematical stages.

## Correctness

- The official E64-distribution random test passed against the FP32 oracle.
- Maximum absolute error was `0.00201416015625`; mean absolute error was
  `0.0001427173`, identical to v041.
- After graph capture, changing activation and route-weight values in place
  produced output bit-identical to a fresh v041 eager execution (`max_abs=0`).
- Constant-data outputs for all three proxies are bit-identical to v041.
- Peak paired allocation was 14.96 GiB, below the 32 GiB device quota.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Symmetric same-output timing used ten warmups, thirty calls per
direction, and twenty-four samples.

Full comparison against v041:

| Case | v041 | v042 | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 1.94833 ms | 1.94636 ms | +0.07% (neutral) |
| case2 | 3.29736 ms | 3.30070 ms | -0.05% (neutral) |
| case3 | 6.54355 ms | 6.39913 ms | **+2.25%** |
| total | 11.78924 ms | 11.64618 ms | **+1.22%** |

All twenty-four aggregate pairs improved. The E64-only gain was independently
reproduced at `+2.13%` and `+2.21%` before the final run.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`374a33e494b3dd5c4273b2a289e4dc8219dc4f8827961a386db480f93962583d`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
