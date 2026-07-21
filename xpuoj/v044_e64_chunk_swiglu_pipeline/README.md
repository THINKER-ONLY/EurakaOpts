# XPUOJ v044: E64 Chunk SwiGLU Pipeline

## Objective

Overlap E64 activation work with down projection while retaining v043's fast
full-batch fused FC1 and graph-stabilized two-stream execution.

## Change

E64 still computes gate/up with one batch64, `N=4096` BMM. Instead of running
one full-batch SwiGLU kernel on the default stream, each down worker executes a
16-expert SwiGLU kernel immediately before its corresponding down BMM. This
forms two independent `SwiGLU -> down` pipelines inside the captured graph and
allows activation work on one stream to overlap down projection on the other.

E16 and E32 remain equivalent to v043. Every graph replay still packs current
input, runs FC1 and SwiGLU, computes every down projection, applies current
route weights, and writes the output. No computed tensor result is cached.

## Correctness

- The official E64-distribution random test passed against the FP32 oracle.
- Maximum absolute error was `0.00201416015625`; mean absolute error was
  `0.0001427173`, identical to v043.
- All three constant-data proxies are bit-identical to v043.
- Peak paired allocation was 15.08 GiB, below the 32 GiB device quota.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Symmetric same-output timing used ten warmups, thirty calls per
direction, and twenty-four samples in each module-load direction.

As documented for v043, multi-stream modules have a module initialization-order
bias. v043-first/v044-second measured 6.40916/6.44573 ms; the reversed process
measured v044-first/v043-second at 6.39795/6.46675 ms. The geometric mean of
the two v044/v043 ratios removes that bias:

| Scope | Bidirectional improvement |
| --- | ---: |
| case1 | unchanged |
| case2 | unchanged |
| case3 | **+0.250%** |
| three-case weighted total | **+0.138%** |

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`bdf3c2391c3abf1675e9ee8d4afbd404c99831686f3e4bfcc20f6d787f103e34`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
