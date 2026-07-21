# XPUOJ v041: E64 Fused FC1

## Objective

Reduce E64 FC1 launch and scheduling overhead while preserving v040's full
recomputation path and numerical behavior.

## Change

Extend the existing E32 fused-FC1 path to E64. Gate and up weights are
combined along the output dimension and executed as one `N=4096` batched
matrix multiplication; the existing combined SwiGLU kernel then consumes the
two halves. E16 remains on separate gate and up BMMs.

The combined weight is preprocessing state only. Every invocation still packs
the current input, executes FC1, SwiGLU, down projection, route multiplication,
and output unpack.

## Correctness

- The official E64-distribution random test passed against the FP32 oracle.
- Maximum absolute error was `0.00201416015625`; mean absolute error was
  `0.0001427173`.
- Peak allocation was 11.23 GiB, below the 32 GiB device quota.
- There is no activation, down-output, or completed-output result cache.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Symmetric same-output timing used ten warmups, thirty calls per
direction, and twenty-four samples.

Full comparison against v040:

| Case | Median paired improvement | Decision |
| --- | ---: | --- |
| case1 | -0.09% | neutral, unchanged code path |
| case2 | +0.11% | neutral, unchanged code path |
| case3 | **+0.88%** | accepted |
| total | **+0.48%** | accepted |

The total fell from 11.83959 ms to 11.78137 ms and all twenty-four aggregate
pairs improved. A focused case3 run measured `+0.80%`, with all twenty-four
pairs positive.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`0c216cd108db83a4b455f74bed9a2091b1345b3049f16484b6700931d0682b30`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
