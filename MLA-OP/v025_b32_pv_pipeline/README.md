# MLA Decode v025: B32 PV Pipeline

## Objective

Overlap materialized-score and value-tile loads with the current PV GEMM on
the dominant batch-32 long-context shapes.

## Change

- enable one-stage software pipelining in the materialized PV loop for B32 at
  8K context and above;
- add a separate `16x32` FP16 shared buffer for raw score prefetches;
- keep the existing shared buffer for scaled scores consumed by the PV GEMM,
  avoiding overlapping writes that the TileLang pipeline planner rejects;
- retain v024's algorithms, numerical expressions, split policy, and all
  non-B32 kernels unchanged.

The new raw-score buffer costs 1 KiB of shared memory. It lets pipeline
planning overlap the next score/value loads with current-tile scaling and
matrix multiplication without changing score rounding or output order.

## Correctness

- All 31 official-shape outputs are bitwise identical to v024.
- Independent FP32-reference checks pass for B32 at 8K, 16K, 32K, and 64K.
- Their maximum absolute errors are respectively `0.00021450`, `0.00019714`,
  `0.00018205`, and `0.00014770`, below the official `rtol=2e-3, atol=2e-3`
  tolerance.
- Every invocation reads current inputs and writes the complete caller-provided
  output; only compiled kernels are cached.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted full run used six warmups, ten batched
calls per module/output combination, eight samples, both output addresses, and
a four-phase Latin rotation over all 31 official shapes:

| Shape group | Cases | v024 total | v025 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Pipelined B32 8K-64K | 4 | 8.47532 ms | 8.15223 ms | **+3.83%** |
| 31-case total | 31 | 18.11300 ms | 17.81236 ms | **+1.67%** |

The two phase-balanced full-suite samples improved by `+1.6677%` and
`+1.6777%`. A higher-sample run over the four changed shapes produced three
balanced improvements of `+3.8053%`, `+3.8296%`, and `+3.8464%`; every changed
shape improved and all outputs were bitwise identical.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`c3daa366a287c1a18f8ae905fea293a2a74ae0942f2840c81d6930cc84829fdf`.
