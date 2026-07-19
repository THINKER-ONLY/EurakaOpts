# MLA Decode v027: B32/8K Fused Split 32

## Objective

Make QK plus output-part-0 fusion profitable for B32/8K by restoring enough
split parallelism to offset the fused accumulator's register pressure.

## Change

- enable v026's QK-part0 fusion for B32/8K;
- increase only B32/8K from 16 to 32 sequence splits;
- retain v026's B32/16K-64K fusion, PV pipeline, and all other policies.

The fused QK kernel computes the first 128 output columns from its resident KV
and score tiles, reducing the following PV grid from four parts to three. The
extra splits are required to recover occupancy; fusion at 4, 8, or 16 splits
was rejected.

## Correctness

- All 31 official-shape outputs match v026 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00024414`.
- The independent B32/8K FP32-reference check passes with maximum absolute
  error `0.00017618`.
- Every invocation reads current inputs and writes the complete caller-provided
  output; only compiled kernels are cached.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted full run used eight warmups, ten
batched calls per module/output combination, twelve samples, both output
addresses, and a four-phase Latin rotation over all 31 official shapes:

| Shape group | Cases | Improvement |
| --- | ---: | ---: |
| Fused B32/8K | 1 | **+3.11%** |
| 31-case total | 31 | **+0.13%** |

The three independently phase-balanced full-suite samples improved by
`+0.1141%`, `+0.1297%`, and `+0.1457%`. Focused alternatives at split 4, 8,
and 64 regressed by approximately `42.7%`, `21.4%`, and `12.7%`, confirming
split 32 as the local optimum.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`e6c0d0072f930b4f8c872188d2039bb196d0e318c9b2f7de7d78fbde4015279a`.
