# MLA Decode v026: B32 QK Part 0

## Objective

Reuse the QK kernel's resident KV tile and materialized scores for the first
output partition on B32 shapes where v023's higher split counts had disabled
the older fusion condition.

## Change

- enable the existing QK plus output-part-0 path for B32/16K and B32/32K;
- retain that path for B32/64K and keep B32/8K on the separate four-part PV
  grid, where fusion regresses;
- reduce the following PV grid from four output partitions to three on the two
  changed shapes;
- retain v025's one-stage B32 PV pipeline and all numerical expressions.

The QK kernel already holds the complete KV tile and scaled scores. Computing
the first 128 output columns there avoids another score read, value read, and
PV grid partition at the cost of a `16x128` FP32 accumulator.

## Correctness

- All 31 official-shape outputs match v025 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00018311`.
- Independent FP32-reference checks pass for B32/16K and B32/32K with maximum
  absolute errors `0.00017810` and `0.00018205`.
- Every invocation reads current inputs and writes the complete caller-provided
  output; only compiled kernels are cached.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted run used six warmups, ten batched calls
per module/output combination, eight samples, both output addresses, and a
four-phase Latin rotation over all 31 official shapes:

| Shape group | Cases | v025 total | v026 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Fused B32/16K and B32/32K | 2 | 3.42693 ms | 3.29809 ms | **+3.76%** |
| 31-case total | 31 | 17.83884 ms | 17.68751 ms | **+0.86%** |

The two phase-balanced full-suite samples improved by `+0.8516%` and
`+0.8699%`. A twelve-sample focused run measured approximately `+2.85%` and
`+4.21%` on B32/16K and B32/32K. B32/8K fusion was rejected after a stable
`4.2%` regression.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`5b1e77d78632c5eeef3cdf36d21509877eb4cb8510a964de8323e23769699547`.
