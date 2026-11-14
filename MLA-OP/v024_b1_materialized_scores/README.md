# MLA Decode v024: B1 Materialized Scores

## Objective

Reuse attention scores across output partitions for low-batch long-context
decoding after v023's higher split counts restored enough C500 occupancy.

## Change

- enable the existing materialized-score path for batch 1 at 8K context and
  above;
- increase `B1/8K` from 16 to 64 splits;
- retain v023's `B1/16K=64` and `B1/32K=B1/64K=128` split choices;
- retain every kernel body, numerical expression, and all non-B1 policies
  unchanged.

The materialized path computes QK and prefix softmax metadata once, then reuses
the stored scores for the output partitions instead of recomputing QK for each
partition. The higher split counts make this path profitable for batch 1.

## Correctness

- All 31 official-shape outputs match v023 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00013733`.
- Independent FP32-reference checks pass for B1 at 8K, 16K, 32K, and 64K.
- Their maximum absolute errors are respectively `0.00009790`, `0.00010538`,
  `0.00012591`, and `0.00008710`.
- Every invocation reads current inputs and writes the complete caller-provided
  output; only compiled kernels are cached.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted run used six warmups, ten batched calls
per module/output combination, eight samples, both output addresses, and a
four-phase Latin rotation over all 31 official shapes:

| Shape group | Cases | v023 total | v024 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Changed B1 8K-64K | 4 | 0.87439 ms | 0.51957 ms | **+40.58%** |
| 31-case total | 31 | 18.53775 ms | 18.18050 ms | **+1.94%** |

The two independently phase-balanced full-suite samples improved by `+1.9063%`
and `+1.9744%`. Per-case improvements for B1 at 8K, 16K, 32K, and 64K were
approximately `+33.7%`, `+37.7%`, `+39.3%`, and `+43.9%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`aec79b54f62cd0cd8dc6d18b3243691d949ba87a8b691cb9b885eddd61a13f06`.
