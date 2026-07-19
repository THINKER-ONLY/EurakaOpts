# MLA Decode v010: Materialized Weights

## Objective

Share online-softmax work across v009's four output partitions after QK logits
have already been materialized.

## Change

- extend the materialized QK kernel to compute split LSE and output part 0;
- normalize each invocation's raw logits once in a second 2D-grid kernel;
- overwrite the temporary logits tensor with FP16 softmax weights;
- let output parts 1-3 consume the shared normalized weights directly for PV;
- retain v009's direct path for batch 1 and batch 2 below 8K;
- retain the adaptive context split policy and partitioned 64-thread reduction.

Compared with v009, the materialized path performs online softmax once instead
of four times and moves one of four PV partitions into the QK kernel. The
largest official shape still uses a 64 MiB invocation-local temporary tensor.
It is overwritten and recomputed for every call, never cached across calls.

## Correctness

- The standard direct-path two-input FP32 checks retain maximum absolute
  errors `0.00038418` and `0.00028522`.
- Independent materialized-path B4/L512 checks on two changed input sets have
  maximum absolute errors `0.00028920` and `0.00064996`.
- All 31 official shapes match v009 with `atol=1e-2, rtol=1e-2`; the largest
  observed v009/v010 absolute difference is `0.00048828125`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes logits and weights, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape/family | Median paired improvement over v009 |
| --- | ---: |
| B2, context 8K / 16K / 32K / 64K | **+13.79% / +12.91% / +16.56% / +17.72%** |
| B4, context 2K / 8K / 16K / 32K / 64K | **+1.47% / +15.97% / +16.75% / +18.06% / +18.61%** |
| B8, context 2K / 8K / 16K / 32K / 64K | **+13.16% / +16.96% / +18.11% / +18.64% / +18.90%** |
| B16, context 2K / 8K / 16K / 32K / 64K | **+15.71% / +18.32% / +18.50% / +18.77% / +18.89%** |
| B32, context 2K / 8K / 16K / 32K / 64K | **+16.92% / +18.62% / +11.20% / +15.40% / +18.21%** |
| 31-case total | **35.3005 ms -> 29.3956 ms (+16.769%)** |

All five aggregate samples improved: `+16.4385%, +16.7466%, +16.7852%,
+16.7690%, +16.7710%`. Every per-shape median improved. Combined with v009,
the accepted same-machine totals moved from v008's `47.2302 ms` to
`29.3956 ms`, about `37.76%` lower.

An earlier standalone LSE/normalization design was rejected because the MACA
layout pass could not infer reduction-fragment layouts. Computing the real
part-0 PV result in the QK kernel provides a useful layout anchor and avoids a
dummy operation.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`b02da1aba474f46d4611ad4539e9b040e97bb4aa1839457019128291ed9a6eab`.
