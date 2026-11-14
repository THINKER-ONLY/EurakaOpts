# MLA Decode v009: Materialized Scores

## Objective

Eliminate the four-way recomputation of QK logits in v008's output-partitioned
main attention kernel for sufficiently parallel workloads.

## Change

- add a 2D-grid QK kernel that computes each head/split score tile once;
- materialize the current invocation's logits in a temporary FP16 tensor;
- let the four output partitions reuse those logits for online softmax and PV;
- use the materialized path for `batch >= 4` and for batch 2 with context at
  least 8K;
- retain v008's direct fused path for batch 1 and batch 2 below 8K;
- retain the adaptive context split policy and partitioned 64-thread reduction.

The materialized path reduces QK GEMMs and Q/K reads by 4x while keeping PV,
online softmax, and split-K reduction unchanged. The largest official shape
uses a 64 MiB temporary logits tensor. It is allocated and recomputed inside
every invocation; it is not cached across calls.

## Correctness

- The standard direct-path two-input FP32 checks retain maximum absolute
  errors `0.00038418` and `0.00028522`.
- Independent materialized-path B4/L512 checks on two changed input sets have
  maximum absolute errors `0.00044115` and `0.00058597`.
- All 31 official shapes match v008 with `atol=1e-2, rtol=1e-2`; the largest
  observed v008/v009 absolute difference is `0.0003662109375`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes temporary logits, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape/family | Median paired improvement over v008 |
| --- | ---: |
| B2, context 8K / 16K / 32K / 64K | **+22.09% / +21.50% / +23.16% / +24.35%** |
| B4, context 2K / 8K / 16K / 32K / 64K | **+20.89% / +20.64% / +23.65% / +24.16% / +24.83%** |
| B8, context 2K / 8K / 16K / 32K / 64K | **+18.37% / +23.98% / +24.52% / +25.06% / +25.43%** |
| B16, context 2K / 8K / 16K / 32K / 64K | **+21.62% / +24.92% / +25.49% / +25.65% / +25.98%** |
| B32, context 2K / 8K / 16K / 32K / 64K | **+24.93% / +26.64% / +27.18% / +27.67% / +27.56%** |
| 31-case total | **47.2302 ms -> 35.1692 ms (+25.534%)** |

All five aggregate samples improved: `+25.4518%, +25.5456%, +25.5628%,
+25.5010%, +25.5335%`. Every per-shape median improved. B1 and B2/2K stay
on the direct path; experiments that forced materialization there were rejected
after B1/8K and B1/64K regressions of about 42% and 65%.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`8c3e66429a44cc46b1d10b00e149b079d1c6f25c1298f61e18dfe8f13a4e650d`.
