# MLA Decode v016: Cached Split LSE

## Objective

Remove repeated global-memory reads of the split LSE from the materialized
output-part kernels.

## Change

- load the 16 FP16 LSE values for the current head tile and split into a
  fragment before entering the score-tile loop;
- use the cached fragment when scaling each saved prefix-relative score tile;
- retain the original FP16 subtraction and `exp2` path, so the numerical
  behavior is unchanged;
- retain v015's QK clear-accum scheduling, fused 576-D QK, shared KV part 0,
  prefix-scaled weights, split policy, and final reduction.

The generated v015 PV kernel loaded the same `glse` element inside every
32-token iteration and repeated that work across three output parts. The LSE
is invariant for the complete head-tile/split CTA, so one pre-loop load is
sufficient.

## Correctness

- All 31 official outputs are bitwise identical to v015.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v015 total | v016 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Materialized path | 24 | 24.09564 ms | 23.97755 ms | **+0.49%** |
| 31-case total | 31 | 25.53129 ms | 25.39584 ms | **+0.57%** |

All five aggregate samples improved: `+0.2691%, +0.5867%, +0.5742%,
+0.5172%, +0.6021%`. B4/2K improved by `+5.30%`; B32/64K improved by
`+0.09%`. The only negative full-run median was `-0.11%` on B4/16K, which a
separate seven-sample repeat measured at `+0.96%` with every sample positive.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`16c9f6b017d573f84d010621f879992d11fced4f97c5d0c973b9458cd32a49a2`.
