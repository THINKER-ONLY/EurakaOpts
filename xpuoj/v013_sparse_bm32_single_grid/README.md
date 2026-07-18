# XPUOJ v013: Sparse BM32 Tier

## Objective

Reduce padded-row work and register pressure further for the ultra-sparse
wide-hidden `(hidden=7168, intermediate=2048)` workload while preserving
v012's BM64 and BM128 paths.

## Parent Evidence

`v012_bm64_single_grid` reduced active sparse tiles from BM128 to BM64 and
improved the three-case local aggregate by 8.80% over v008. On the 256-expert,
4096-row proxy, however, experts average only 16 valid rows, so each active
BM64 CTA still computes four times as many rows as it writes.

## Single Change

Add a BM32 tier before v012's BM64 tier for `(7168, 2048)` compilations:

```text
average valid rows < 32: BM32, four subblocks per metadata block
average valid rows < 64: BM64, two subblocks per metadata block
otherwise:               BM128, one subblock per metadata block
```

The existing block-uniform `actual_rows > 0` guard skips empty subblocks. All
tiers remain in the same FC1 and FC2 grids, so `run_kernel` still launches
exactly two kernels.

## Mechanism

BM32 halves the active padded rows on the 16-row-per-expert proxy relative to
BM64. The smaller M fragment also lowers generated-kernel register pressure:

| Kernel | v012 BM64 MT registers | v013 BM32 MT registers | v012 max warps/PEU | v013 max warps/PEU |
| --- | ---: | ---: | ---: | ---: |
| FC1 | 148 | 108 | 3 | 4 |
| FC2 | 158 | 122 | 3 | 4 |

## Correctness And Scope

- Random inputs and weights with eight uneven expert tails
  `17,15,16,18,14,20,13,15` are bit-identical to v012.
- The generated MACA device source is byte-identical to v012 for the
  `(2048,8192)` path and for a dense `(7168,2048)` input with one expert and
  142 rows. This verifies that the BM32 tier does not perturb non-matching
  specializations.
- v012 already passed the FP32 oracle on both official dimension pairs; v013
  changes only the now bit-identical sparse specialization described above.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Baseline and candidate calls use the same allocations and are
alternated to reduce interference from activity on another slice of the
physical GPU.

The archived on-disk candidate was rerun on the full 256-expert sparse proxy
with 10 warmups, 20 measured iterations per call, and five paired samples:

```text
v012 median: 44.8295 ms
v013 median: 39.4686 ms
paired:      11.9608%, 11.9875%, 12.0246%, 11.9487%, 11.8804%
```

The median paired improvement is **+11.9608%**, with all five pairs positive
and a narrow 0.14 percentage-point range. Candidate output was bit-identical
to v012 in this run, and both official-dimension FP32 oracle checks passed. An
earlier run under heavier neighboring-slice interference had a +3.3472% median
with five of seven pairs positive; a smaller 64-expert sparse proxy gave a
**+9.1206%** median across seven pairs, all positive.

Case 1 and case 3 do not select BM32 and compile to byte-identical device code,
so their expected change is zero. The stable case-2 paired result is the
discriminating measurement.

Decision: **accepted as the new local C500 baseline and recommended for XPUOJ
measurement**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
