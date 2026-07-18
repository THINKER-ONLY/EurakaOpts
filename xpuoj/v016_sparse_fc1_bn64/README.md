# XPUOJ v016: Sparse FC1 BN64

## Objective

Test whether a smaller FC1 output tile improves residency on v013's
register-constrained BM32 sparse path.

## Parent Evidence

`v013_sparse_bm32_single_grid` lowers sparse FC1 register use to 108 MT
registers with BM32/BN128/BK64. Reducing BN could shrink both accumulator
fragments and each gate/up weight tile.

## Single Change

For the BM32 specialization only, change `fc1_block_n` from 128 to 64. BM64,
BM128, FC2, BM, BK, threads, pipeline stages, swizzle, arithmetic, and metadata
handling remain unchanged.

## Correctness

The candidate compiled successfully on MetaX C500. On a random-routing
64-expert proxy with 1024 valid rows, valid and padded output was bit-identical
to v013 (`max_abs=0`, `mean_abs=0`).

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The proxy used five warmups, ten measured calls per sample,
and seven alternating paired samples:

| Metric | v013 FC1 BN128 | v016 FC1 BN64 |
| --- | ---: | ---: |
| Median time | 9.8659 ms | 11.3686 ms |

Paired changes were `-14.6001%, -15.4574%, -15.3389%, -14.6673%, -15.2206%,
-15.0259%, -15.2143%`. The median is **-15.2143%**, and all seven pairs
regressed.

Any residency benefit is outweighed by doubling the FC1 N-grid and its CTA
setup, input loads, and scheduling work.

Decision: **rejected**. Retain FC1 BN128 on the BM32 sparse path.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
