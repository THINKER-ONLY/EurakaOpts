# XPUOJ v014: Sparse BM16 Tier

## Objective

Test whether the ultra-sparse wide-hidden path benefits from reducing v013's
BM32 compute tile to BM16 when experts average at most 16 valid rows.

## Parent Evidence

`v013_sparse_bm32_single_grid` reduces the 256-expert, 4096-row case2 proxy
by 11.96% relative to v012. That input averages exactly 16 rows per expert, so
a BM16 tier could remove the remaining row padding inside each active BM32
tile.

## Single Change

Before v013's BM32 tier, select BM16 for `(7168, 2048)` when:

```text
total_valid_tokens <= num_experts * 16
```

Each 128-row metadata block then maps to eight subblocks in the existing FC1
and FC2 grids. Empty subblocks retain the block-uniform `actual_rows > 0`
guard. All other specializations remain unchanged.

## Correctness

The candidate compiled successfully on MetaX C500. On a random-routing
64-expert proxy with 1024 valid rows, its valid and padded output was
bit-identical to v013 (`max_abs=0`, `mean_abs=0`).

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The 64-expert proxy used five warmups, ten measured calls per
sample, and seven alternating paired samples:

| Metric | v013 BM32 | v014 BM16 |
| --- | ---: | ---: |
| Median time | 9.8614 ms | 12.2069 ms |

Paired changes were `-23.6864%, -23.7846%, -23.9707%, -23.9559%, -23.8256%,
-23.6534%, -23.6655%`. The median is **-23.7846%**, and all seven pairs
regressed.

BM16 reduces active row arithmetic but doubles the subblock grid relative to
BM32. On this hardware, extra CTA scheduling and the less favorable GEMM tile
outweigh the removed padding.

## XPUOJ Result

```text
Status:          Accepted
Total score:     61.33
Displayed time:  not provided
```

The score is unchanged from v008, v012, and v013 despite the large local proxy
regression. The judge score is therefore not sensitive enough to overturn the
direct paired C500 measurement.

Decision: **rejected as a local development baseline, neutral on XPUOJ**.
Retain BM32 as the smallest sparse M tile.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
