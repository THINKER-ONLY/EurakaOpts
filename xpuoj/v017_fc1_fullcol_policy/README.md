# XPUOJ v017: FC1 FullCol Warp Policy

## Objective

Test whether assigning more C500 warps to FC1's N dimension improves gate/up
weight-side work across sparse and dense official dimension paths.

## Parent Evidence

`v013_sparse_bm32_single_grid` uses TileLang's default `Square` policy. With
256 threads and C500's 64-thread warp, FC1 partitions four warps as `2x2`
along M/N. `FullCol` instead requests `1x4`.

## Single Change

Set `policy=T.GemmWarpPolicy.FullCol` on both FC1 GEMMs. FC2, tile sizes,
threads, pipeline stages, swizzle, arithmetic, routing, and workspace remain
unchanged. The change applies to every shape so the experiment covers more
than the local sparse proxy.

## Correctness

The candidate compiled successfully for all three tested specializations. Its
valid and padded outputs were bit-identical to v013 on each proxy
(`max_abs=0`, `mean_abs=0`).

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each proxy used five warmups, ten measured calls per sample,
and seven alternating paired samples:

| Proxy | v013 median | v017 median | Median paired change |
| --- | ---: | ---: | ---: |
| BM32 sparse `(7168,2048,64,1024)` | 9.8522 ms | 11.6355 ms | **-18.10%** |
| BM128 wide `(7168,2048,8,1024)` | 3.4804 ms | 5.0718 ms | **-45.72%** |
| BM128 narrow `(2048,8192,1,512)` | 2.4295 ms | 3.2116 ms | **-32.25%** |

All 21 paired samples regressed. Splitting the N dimension across all four
warps increases the cost of the M-side input fragments and produces a much
less favorable FC1 mapping than the default balanced partition.

Decision: **rejected**. Retain the default Square (`2x2`) FC1 warp policy.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
