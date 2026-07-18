# XPUOJ v015: Sparse FC1 BK32

## Objective

Test whether reducing FC1's K tile lowers shared-memory pressure enough to
improve v013's BM32 wide-hidden sparse path.

## Parent Evidence

`v013_sparse_bm32_single_grid` is the local baseline. Its BM32 specialization
uses FC1 BM32/BN128/BK64 while iterating over a large hidden dimension of 7168.

## Single Change

For the BM32 specialization only, change `fc1_block_k` from 64 to 32. BM64,
BM128, FC2, grid geometry, threads, pipeline stages, swizzle, arithmetic, and
metadata handling remain unchanged.

## Correctness

The candidate compiled successfully on MetaX C500. On a random-routing
64-expert proxy with 1024 valid rows, valid and padded output was bit-identical
to v013 (`max_abs=0`, `mean_abs=0`).

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The proxy used five warmups, ten measured calls per sample,
and seven alternating paired samples:

| Metric | v013 FC1 BK64 | v015 FC1 BK32 |
| --- | ---: | ---: |
| Median time | 9.8087 ms | 10.6511 ms |

Paired changes were `-8.1264%, -8.8191%, -8.6391%, -8.6235%, -8.5686%,
-8.4378%, -8.7410%`. The median is **-8.6235%**, and all seven pairs
regressed.

Halving BK also doubles FC1's K-loop iterations. The additional copies,
synchronization, and loop overhead outweigh any resource benefit on this
specialization.

## XPUOJ Result

```text
Status:          Accepted
Total score:     61.33
Displayed time:  43.384 ms
Case scores:     62 / 61 / 61
Case times:      7.080 / 12.465 / 23.839 ms
```

The XPUOJ SPJ reports 142 valid rows per expert in every testcase. The BM32
path, and therefore this version's FC1 BK32 change, is not selected online.
The service-side submission source is byte-identical to this archived file.

Decision: **rejected as a local sparse strategy, neutral on XPUOJ**. Retain
FC1 BK64 and BM128 for the official configurations.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
