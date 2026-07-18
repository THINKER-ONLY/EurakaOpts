# XPUOJ v012: Sparse BM64 Single Grid

## Objective

Reduce padded-row work for the wide-hidden `(hidden=7168,
intermediate=2048)` shape without reintroducing v001's separate full/tail
kernel launches.

## Parent Evidence

`v008_fc2_bk64` is the best XPUOJ result at 61.33 points and 43 ms. MXCC
reports 248/238 MT registers and `staticMaxWarps/PEU=2` for its FC1/FC2
kernels. On the local 256-expert sparse proxy, every expert has about 16 valid
rows but v008 computes a full BM128 tile.

## Single Change

For `(7168, 2048)` compilations with fewer than 64 valid rows per expert on
average, each 128-row metadata block is represented by two BM64 CTAs in the
same kernel grid:

```text
metadata_bx = bx // 2
subblock    = bx % 2
block_start = metadata_bx * 128 + subblock * 64
```

A block-uniform `actual_rows > 0` guard skips the entire GEMM body for empty
halves. Dense `(7168, 2048)` inputs and all other shapes preserve BM128.
There are still exactly two kernel launches per `run_kernel` call.

## Mechanism

The sparse 256-expert proxy has 4096 valid rows but 32768 rows after the
official 128-row per-expert padding rule. BM64 cuts active padded compute in
half when an expert has fewer than 64 rows. It also reduces generated-kernel
register pressure:

| Kernel | v008 MT registers | v012 MT registers | v008 max warps/PEU | v012 max warps/PEU |
| --- | ---: | ---: | ---: | ---: |
| FC1 | 248 | 148 | 2 | 3 |
| FC2 | 238 | 158 | 2 | 3 |

## Correctness

- `(hidden=7168, intermediate=2048)`, eight sparse experts with counts
  `17,15,16,18,14,20,13,15`: maximum absolute error `0.0009765625`, mean
  absolute error `4.981675374438055e-06` against the FP32 oracle.
- `(hidden=2048, intermediate=8192)`, one expert with 142 rows: maximum
  absolute error `0.0009765625`, mean absolute error
  `4.290532615414122e-06`.
- Candidate output is bit-identical to v008 on all three full proxy inputs,
  including zero-valued padded rows.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each sample uses 10 warmups and 20 measured iterations. The
physical GPU had activity on another slice, so baseline and candidate calls
were paired and alternated; the decision uses the median paired improvement.

| Proxy | v008 median | v012 median | Median paired change | Decision |
| --- | ---: | ---: | ---: | --- |
| case 1: `(2048,8192,8,4096)` | 14.8035 ms | 11.8093 ms | +8.30% | noisy/neutral; BM128 preserved |
| case 2: `(7168,2048,256,4096)` | 79.2765 ms | 64.5513 ms | **+17.84%** | improved; all five pairs positive |
| case 3: `(7168,2048,256,32768)` | 125.0715 ms | 118.6911 ms | +1.15% | neutral; BM128 preserved |
| aggregate | 219.1514 ms | 195.0517 ms | **+8.80%** | accepted locally |

Case 2 paired improvements were `17.84%, 23.02%, 13.46%, 8.62%, 21.67%`.
Case 1 and case 3 contained both positive and negative pairs and are treated as
neutral rather than evidence of a speedup.

Decision: **accepted as the new local C500 baseline**. An XPUOJ submission is
deferred until additional local strategies produce a clearly larger aggregate
gain.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
