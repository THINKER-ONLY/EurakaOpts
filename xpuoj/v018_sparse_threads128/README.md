# XPUOJ v018: Sparse Threads128 Control

## Objective

Evaluate C500 threadblock size independently for v013's BM32 specialization.
This is retained as a control experiment after the XPUOJ SPJ configurations
showed that the specialization is not selected online.

## Parent Evidence

The automatic stage/thread search found that global 128-thread configurations
heavily regress BM128, but improve BM32. A stage-specific decomposition on the
64-expert sparse proxy measured:

| FC1 threads | FC2 threads | Median paired change |
| ---: | ---: | ---: |
| 128 | 128 | **+6.21%** |
| 128 | 256 | +1.79% |
| 256 | 128 | +4.41% |
| 256 | 256 | +0.22% (equivalent baseline) |

## Single Change

Select 128 threads when `routing_block_m == 32`; otherwise retain 256:

```python
threads = 128 if routing_block_m == 32 else 256
```

Both FC1 and FC2 use the selected count. BM/BN/BK, pipeline stage, warp policy,
swizzle, arithmetic, routing, and workspace remain unchanged.

## Correctness And Scope

- The 64-expert and full 256-expert sparse proxies are bit-identical to v013
  on valid and padded rows (`max_abs=0`, `mean_abs=0`).
- The full benchmark's two official-dimension FP32 oracle checks pass.
- BM64, BM128, and the `(2048,8192)` shape still select 256 threads.
- All three XPUOJ cases have 142 valid rows per expert and select BM128, so
  this version generates the same official kernels as v008.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The full 256-expert case2 proxy used ten warmups, twenty
measured calls per sample, and five alternating paired samples:

| Metric | v013 threads256 | v018 threads128 |
| --- | ---: | ---: |
| Median time | 39.4561 ms | 37.8551 ms |
| Median paired change | | **+4.0729%** |

Paired improvements were `4.1452%, 4.0488%, 4.0074%, 4.1234%, 4.0729%`.
All five pairs improved with a narrow 0.14 percentage-point range.

Decision: **rejected for official tuning, positive only on the non-official
sparse diagnostic load**. The change is functionally correct but cannot affect
any XPUOJ testcase, so it is not submitted online and v008 remains the official
baseline.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
