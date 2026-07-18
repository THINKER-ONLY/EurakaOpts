# XPUOJ v021: Fast Math And Fused FC1 Epilogue

## Objective

Reduce the non-GEMM FC1 overhead after v020 established FullRow for all three
GEMMs.

## Changes

This version combines two independent, individually positive changes:

1. Enable `TL_ENABLE_FAST_MATH` while retaining disabled warp specialization.
2. Fuse SiLU, gate/up multiplication, valid-row masking, and the `up_logits`
   write into one FC1 epilogue loop.

GEMM policies, tile sizes, thread counts, pipeline stages, swizzles, routing,
and workspace management are unchanged from v020.

## Correctness

- The standalone randomized check passes for counts `[142, 65, 128]`.
- All three official-shape proxies are bit-identical to v020 on valid and
  padded rows.
- Both FP32 oracle checks match the same `atol=1e-2, rtol=1e-2` tolerance as
  v020.

## Stable Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run used five warmups, ten measured calls per
sample, and five alternating paired samples:

| Case | v020 median | v021 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 6.2984 ms | 6.2539 ms | +0.73% |
| case2 | 10.4203 ms | 10.3916 ms | +0.27% |
| case3 | 20.0684 ms | 20.0198 ms | +0.29% |
| total | 36.7871 ms | 36.6654 ms | **+0.36%** |

All five aggregate pairs improved. A longer follow-up run was contaminated by
96% physical-GPU activity from another sGPU slice and is excluded from the
aggregate; its uncontended case2 pairs repeated +0.25% and +0.28%.

Decision: **accepted as the local baseline, not submitted online**. The gain is
consistent but below the threshold for spending an XPUOJ evaluation.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
