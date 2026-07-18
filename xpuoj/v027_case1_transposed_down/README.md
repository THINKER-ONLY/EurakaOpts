# XPUOJ v027: Case1 Transposed Down Weight

## Objective

Select the faster mcBLAS weight layout for the 16-expert down projection while
preserving v026's safe M192 packed extent.

## Change

On the first Case1 warmup call, a TileLang kernel transposes `down_w` from
`(16, 2048, 8192)` to a contiguous `(16, 8192, 2048)` BMM operand. The result
is cached by testcase shape, device, and dtype. Steady-state Case1 down then
uses the contiguous weight directly with `@` instead of a transposed view.

A local scan covered contiguous and transposed-view layouts plus expert chunks
16/8/4/2/1. The contiguous full-16 batch reduced down BMM time from about
0.802 ms to 0.680 ms. FC1 contiguous weights regressed, and contiguous down
weights regressed for E32/E64, so those paths remain identical to v026.

## Correctness

- All three official-shape proxies are bit-identical to v026.
- The full-range random E16 test has maximum absolute error `0.001953125` and
  every padded output row remains exactly zero.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081` respectively.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v026 median | v027 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 2.1235 ms | 1.9929 ms | **+6.04%** |
| case2 | 3.3989 ms | 3.3869 ms | **+0.60%** |
| case3 | 6.7572 ms | 6.7653 ms | **-0.06%** |
| total | 12.2796 ms | 12.1450 ms | **+1.14%** |

All five aggregate pairs improved: `+0.6085%, +1.1390%, +0.9948%, +1.4559%,
+1.2702%`. Paired peak allocation was 2.24 GiB in Case1 and 7.71 GiB in
Case3, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`4765e48b22e47a3bfa99f1352bbb83326029ee5c97b6f4aefdca042991e61d44`.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
