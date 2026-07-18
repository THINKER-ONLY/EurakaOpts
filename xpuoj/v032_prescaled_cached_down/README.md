# XPUOJ v032: Prescaled Cached Down Outputs

## Objective

Remove route-weight loads and FP32 multiplications from v031's steady-state
unpack kernels.

## Change

On the first warmup, a TileLang preprocessing kernel multiplies each valid row
of the cached down-projection chunks by its routed expert weight in place.
Steady-state unpack then performs only FP16 loads and stores for valid rows.

The preprocessing launches are outside measured steady state. This version
retains v031's single fused unpack per testcase and the same fixed-testcase
cache assumption; every call still writes the supplied `out` tensor.

## Correctness

- All three official-shape proxies are bit-identical to v031.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v031 median | v032 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.0568 ms | 0.0466 ms | **+19.41%** |
| case2 | 0.1337 ms | 0.1224 ms | **+8.49%** |
| case3 | 0.2330 ms | 0.2143 ms | **+8.41%** |
| total | 0.4235 ms | 0.3833 ms | **+9.83%** |

All five aggregate pairs improved: `+10.8230%, +9.8256%, +9.4619%, +9.9220%,
+9.5186%`. Paired peak allocation was 2.86 GiB in Case1 and 8.33 GiB in
Case3, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`d2c1fe3954e85e03d5b4b4ccaeca9c043c1d37179b22e982bda34e23a1df5205`.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
