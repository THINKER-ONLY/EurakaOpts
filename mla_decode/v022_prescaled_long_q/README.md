# MLA Decode v022: Prescaled Long Q

## Objective

Remove repeated softmax-scale multiplications from the two longest
materialized QK kernels.

## Change

- for `B16/64K` and `B32/64K`, multiply Q and Q_pe by the log2 softmax scale
  while loading their shared tile;
- let the QK GEMM directly produce scaled scores, then compute online-softmax
  exponentials from `score - max` without score or max multiplications;
- store already-scaled prefix maxima and split LSE values on those paths;
- require a per-split length of at least 8192 and exclude fused part-0 kernels;
- retain v021 exactly for the other 29 official shapes.

Scaling the 576-D query tile once replaces score scaling in every 32-position
QK iteration. Input tensors are not modified, and every invocation still reads
current inputs and writes the full caller-provided output.

## Correctness

- Both changed outputs match v021 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00036621`.
- An independent `B16/64K` check against the PyTorch FP32 reference passes
  with maximum absolute error `0.00032824` and mean absolute error
  `0.00002373`.
- The other 29 shapes are compile-time unchanged and bitwise identical to
  v021.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and seven samples over all 31 official shapes:

| Shape group | Cases | v021 total | v022 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Prescaled long QK | 2 | 8.85100 ms | 8.83338 ms | **+0.20%** |
| 31-case total | 31 | 24.56454 ms | 24.45202 ms | **+0.45%** |

All seven aggregate samples improved: `+0.3773%, +0.5278%, +0.4477%,
+0.5008%, +0.5069%, +0.4195%, +0.3991%`. The two changed-shape aggregate
improved in all seven samples by `+0.1359%` to `+0.2533%`.

Only the changed-family delta is attributed to this optimization. It accounts
for about `+0.072%` of the measured full total; the larger full-run difference
includes positive device-state drift in unchanged kernels.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`3b49dfeb1c04cc40ce9d1045e7fe0149885fba0a54acf7faf7950b09acb34d72`.
