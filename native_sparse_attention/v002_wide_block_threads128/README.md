# NSA v002: Wide-Block 128-Thread Path

## Objective

Increase parallelism for block-size 32/64 attention without changing the
official 64-thread path used by smaller blocks.

## Change

Shapes with `D >= 64` and `block_size >= 32` use 128 threads. At 128 threads,
the direct FP32-fragment to FP16-fragment copy has incompatible TileLang MACA
layouts. The optimized path therefore writes the softmax scores to a small
FP16 shared tile before the PV GEMM. All other shapes retain the baseline
fragment path and 64 threads.

This version only caches compiled kernels. Every call reads the current Q, K,
V, block indices, and output buffer.

## Correctness

- The initial and changed-input checks at the same shape match the FP32 sparse
  reference with maximum absolute errors `0.00159550` and `0.00134659`.
- Every proxy output matches v001 with `atol=1e-2, rtol=1e-2`.
- The largest observed v001/v002 difference is `0.00048828125` on D128.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples:

| Case | v001 median | v002 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| short S1: B1, L1024, D64, BS16 | 0.0421 ms | 0.0403 ms | noise; same kernel |
| long S1: B1, L16384, D64, BS16 | 0.1966 ms | 0.1952 ms | noise; same kernel |
| multi S4: B4, L1024, D64, BS16 | 0.1299 ms | 0.1280 ms | noise; same kernel |
| wide: B2, L1024, D128, BS32 | 0.2557 ms | 0.1743 ms | **+31.79%** |
| 64K stress: B1, L65536, D64, S16, BS64 | 80.6850 ms | 49.6288 ms | **+38.49%** |
| proxy total | 81.3093 ms | 50.1666 ms | **+38.30%** |

All five aggregate samples improved: `+38.2831%, +38.3005%, +38.3036%,
+38.3106%, +38.3119%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`6354f2f2e10fd949a7506a635e0fb3f682b7524db830fa017773ce98bd2e71c8`.
