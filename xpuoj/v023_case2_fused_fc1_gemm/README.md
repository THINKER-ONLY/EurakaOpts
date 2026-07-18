# XPUOJ v023: Case2 Fused FC1 Gate/Up GEMM

## Objective

Reduce FC1 synchronization overhead without changing the established v022
tile sizes, FC2 schedule, fast-math mode, or safe-memory setting.

## Change

For the official `(hidden=7168, intermediate=2048, experts=32)` case only,
concatenate each 128-column gate and up weight tile in one 256-column shared
tile and execute one `N=256` FullRow GEMM instead of two `N=128` GEMMs. The
combined FP32 accumulator is split in the epilogue by copying its gate half to
a fragment before applying SiLU and multiplying by the up half.

Both weight tiles still occupy 32 KiB total shared memory, and the GEMM uses
the same BM128/BK64, 256 threads, and one pipeline stage. The generated FC1
loop has one fewer static `__syncthreads()` site, so every K iteration performs
one shared-memory synchronization instead of independently synchronizing the
gate and up streams.

The original v022 factory remains selected for case1 and case3. Their generated
MACA device sources are byte-identical to v022. The fused factory is also used
for the one-expert `(7168,2048)` correctness oracle.

## Correctness

- All three official-shape proxies are bit-identical to v022 on valid and
  padded rows.
- The `(2048,8192)` FP32 oracle has maximum absolute error `0.00040603`.
- The fused `(7168,2048)` FP32 oracle has maximum absolute error `0.00095081`.
- Alternative fused BK32/BK128, BN64/BN256, 128/512 threads, and stage0/stage2
  trials were correct but regressed and remain internal autotuning trials.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started with physical GPU utilization at 0%,
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v022 median | v023 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 6.2568 ms | 6.2486 ms | +0.13% (byte-identical control) |
| case2 | 10.1298 ms | 9.9047 ms | **+2.21%** |
| case3 | 19.6848 ms | 19.6532 ms | +0.12% (byte-identical control) |
| total | 36.0714 ms | 35.8066 ms | **+0.73%** |

All five aggregate pairs improved: `+0.7104%, +0.7539%, +0.5689%, +0.7222%,
+0.6759%`. The case2 pairs were `+2.21%, +2.34%, +2.13%, +2.22%, +2.15%`.
Only case2's improvement is attributed to generated-code changes; the small
case1/case3 differences are measurement noise.

Decision: **accepted as the local baseline**. The gain is stable but was below
the standalone large-improvement threshold.

## XPUOJ Result

Submission `#64011` was accepted:

```text
Status:          Accepted
Total score:     72.67
Displayed time:  28.390 ms
Case scores:     73 / 73 / 72
Case times:      4.701 / 7.922 / 15.767 ms
```

The online total is 2.20% faster than v020's 29.028 ms and 34.24% faster
than v008's 43.172 ms. Case2 improved by 4.57% over v020, while case1 was
effectively unchanged. Every testcase passed, and the submitted source has
SHA-256 `6faa0039d92dfe7e1c5bf7c27bfe5a1dcb16c8320ab3ee93642fc93e5e07360e`.

Final decision: **accepted as the new XPUOJ baseline**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
