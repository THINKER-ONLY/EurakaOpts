# XPUOJ v007: FC1 BK64 Dual-Buffer Pipeline

## Objective

Preserve v004's 32 KiB FC1 shared-weight footprint while removing the serial
single-buffer lifecycle and its two explicit barriers per K tile.

## Parent Evidence

`v004_fc1_shared_weight` remains the best C500 result at 56.33 points and
52 ms. The v005 epilogue change was neutral, and the v006 512-thread change
regressed to 54.67 points and 55 ms. This version branches directly from v004.

## Single Change

FC1 changes from one `128 x 128` FP16 weight buffer at BK128 to two
`128 x 64` FP16 Gate/Up buffers at BK64:

```text
v004: one 128 x 128 buffer = 32 KiB, serial loop, two explicit barriers
v007: two 128 x 64 buffers = 32 KiB, pipelined loop, no explicit barriers
```

FC1 retains BM128, BN128, 256 threads, FP32 accumulators, input reuse, CTA
grid, FLOPs, and total global input/weight bytes. FC2 remains byte-for-byte
equivalent at BK128 and BN256. Kernel count, public API, metadata indexing,
caches, dtypes, and mathematical operations are unchanged.

## Hypothesis And Risk

Separate buffers remove the overwrite dependency that forced v004 to serialize
Gate and Up weight lifetimes. BK64 keeps their combined shared footprint at
32 KiB and is supported by the local TileLang MetaX grouped-GEMM example.

The trade-off is twice as many FC1 K-loop iterations and pipeline-control
steps. This is a portable synchronization/resource hypothesis, not a claimed
C500 speedup before XPUOJ measurement.

## Verification

- AST tests require FC1 BK64, two 16 KiB buffers, `T.Pipelined`, and no
  explicit FC1 barriers.
- AST normalization proves this lifecycle is the only source difference from
  v004 and that FC2 is unchanged.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- `(hidden=2048, intermediate=8192)`: maximum absolute error
  `0.001953125`, mean absolute error `5.124584276927635e-05`.
- `(hidden=7168, intermediate=2048)`: maximum absolute error
  `0.001953125`, mean absolute error `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.

## XPUOJ Result

```text
Status:          Accepted
Total score:     56.67
Displayed time:  51 ms
Memory:          22.2 G
Case scores:     58 / 56 / 56
Case times:      8 / 15 / 28 ms (display-rounded)
```

The v004 parent scored 56.33 at 52 ms, with case scores 57/56/56 and times
9/15/28 ms. The improvement is isolated to case 1; cases 2 and 3 are unchanged
at the judge's displayed resolution.

Case 1 uses `hidden=2048`, so FC1 changes from 16 BK128 serial iterations with
32 explicit barriers to 32 BK64 pipelined iterations without explicit
barriers. Cases 2 and 3 use `hidden=7168`; their unchanged results indicate
that the doubled long-K loop/pipeline-control work approximately offsets the
removed barriers. Aggregate judge timing cannot separate those costs more
precisely.

Decision: **accepted as the new best baseline**. Preserve the BK64 dual-buffer
FC1 schedule for case 1. Any further change should be tested independently and
must not assume that the same mechanism has headroom in the wide-hidden cases.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
