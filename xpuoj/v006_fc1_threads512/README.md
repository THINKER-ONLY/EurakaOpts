# XPUOJ v006: FC1 Threads 512

## Objective

Test whether assigning FC1's two 128 x 128 FP32 accumulator fragments across
512 threads reduces per-thread register pressure on C500.

## Parent Evidence

`v004_fc1_shared_weight` scored 56.33 on XPUOJ with case scores 57/56/56 and
a displayed total time of 52 ms. The v005 epilogue experiment had the same
score and time, so this version branches directly from v004.

## Single Change

The FC1 `T.Kernel` launch changes from 256 to 512 threads. FC2 remains at 256
threads. FC1 shared-memory reuse, serial K-loop and barriers, FC2 BN256, block
sizes, CTA grids, kernel count, arithmetic, dtypes, metadata indexing, caches,
and public API are unchanged.

## Hypothesis And Risk

This does not reduce FLOPs, memory traffic, kernel count, or CTA count. It tests
one mechanism only: more FC1 threads may lower per-thread fragment/register
ownership and reduce spilling or improve scheduling. The trade-off is that a
larger thread block can lower residency or incur additional synchronization
cost. Only XPUOJ can establish C500 performance.

## Verification

- AST tests require FC1 to use 512 threads and FC2 to retain 256.
- AST normalization proves the FC1 thread count is the only source difference
  from v004.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- `(hidden=2048, intermediate=8192)`: maximum absolute error
  `0.001953125`, mean absolute error `5.124584276927635e-05`.
- `(hidden=7168, intermediate=2048)`: maximum absolute error
  `0.001953125`, mean absolute error `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.

## XPUOJ Result

```text
Status:          Accepted (correctness)
Total score:     54.67
Displayed time:  55 ms
Memory:          22.2 G
Case scores:     56 / 54 / 54
Case times:      9 / 16 / 31 ms (display-rounded)
```

The v004 parent scored 56.33 at 52 ms, with case scores 57/56/56 and times
9/15/28 ms. The 512-thread FC1 therefore regressed all case scores, increased
the total time by 3 ms, and hurt the two wide-hidden cases most.

FC1 executes two `T.sync_threads()` calls per hidden K tile. Case 1 has 16 K
tiles (`2048 / 128`) and therefore 32 barriers per CTA, while cases 2 and 3
have 56 K tiles (`7168 / 128`) and therefore 112 barriers. The larger
regression in cases 2 and 3 is consistent with 512-thread CTA residency and/or
barrier cost outweighing any reduction in per-thread register pressure. The
aggregate judge data cannot distinguish those two costs without a C500
profiler, so no stronger root-cause claim is made.

Decision: **rejected**. Keep v004 as the best baseline, do not inherit from
v006, and stop exploring larger FC1 thread blocks unless target profiling
provides contrary evidence.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
