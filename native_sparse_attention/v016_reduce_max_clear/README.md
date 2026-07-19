# NSA v016: Reduce-Max Clear

## Objective

Remove redundant score-maximum initialization from the six official S4/S8
online-softmax shapes.

## Change

- retain the pre-loop `scores_max = -inf` state required by the first selected
  block;
- after saving the previous block maximum, remove the separate per-block
  `scores_max = -inf` fill;
- let the existing `T.reduce_max(..., clear=True)` initialize the current
  reduction destination;
- retain v015's QK clear-accum scheduling, prenormalized S1/S2 weights,
  gathered S2 kernel, thread and GEMM policies, and shared-memory reuse.

## Correctness

- All 109 official outputs are bitwise identical to v015.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v015 total | v016 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| S4/S8 changed path | 6 | 0.40557 ms | 0.39832 ms | **+1.79%** |
| 109-case total | 109 | 5.80239 ms | 5.68514 ms | **+1.98%** |

All five aggregate samples improved: `+2.0005%, +2.0912%, +1.9320%,
+1.9844%, +1.8961%`. Every per-shape median improved. A separate targeted
S4/S8 run measured `+2.19%`, with all six shapes between `+1.46%` and
`+3.19%`. Because the other 103 compile-time paths are unchanged, their
positive difference in the full run is treated as device-state drift rather
than attributed to this change.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`baf94f0d49a6b90acdd67a23d4a8f1690bbdcfa21d91f3bb86b417e4deded36f`.
