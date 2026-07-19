# NSA v006: D128 BS32 FullCol

## Objective

Improve the eight published `D=128, selected_blocks=1, block_size=32` shapes
without changing other NSA shape families.

## Change

- select `T.GemmWarpPolicy.FullCol` at compile time for D128/S1/BS32;
- retain `FullRow` for every other shape;
- keep v005's thread policy, single-block softmax specialization, shared-memory
  reuse, and gathered S2 kernel unchanged.

The policy applies to both QK and PV GEMMs for this shape family. Every call
recomputes from its current tensors and block indices; only compiled kernels are
cached.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All eight D128/S1/BS32 outputs are bitwise identical to v005.
- The expanded regression set covers S1, S2, S4, S8, BS16, BS32, BS64, recent
  and historical indices, and the 64K/S16 stress shape.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted family run used five warmups, twenty
alternating measured calls per sample, and five samples:

| Published family | v005 total | v006 total | Median paired improvement |
| --- | ---: | ---: | ---: |
| 8 x D128/S1/BS32 shapes | 0.5766 ms | 0.5689 ms | **+1.43%** |

All five aggregate samples improved: `+1.1252%, +1.6371%, +1.4288%,
+1.5652%, +1.3124%`. Individual improvements ranged from about `+0.6%` to
`+3.3%`; repeated runs confirmed that sub-percent single-case movement is event
timing noise while the family aggregate remains positive.

The complete local regression total, dominated by the non-official 64K/S16
stress case, improved by `+0.13%` with no stable case regression. Attempts to
use 256 threads for D128/BS32 and 128 threads for D128/BS16 were rejected by the
MACA layout inference constraints.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`3c30898f2ce75b71ffcebb0f4df9cb65c365dc887367b43b7ff43f019b5c42ec`.
