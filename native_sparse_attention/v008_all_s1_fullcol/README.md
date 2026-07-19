# NSA v008: All S1 FullCol

## Objective

Finish tuning the seven remaining official single-block shapes after v007 by
covering D64 with block sizes 32 and 64.

## Change

- select `T.GemmWarpPolicy.FullCol` at compile time for every S1 shape;
- retain `FullRow` for all S2, S4, S8, and S16 shapes;
- keep v007's thread policy, single-block softmax specialization,
  shared-memory reuse, and gathered S2 kernel unchanged.

The new policy applies to both QK and PV GEMMs for three D64/BS32 and four
D64/BS64 official shapes. Every call recomputes from its current tensors and
block indices; only compiled kernels are cached.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All seven newly selected official outputs are bitwise identical to v007.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples with deterministic
historical block indices:

| Official family | Cases | v007 total | v008 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D64 / S1 / BS32 | 3 | 0.1102 ms | 0.1079 ms | **+2.60%** |
| D64 / S1 / BS64 | 4 | 0.1561 ms | 0.1517 ms | **+2.47%** |
| Combined | 7 | 0.2663 ms | 0.2596 ms | **+2.523%** |

All five combined samples improved: `+2.6734%, +1.8346%, +2.5540%,
+2.2574%, +2.5233%`. Every per-shape median improved; the smallest measured
gain was `+0.90%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`8149edbe5df5ca087ca873e77dc23fc01fb313549189a7ee43b2093c6e2b87bd`.
