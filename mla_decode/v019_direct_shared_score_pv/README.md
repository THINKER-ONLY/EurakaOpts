# MLA Decode v019: Direct Shared Score PV

## Objective

Extend v018's shared-score PV staging optimization to the seven direct split-K
shapes.

## Change

- retain the FP16 `S_shared` tile already produced by the direct online
  softmax path;
- pass `S_shared` directly to the PV GEMM;
- remove the separate FP16 `acc_s_cast` fragment and the
  `S_shared -> acc_s_cast` copy from the shared split-K kernel;
- retain v018's shared-score materialized part-0 PV, adaptive reduce-max clear,
  cached split LSE, QK clear-accum, fused 576-D QK, and split reduction.

The FP32 exponential weights are still converted to FP16 through the same
`acc_s -> S_shared` copy. Only the second staging copy is removed, so numerical
operations and rounding are unchanged.

## Correctness

- All 31 official outputs are bitwise identical to v018.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v018 total | v019 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Direct changed path | 7 | 1.44090 ms | 1.39284 ms | **+3.34%** |
| 31-case total | 31 | 25.35255 ms | 25.16846 ms | **+0.72%** |

All five aggregate samples improved: `+0.3613%, +0.7954%, +0.8137%,
+0.6643%, +0.7250%`. Every per-shape median improved. The seven direct shapes
improved by `+2.75%` to `+5.17%`. Because materialized code paths are
compile-time unchanged, their positive full-run difference is treated as
device-state drift rather than attributed to this change.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`d8fa53a1d8df8d124d95f6fddef437770ec60ec27103b0bc3faebb889ae2435d`.
