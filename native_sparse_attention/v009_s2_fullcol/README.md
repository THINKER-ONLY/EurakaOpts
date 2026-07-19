# NSA v009: S2 FullCol

## Objective

Tune the three official D64/S2/BS16 shapes that v005's gathered kernel turns
into one 32-position attention tile.

## Change

- select `T.GemmWarpPolicy.FullCol` for D64/S2/BS16;
- retain v008's `FullCol` policy for all S1 shapes;
- retain `FullRow` for S4, S8, S16, and other multi-block shapes;
- keep the gathered two-block kernel, 128-thread CTA, masking, softmax, and
  shared-memory reuse unchanged.

The policy applies to both gathered QK and PV GEMMs. Every call recomputes from
its current tensors and block indices; only compiled kernels are cached.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All three official S2 outputs are bitwise identical to v008.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples with deterministic
historical block indices:

| Official family | Cases | v008 total | v009 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D64 / S2 / BS16 | 3 | 0.1296 ms | 0.1250 ms | **+2.874%** |

All five aggregate samples improved: `+2.8737%, +1.8728%, +3.5753%,
+3.9528%, +2.6549%`. Per-shape median improvements were `+4.90%, +2.92%,
+2.63%` for the B1/L256, B2/L512, and B4/L1024 cases respectively.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`025851c9678bbeec00413b5cc2997f01e2e82fbf46b4bfc477543d111fb9e395`.
