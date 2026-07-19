# NSA v010: S4 S8 FullCol

## Objective

Tune the final six official NSA shapes: D64/BS16 with four or eight selected
blocks on the serial online-softmax kernel.

## Change

- select `T.GemmWarpPolicy.FullCol` for D64/S4/BS16 and D64/S8/BS16;
- retain v009's `FullCol` policies for all S1 shapes and gathered D64/S2/BS16;
- retain `FullRow` for other multi-block shapes, including the local S16
  stress proxy;
- keep the serial block loop, online softmax, 64-thread CTA, masking, and
  shared-memory reuse unchanged.

The policy applies to each per-block QK and PV GEMM. Every call recomputes from
its current tensors and block indices; only compiled kernels are cached.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All six official S4/S8 outputs are bitwise identical to v009.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples with deterministic
historical block indices:

| Official family | Cases | v009 total | v010 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D64 / S4 / BS16 | 3 | 0.1711 ms | 0.1672 ms | **+2.04%** |
| D64 / S8 / BS16 | 3 | 0.2348 ms | 0.2320 ms | **+1.15%** |
| Combined | 6 | 0.4058 ms | 0.3991 ms | **+1.512%** |

All five combined samples improved: `+1.5117%, +1.2620%, +1.4417%,
+1.7277%, +1.8985%`. Every per-shape median improved; the smallest measured
gain was `+0.69%`.

Together with the previously measured S1, S2, and wide-block families, the
local harness now covers all 109 official NSA cases.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`d8885b49ab0f0abd5873475fbad5ae200f556ddf2e48af41d4f4f8033f70d8e6`.
