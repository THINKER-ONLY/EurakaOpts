# NSA v017: S2 Post-Mask Clear-Accum

## Objective

Eliminate the standalone score-fragment clear from the gathered two-block
kernel without weakening its block-index or causal-mask semantics.

## Change

- gather K blocks as before and continue zero-filling invalid K regions;
- initialize the complete QK accumulator through GEMM `clear_accum=True`;
- apply partial causal and invalid-block score masks after QK instead of
  seeding the accumulator before QK;
- leave complete historical blocks untouched after QK;
- retain v016's reduce-max clear, prenormalized weights, gathered PV, thread
  and GEMM policies, and shared-memory reuse.

The post-mask path supports arbitrary mixtures of historical, partial,
duplicate, invalid, and future block indices. Invalid K regions are still
zero-filled before QK, and their resulting scores are overwritten with
`-inf` before softmax.

## Correctness

- All 109 official outputs are bitwise identical to v016.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; every invocation reads current inputs and
  block indices and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v016 total | v017 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| S2 changed path | 3 | 0.13949 ms | 0.13486 ms | **+3.32%** |
| 109-case total | 109 | 5.82316 ms | 5.70967 ms | **+1.92%** |

All five aggregate samples improved: `+0.3694%, +2.0207%, +1.9198%,
+1.9125%, +3.1107%`. Every per-shape median improved. A separate targeted S2
run measured `+3.31%`; the three official shapes improved by `+3.00%` to
`+4.67%`. Because the other 106 compile-time paths are unchanged, their
positive full-run difference is treated as device-state drift rather than
attributed to this change.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`60cee01bdbab44f3df3c470be17137b8997cf064de98493485e28311bb14c402`.
