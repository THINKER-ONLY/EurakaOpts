# NSA v013: S2 Premask Clear

## Objective

Remove redundant score initialization from the gathered two-block kernel for
the three official D64/S2/BS16 shapes.

## Change

- clear the complete 32-position FP32 score fragment once before gathering;
- leave complete historical blocks at zero instead of running a 16x16
  per-element initialization loop for each block;
- overwrite current or partially visible blocks with the original causal mask;
- overwrite invalid blocks with `-inf` as before;
- retain v012's GEMM policies, gathered QK/PV operations, softmax, thread
  policy, and shared-memory reuse.

The initialization remains correct for arbitrary mixtures of historical,
partial, duplicate, invalid, and future block indices.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All three official S2 outputs are bitwise identical to v012.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples with deterministic
historical block indices:

| Official family | Cases | v012 total | v013 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D64 / S2 / BS16 | 3 | 0.1375 ms | 0.1320 ms | **+3.830%** |

All five aggregate samples improved: `+3.8298%, +3.4178%, +4.6042%,
+3.5448%, +4.2196%`. Per-shape median improvements were `+3.14%, +4.42%,
+4.18%` for the B1/L256, B2/L512, and B4/L1024 cases respectively.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`6fef6704570afa79d5f445eb612f459ffdfd00123dd115fabe3c4a1c137a8db0`.
