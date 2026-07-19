# NSA v018: Short Shared Score PV

## Objective

Improve short D64/D128 single-block kernels by using shared-memory score tiles
for PV without regressing D32, multi-block, or long-sequence paths.

## Change

- for 64-thread `S == 1`, `D >= 64`, and `seq_len <= 4096` kernels, convert
  normalized FP32 scores into `S_shared` and run the PV GEMM from shared;
- retain the FP16 fragment PV path for D32, all 64-thread multi-block kernels,
  and longer 64-thread single-block kernels;
- retain the existing shared-score path for 128/256-thread kernels;
- retain v017's S2 post-mask clear-accum, reduce-max clear, QK clear-accum,
  prenormalized weights, GEMM policies, and K/V shared-memory reuse.

The shared PV layout lowers fixed staging overhead for short D64/D128 S1
kernels. D32 and S8 experiments regressed with the same layout, and long
D64/D128 shapes were noise-level, so they remain on the prior fragment path.

## Correctness

- All 109 official outputs are bitwise identical to v017.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; every invocation reads current inputs and
  block indices and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v017 total | v018 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Changed short D64/D128 BS16 | 48 | 2.60027 ms | 2.55008 ms | **+1.93%** |
| 109-case total | 109 | 5.77075 ms | 5.65311 ms | **+2.06%** |

All five aggregate samples improved: `+1.9071%, +2.0721%, +2.0609%,
+1.8565%, +2.0727%`. Every per-shape median improved, and all outputs are
bitwise identical. Unselected paths are compile-time unchanged, so the changed
family result is the primary performance evidence.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`a47024796fef9a20d81bd517ab8a3574486f3a2fad423336825b48879fb10f8b`.
