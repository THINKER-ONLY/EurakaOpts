# NSA v024: Warp Specialization

## Objective

Allow the current TileLang/MACA compiler to apply its default warp
specialization scheduling to every official kernel.

## Change

- remove the explicit `TL_DISABLE_WARP_SPECIALIZED` pass configuration;
- retain fast math and all v023 kernel source, shapes, thread counts, GEMM
  policies, score paths, masks, and numerical expressions unchanged;
- retain v023's factored S4 scores and rescale, factored S8 rescale, cached S2
  indices, prenormalized weights, shared-score policy, and shared K/V storage.

This is a compiler-scheduling change only. Every invocation still reads the
current Q, K, V, and block indices and writes the complete caller-provided
output.

## Correctness

- All 109 official candidate outputs are bitwise identical to v023.
- The standard two-input FP32-reference gate retains maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; no input-dependent result is reused.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted cached run used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v023 total | v024 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Warp-specialized kernels | 109 | 6.17238 ms | 6.04357 ms | **+2.04%** |

All seven aggregate samples improved: `+2.0736%, +1.9507%, +2.0558%,
+2.1342%, +1.8503%, +1.9474%, +2.0375%`. The initial cache-populating run
also improved by `+2.38%` median, with all five aggregate samples positive.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`3c36bd3536cf522cdab862d10377da1c64b4e60182f2e13d933fdb179546901a`.
