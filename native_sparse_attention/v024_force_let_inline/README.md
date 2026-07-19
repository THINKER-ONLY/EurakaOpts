# NSA v024: Force Let Inline

## Objective

Remove residual symbolic indexing and temporary-expression overhead from the
generated MetaX kernels.

## Change

- enable TileLang's `TL_FORCE_LET_INLINE` pass configuration;
- inline let-bound expressions before the normal MACA lowering pipeline;
- retain all v023 kernel algorithms, thread counts, GEMM policies, masks,
  numerical expressions, and shared-memory paths unchanged;
- retain v023's factored S4 scores and rescale, factored S8 rescale, cached S2
  indices, prenormalized weights, and shared K/V storage.

The pass produces materially different MetaX device code, unlike pass flags
that are ignored by the MACA pipeline. Every invocation still reads current
inputs and writes the complete caller-provided output.

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
| Force-let-inline kernels | 109 | 5.60716 ms | 5.27426 ms | **+5.91%** |

All seven aggregate samples improved: `+5.7272%, +5.6903%, +6.0501%,
+5.1816%, +6.0257%, +6.6479%, +5.9113%`. The initial cache-populating run
also improved by `+6.92%` median, with all three aggregate samples positive.
Every per-case median in the cached run improved.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`e93a00a5454e626a78bb1fd980ac4292a2496488d859d0823c05c52432c9462d`.
