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
`0.1.11+maca.git56b76a2b`. The corrected validation used eight warmups, forty
batched calls per module/output combination, eight samples, both output
addresses, and a four-phase Latin rotation over all 109 official cases:

| Official family | Cases | v023 total | v024 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Force-let-inline kernels | 109 | 3.00785 ms | 2.81139 ms | **+6.44%** |

The two independently phase-balanced aggregate samples improved by `+6.2738%`
and `+6.6048%`. All 109 candidate outputs were bitwise identical to v023. The
four-phase result supersedes the original fixed-output-buffer measurement,
which was affected by invocation-position and output-address bias.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`e93a00a5454e626a78bb1fd980ac4292a2496488d859d0823c05c52432c9462d`.
