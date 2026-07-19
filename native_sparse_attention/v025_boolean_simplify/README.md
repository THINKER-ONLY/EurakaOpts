# NSA v025: Boolean Simplify

## Objective

Canonicalize compound predicates so the MACA lowering pipeline can remove
redundant control-flow and bound expressions.

## Change

- enable TileLang Simplify's `convert_boolean_to_and_of_ors` option;
- retain v024's force-let-inline pass and all kernel algorithms, thread counts,
  GEMM policies, masks, and numerical expressions;
- retain v024's factored S4/S8 expressions, cached S2 indices, prenormalized
  weights, shared-score policy, and shared K/V storage.

The pass reduces the generated source size of the heaviest D64 kernel from
11276 to 10974 bytes. It changes compiler control-flow only; every invocation
still reads current inputs and writes the complete caller-provided output.

## Correctness

- All 109 official candidate outputs are bitwise identical to v024.
- The standard two-input FP32-reference gate retains maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; no input-dependent result is reused.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted cached run used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v024 total | v025 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Boolean-simplified kernels | 109 | 5.64332 ms | 5.48479 ms | **+2.90%** |

All seven aggregate samples improved: `+2.2395%, +2.9685%, +2.8846%,
+2.7708%, +2.9896%, +2.8961%, +2.9236%`. Every per-case median improved,
and all three cache-populating aggregate samples were also positive.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`a6a7fb2d884418027adc3a7b74eb9bb90be4470e9b43a9cf1c6e188279e809d3`.
