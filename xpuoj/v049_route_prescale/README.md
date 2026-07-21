# XPUOJ v049: Route Prescale

## Objective

Move routed-weight scaling to the narrowest existing activation stage for the
E32 and E64 streamed paths, reducing elementwise work without caching or
skipping any required computation.

## Change

For E32 and E64, each 16-expert SwiGLU chunk now multiplies its valid
`M x 2048` activation rows by the current FP32 route weights before the down
projection. The resulting down BMM therefore produces an already weighted
`M x 7168` output, and the final unpack kernel only copies valid rows.

Previously the same scalar route weight was applied independently to all 7168
elements of each down-projection output. The new ordering performs the scaling
over 2048 elements per row, eliminating approximately 71% of those elementwise
multiplications. E16 deliberately retains v048's original route-in-unpack path.
Every invocation still packs the current input and recomputes FC1, SwiGLU,
route scaling, down projection, and every valid output row. There is no output,
input-identity, or timing cache.

The operation is an algebraic reassociation in FP16:
`(activation @ W) * route` becomes `(activation * route) @ W`. Small
rounding differences are expected and are covered by the correctness results
below.

## Correctness

Random tests using all three disclosed expert distributions passed against the
FP32 oracle:

| Case | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| E16 | 0.001953125 | 0.0001468057 |
| E32 | 0.001953125 | 0.0001413113 |
| E64 | 0.00244140625 | 0.0001430177 |

After graph capture, activation and route weights were modified in place for
E32 and E64. Graph replay and a fresh eager execution were bit-identical for
both distributions (`max_abs=0`). Padded-row checks also passed.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Because multi-stream module load order materially biases
timing, the final comparison used independent fresh processes in the crossed
order `v048, v049, v049, v048, v048, v049`.

| Case | v048 process medians | v049 process medians | Improvement |
| --- | --- | --- | ---: |
| E32 | 3.12361, 3.12178, 3.12349 ms | 3.11496, 3.11216, 3.11209 ms | **+0.363%** |
| E64 | 6.32265, 6.33171, 6.31936 ms | 6.27921, 6.28628, 6.27534 ms | **+0.687%** |

E16 is byte-equivalent to v048. Weighting the unchanged E16 runtime and the two
measured improvements gives an estimated aggregate improvement of
**approximately +0.48%** over v048.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`de5ee3eff43ae65002542c979e3fd29f02177e707a54a081468ec9c6d9e29249`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
