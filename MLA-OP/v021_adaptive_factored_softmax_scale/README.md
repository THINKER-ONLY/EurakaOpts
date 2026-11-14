# MLA Decode v021: Adaptive Factored Softmax Scale

## Objective

Reduce scalar work in MLA score exponentiation while retaining v020's
preferred arithmetic on long memory-bound and fused-part-0 shapes.

## Change

- replace `score * scale - max * scale` with the algebraically equivalent
  `(score - max) * scale` in 26 profitable official shapes;
- apply the same factorization to the online-softmax max-rescaling expression;
- retain v020's original expression for `B32/8K`, `B32/16K`, and `B32/32K`,
  whose QK kernels keep the fused part-0 accumulator;
- retain the original expression for `B1/64K` and `B32/64K`, where repeated
  runs found no stable arithmetic-bound gain;
- retain v020's adaptive QK no-part-0, direct and materialized shared-score PV,
  adaptive reduce-max clear, cached split LSE, QK clear-accum, fused 576-D QK,
  and split reduction.

The factorization removes one multiplication per score in the generated
kernel. It changes only floating-point evaluation order; all input reads,
attention positions, reductions, and output writes remain unchanged.

## Correctness

- All 31 candidate outputs match v020 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00018311`.
- Independent `B4/2K` checks against the PyTorch FP32 reference passed for two
  random seeds, with maximum absolute errors `0.00015938` and `0.00017785`.
- The five retained shapes are compile-time unchanged and bitwise identical
  to v020.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v020 total | v021 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Factored softmax path | 26 | 12.43717 ms | 12.36477 ms | **+0.58%** |
| 31-case total | 31 | 24.73016 ms | 24.63099 ms | **+0.39%** |

All five aggregate samples improved: `+0.2311%, +0.4552%, +0.4219%,
+0.3707%, +0.3926%`. Every per-shape median improved. The changed-family
samples improved by `+0.3507%` to `+0.6804%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`14fd9a0f632021664d2f3b9ddf6928046dcdf738b8ca1777402752c74e6ca34b`.
