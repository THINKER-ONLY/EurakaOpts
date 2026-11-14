# MLA Decode v020: Adaptive QK Without Part 0

## Objective

Reduce register pressure in the materialized-score QK kernel by moving output
part 0 to the separate PV grid when that trade is profitable.

## Change

- omit the `16x128` FP32 part-0 output accumulator, rescaling, PV GEMM, and
  output store from the QK kernel on 21 materialized-score shapes;
- run all four 128-D output parts in the following PV grid on those shapes;
- retain v019's fused QK part-0 path for `B32/8K`, `B32/16K`, and `B32/32K`,
  where the extra PV grid work outweighs the QK register-pressure reduction;
- retain v019's direct shared-score PV, shared-score materialized PV, adaptive
  reduce-max clear, cached split LSE, QK clear-accum, fused 576-D QK, and split
  reduction.

The selection is compile-time shape specialization. Every invocation still
reads current inputs, computes full attention, and writes all 512 output
elements.

## Correctness

- All 31 candidate outputs match v019 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum v019/candidate absolute difference is
  `0.00061035`.
- Independent `B4/2K` checks against the PyTorch FP32 reference passed for two
  random seeds, with maximum absolute errors `0.00015938` and `0.00017785`.
- The three retained `B32` shapes and all seven direct-path shapes are
  compile-time unchanged from v019.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v019 total | v020 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Changed materialized path | 21 | 18.08300 ms | 17.41606 ms | **+3.69%** |
| 31-case total | 31 | 25.49486 ms | 24.80146 ms | **+2.72%** |

All five aggregate samples improved: `+2.5615%, +2.7559%, +2.7178%,
+2.7232%, +2.7941%`. Every per-shape median improved. The retained fused
`B32/8K`, `B32/16K`, and `B32/32K` outputs are bitwise identical to v019.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`58b35ba46c3f2e83addaefa565658eb173c6f80176ce7dacffe11b613fd2cbcb`.
