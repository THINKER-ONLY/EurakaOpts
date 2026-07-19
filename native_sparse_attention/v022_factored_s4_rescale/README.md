# NSA v022: Factored S4 Rescale

## Objective

Extend v021's max-rescaling factorization to the three S4 official kernels.

## Change

- replace `previous_max * scale - current_max * scale` with the algebraically
  equivalent `(previous_max - current_max) * scale` for S4;
- retain the original per-score expression, matching v021's validated S8
  optimization granularity;
- retain v021's S8 rescale, cached S2 block indices, factored S1/S2 score
  scale, shared-score PV, reduce-max and QK clear-accum, prenormalized weights,
  GEMM policies, and shared K/V storage.

The change removes one multiplication per query-head row and selected block.
It only changes floating-point evaluation order; block indices, masks, score
positions, and output coverage are unchanged.

## Correctness

- All three changed outputs match v021 within the official `rtol=1e-2,
  atol=1e-2` tolerance; the maximum absolute difference is `0.00048828`.
- Independent historical-index S4 checks against the PyTorch FP32 reference
  pass for two random seeds, with maximum absolute errors `0.00175357` and
  `0.00178361`.
- The other 106 official shapes are compile-time unchanged and bitwise
  identical to v021.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted cached repeat used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v021 total | v022 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Factored S4 rescale | 3 | 0.18240 ms | 0.17865 ms | **+2.06%** |
| 109-case total | 109 | 5.81775 ms | 5.69120 ms | **+2.27%** |

All seven aggregate samples improved: `+2.0593%, +2.2660%, +4.3087%,
+2.1900%, +2.3182%, +2.2787%, +2.1300%`. The changed S4 family improved in
all seven samples by `+1.4775%` to `+2.1102%`, and all three S4 medians
improved.

Only the S4 family result is attributed to this change. The larger full-run
difference includes positive device-state drift in unchanged kernels.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`ba730608ad8f3e16071ef9ec4e405c2a6ddc264a94db6ddec598445f08f95a9f`.
