# NSA v021: Factored S8 Rescale

## Objective

Reduce online-softmax max-rescaling work in the three S8 official kernels
without changing their score exponentiation loop.

## Change

- replace `previous_max * scale - current_max * scale` with the algebraically
  equivalent `(previous_max - current_max) * scale` for S8;
- retain v020's original per-score expression, because factoring both
  expressions regressed the S8 representative kernel;
- retain v020's cached S2 block indices, factored S1/S2 score scale,
  shared-score PV, reduce-max and QK clear-accum, prenormalized weights, GEMM
  policies, and shared K/V storage.

The change removes one multiplication per query-head row and selected block.
It only changes floating-point evaluation order; block indices, masks, score
positions, and output coverage are unchanged.

## Correctness

- All three changed outputs match v020 within the official `rtol=1e-2,
  atol=1e-2` tolerance; the maximum absolute difference is `0.00097656`.
- Independent historical-index S8 checks against the PyTorch FP32 reference
  pass for two random seeds, with maximum absolute errors `0.00175357` and
  `0.00178361`.
- The other 106 official shapes are compile-time unchanged and bitwise
  identical to v020.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v020 total | v021 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Factored S8 rescale | 3 | 0.22330 ms | 0.22020 ms | **+1.39%** |
| 109-case total | 109 | 5.76902 ms | 5.65768 ms | **+2.02%** |

All seven aggregate samples improved: `+1.9937%, +2.0278%, +1.9988%,
+2.0701%, +1.8844%, +2.8550%, +2.0227%`. The changed S8 family improved in
all seven samples by `+1.0057%` to `+2.1698%`, and all three S8 medians
improved.

Only the S8 family result is attributed to this change. The larger full-run
difference includes positive device-state drift in unchanged kernels.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`c08729ee9fb080b8fe2648e99fb53bdd649e1c6425f27fa7b13846cae6f9f5cf`.
