# NSA v023: Factored S4 Scores

## Objective

Remove a repeated scale multiplication from every S4 score exponentiation.

## Change

- replace `score * scale - max * scale` with the algebraically equivalent
  `(score - max) * scale` for S4 score weights;
- retain v022's already-factored S4 max-rescale expression;
- retain the original S8 score expression, which regressed when factored;
- retain v022's S4/S8 max rescale, cached S2 block indices, factored S1/S2
  score scale, shared-score PV, reduce-max and QK clear-accum, prenormalized
  weights, GEMM policies, and shared K/V storage.

The change removes one multiplication per score and selected block. It only
changes floating-point evaluation order; block indices, masks, score
positions, and output coverage are unchanged.

## Correctness

- All three changed outputs match v022 within the official `rtol=1e-2,
  atol=1e-2` tolerance; the maximum absolute difference is `0.00097656`.
- Independent historical-index S4 checks against the PyTorch FP32 reference
  pass for two random seeds, with maximum absolute errors `0.00175357` and
  `0.00178361`.
- The other 106 official shapes are compile-time unchanged and bitwise
  identical to v022.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v022 total | v023 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Factored S4 scores | 3 | 0.17030 ms | 0.16721 ms | **+1.82%** |
| 109-case total | 109 | 5.77988 ms | 5.66895 ms | **+1.89%** |

All seven aggregate samples improved: `+1.8523%, +1.9951%, +4.1736%,
+2.1215%, +1.8825%, +1.8916%, +1.8887%`. The changed S4 family improved in
all seven samples by `+1.3799%` to `+2.1801%`, and all three S4 medians
improved.

Only the S4 family result is attributed to this change. The larger full-run
difference includes positive device-state drift in unchanged kernels.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`55576bceaf8c6cf6acb6d4d9007fed02b5f35249482f63c0f83654f540c695a5`.
