# NSA v019: Factored Softmax Scale

## Objective

Reduce scalar work in the score exponentiation for the 100 single-block and
three gathered two-block official cases.

## Change

- replace `score * scale - max * scale` with the algebraically equivalent
  `(score - max) * scale` for S1 and gathered S2 softmax weights;
- apply the same factorization to the online-softmax max-rescaling expression
  for S2-compatible code paths;
- retain v018's original expression for S4 and S8, where the factorized form
  did not improve the representative kernels;
- retain v018's adaptive shared-score PV, S2 post-mask clear-accum, reduce-max
  clear, QK clear-accum, prenormalized weights, GEMM policies, and shared K/V
  storage.

The factorization removes one multiplication per score in the generated
kernel. It changes only floating-point evaluation order; block-index handling,
causal masks, memory accesses, and output coverage are unchanged.

## Correctness

- All 109 official outputs match v018 within the official `rtol=1e-2,
  atol=1e-2` tolerance; the maximum absolute difference is `0.00195313`.
- The standard changed-input PyTorch-reference checks pass with maximum
  absolute errors `0.00085354` and `0.00094223`.
- S4 and S8 are compile-time unchanged from v018.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted cached repeat used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v018 total | v019 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| S1 | 100 | 5.41463 ms | 5.30715 ms | **+1.99%** |
| S2 | 3 | 0.13865 ms | 0.13519 ms | **+2.49%** |
| Changed S1/S2 | 103 | 5.55328 ms | 5.44234 ms | **+2.00%** |
| 109-case total | 109 | 5.96934 ms | 5.85286 ms | **+1.94%** |

All five aggregate samples improved: `+0.3069%, +1.9571%, +1.9584%,
+1.9233%, +1.9358%`. Every per-shape median improved. An independent first
full run measured `+2.1845%` overall and `+2.1784%` on the changed family,
confirming the direction across separate runs.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`7d2759fe96e17fca06760012c0d8d920173840451448d7854f6f7957b015d61c`.
