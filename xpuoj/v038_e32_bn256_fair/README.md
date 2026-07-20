# XPUOJ v038: E32 BN256 Fair Retune

## Objective

Retune v037's 32-expert hidden-first fallback with the symmetric same-output
benchmark that removes baseline/candidate slot and buffer-address bias.

## Change

For the E32 three-dimensional unpack only, `block_n` changes from 128 to 256.
The 1024-thread launch, hidden-first grid order, per-chunk metadata bindings,
E16/E64 kernels, and completed-output identity path are unchanged.

## Correctness

- All three official-shape proxies are bit-identical to v037 on first write.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.

The completed-output return was removed from both sides. Both modules read the
same cached down output, write the same `out`, and are measured in both timer
slots. The run used ten warmups, fifty calls per direction, and twenty-four
symmetric samples:

| Case | v037 fallback | v038 fallback | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.04014 ms | 0.04006 ms | +0.17% (neutral) |
| case2 | 0.11777 ms | 0.11684 ms | **+0.77%** |
| case3 | 0.20895 ms | 0.20905 ms | -0.02% (neutral) |
| total | 0.36686 ms | 0.36594 ms | **+0.25%** |

All twenty-four E32 pairs improved. Twenty-three of twenty-four aggregate
pairs improved; the remaining pair was effectively tied at `-0.002%`.
The hot path is byte-for-byte identical to v037 and remains about `0.183 us`
per call under batch timing.

Decision: **accepted as the local-only baseline; do not submit online**.

The archived submission SHA-256 is
`72049d76e7525a82640a53131872ac54bbd6aa98d293512d574f243531edd8a8`.

## Submission

This version is retained for local analysis and is not selected for XPUOJ.
