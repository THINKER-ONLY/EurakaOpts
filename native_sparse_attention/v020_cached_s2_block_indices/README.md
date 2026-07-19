# NSA v020: Cached S2 Block Indices

## Objective

Remove redundant block-index loads from the gathered two-block kernel.

## Change

- load and multiply the two selected block indices once per CTA;
- cache both block starts in a two-element local int32 buffer;
- reuse the cached starts during K gather, score masking, and V gather instead
  of reading `BlockIndices` independently in all three phases;
- retain v019's factored softmax scale, shared-score PV, S2 post-mask
  clear-accum, reduce-max clear, QK clear-accum, prenormalized weights, GEMM
  policies, and shared K/V storage.

The cached values come from the current invocation. Historical, partial,
duplicate, invalid, and future indices retain the same masking and gather
semantics.

## Correctness

- All 109 official outputs are bitwise identical to v019.
- The standard two-input PyTorch-reference checks retain maximum absolute
  errors `0.00085354` and `0.00094223`.
- The 106 non-S2 shapes are compile-time unchanged from v019.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted full run used five warmups, twenty
alternating measured calls per sample, and seven samples over all 109 official
cases:

| Official family | Cases | v019 total | v020 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Cached S2 path | 3 | 0.13140 ms | 0.12682 ms | **+3.49%** |
| 109-case total | 109 | 5.84273 ms | 5.72065 ms | **+2.04%** |

All seven aggregate samples improved: `+2.0435%, +1.9496%, +2.1887%,
+2.0655%, +3.3272%, +1.9164%, +0.4502%`. The changed S2 family improved in
all seven samples by `+2.4467%` to `+3.6146%`, and all three S2 medians
improved.

Only the S2 family result is attributed to this change. The larger full-run
difference includes positive device-state drift in unchanged kernels.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`c637c57e2478d9112a789e7ba6f16462c05d412df2190c9fa7e6739ca9fa1b2a`.
