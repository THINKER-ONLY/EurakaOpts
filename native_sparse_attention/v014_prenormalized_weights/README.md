# NSA v014: Prenormalized Weights

## Objective

Reduce normalization cost in the 100 official single-block cases and the three
gathered two-block cases.

## Change

- after the score exponential and reduction, compute one reciprocal softmax
  sum per query head;
- multiply the score fragment by that reciprocal before casting the weights to
  FP16 and running the PV GEMM;
- remove the elementwise division over the FP32 output fragment;
- apply the same transformation to the gathered S2 kernel, where all selected
  scores are available before the single PV GEMM;
- retain v013's thread policy, GEMM policies, historical-block clearing,
  causal masks, shared-memory reuse, and S4/S8 online-softmax path.

For S1 this replaces `G * D` output divisions with `G` reciprocal operations
and `G * block_size` weight multiplications. For gathered S2 it replaces
`G * D` output divisions with `G` reciprocals and `G * 2 * block_size`
multiplications.

## Correctness

- All 109 official outputs match v013 with maximum absolute difference
  `0.00195313` and maximum mean absolute difference `0.00007662`, within the
  official `rtol=1e-2, atol=1e-2` threshold.
- Direct PyTorch-reference checks pass for D32/BS16, D128/BS16, and D64/BS64
  S1 cases. Their maximum absolute errors are `0.00095367`, `0.00096989`, and
  `0.00095272`, respectively.
- The standard gathered-S2 two-input checks improve from v013's maximum errors
  `0.00150776` and `0.00143313` to `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v013 total | v014 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| S1 | 100 | 5.24695 ms | 5.11270 ms | **+2.56%** |
| S2 | 3 | 0.12838 ms | 0.12383 ms | **+3.55%** |
| D32 | 33 | 1.34443 ms | 1.30400 ms | **+3.01%** |
| D64 | 45 | 2.41228 ms | 2.35228 ms | **+2.49%** |
| D128 | 31 | 2.02711 ms | 1.98177 ms | **+2.24%** |
| BS16 | 94 | 4.95168 ms | 4.82515 ms | **+2.56%** |
| BS32 | 11 | 0.68732 ms | 0.67336 ms | **+2.03%** |
| BS64 | 4 | 0.14482 ms | 0.13955 ms | **+3.64%** |
| 109-case total | 109 | 5.78382 ms | 5.63805 ms | **+2.48%** |

All five aggregate samples improved: `+2.4100%, +2.4760%, +2.6953%,
+3.4044%, +2.3184%`. No per-shape median regressed. The unchanged S4/S8
paths remained positive in the paired full run.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`f7d3c612a1058b99fb0240a337bf0d408ad3360297f88939c1924b2740da60f7`.
