# NSA v005: Gather Two Blocks

## Objective

Specialize the published `S=2, block_size=16` shape family by replacing two
small attention iterations with one 32-position attention tile.

## Change

- gather the two K blocks into one `[32, D]` shared tile;
- apply invalid-block and causal masks to the corresponding score regions;
- run one QK GEMM, one softmax, and one PV GEMM across both blocks;
- overwrite the K shared tile with gathered V after QK, preserving v004's
  explicit K/V lifetime reuse;
- dispatch the gathered 128-thread kernel only for
  `S == 2, block_size == 16, D >= 64`.

Duplicate block indices retain duplicate attention positions, and invalid
indices remain masked. Every call reads its current tensors and block indices;
only compiled kernels are cached.

## Correctness

- The D64/S2 independent FP32 checks on two changed input sets have maximum
  absolute errors `0.00150776` and `0.00143313`.
- The gathered output matches v004 with `atol=1e-2, rtol=1e-2`; maximum observed
  difference is `0.0009765625`.
- S1, S4, S8, and the 64K/S16 stress proxy remain on the v004 path and match it
  exactly.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted official-shape proxy used five warmups,
twenty alternating measured calls per sample, and five samples:

| Case | v004 median | v005 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| B4, L1024, D64, S2, BS16 | 0.0605 ms | 0.0568 ms | **+6.12%** |

All five samples improved: `+6.1760%, +6.5273%, +6.1229%, +6.0580%,
+5.9894%`. A separate five-sample run measured `+5.51%`, confirming the result.
The six-case regression total, dominated by the non-official 64K/S16 stress
case, improved by `+0.08%` with no measured regression.

Using 64 instead of 128 threads regressed the S2 proxy by `5.17%`, so it was
rejected. Gathering S4 and S8 was also rejected after regressions of about
`13.5%` and `150%`, respectively.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`0d20f63c483eb9e36afa18d3f8b61178279047c6fa04d08f7edb4ab4874d7854`.
