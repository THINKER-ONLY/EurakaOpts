# MLA Decode v005: Adaptive Split Policy

## Objective

Tune context splitting across all 31 published MLA race shapes instead of using
one split count for a broad batch range.

## Change

- batch 1 keeps split 16;
- batch 2 uses split 8 at 2K, split 32 at 8K, and split 64 at 16K and above;
- batch 4 uses split 32 for published contexts;
- batch 8 uses split 16;
- batch 16 keeps the v004 split 8 policy;
- batch 32 and larger keep split 4.

The aggressive branches require the context to divide into whole `block_n=32`
tiles. Other shapes fall back to the conservative v004 policy. The
output-quarter kernel and numerical operations are unchanged.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00038418` and `0.00028522`.
- All 31 published batch/context shapes match v004 with
  `atol=1e-2, rtol=1e-2`.
- The largest observed v004/v005 difference is `0.00042724609375`.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples over all 31 published
shapes:

| Batch/context family | Median paired improvement over v004 |
| --- | ---: |
| B2, 8K / 16K / 32K / 64K | **+14.67% / +24.11% / +30.10% / +33.17%** |
| B4, 2K / 8K / 16K / 32K / 64K | **+11.08% / +28.09% / +32.18% / +34.25% / +35.43%** |
| B8, 2K / 8K / 16K / 32K / 64K | **+7.65% / +13.15% / +14.82% / +15.48% / +15.57%** |
| 31-case total | **50.2704 ms -> 46.9233 ms (+6.64%)** |

All five aggregate samples improved: `+6.5905%, +6.6267%, +6.6617%,
+6.6370%, +6.6671%`.

The search also rejected unsafe or slower points: a fixed split32 fails when a
test context is smaller than one `block_n` per split; batch4 split64 regressed
2K by about 21%; batch8 split4 regressed about 31%; and batch8 split32 was
slower than split16.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`da2908c17c2734965e625e53d7b7109d1f8c87cf82c1642619b1b7bcf740f019`.
