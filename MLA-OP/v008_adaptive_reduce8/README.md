# MLA Decode v008: Adaptive Eight-Way Reduce

## Objective

Increase split-K merge parallelism where v007's four 128-column reduction CTAs
do not fully occupy the MetaX C500.

## Change

- retain the four 128-column main-attention output partitions;
- use eight 64-column reduction CTAs per head when `batch <= 4` or
  `seqlen_kv <= 2048`;
- retain v007's four 128-column reduction CTAs for all other shapes;
- retain 64 threads per reduction CTA, the same LSE normalization, main
  attention kernel, and adaptive context split policy.

The selected low-batch and short-context shapes gain merge grid parallelism
without increasing QK or PV work. High-batch and longer-context shapes keep the
v007 layout because their merge cost is already hidden by the main kernel.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00038418` and `0.00028522`.
- All 31 published MLA shapes match v007 with `atol=1e-2, rtol=1e-2`.
- The largest observed v007/v008 absolute difference is
  `0.00006103515625`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 published shapes:

| Shape/family | Median paired improvement over v007 |
| --- | ---: |
| B1, context 2K / 4K / 8K | **+2.87% / +2.95% / +2.62%** |
| B1, context 16K / 32K / 64K | **+1.35% / +0.79% / +0.42%** |
| B2, context 2K / 8K / 16K | **+2.72% / +2.28% / +1.89%** |
| B4, context 2K / 8K / 16K | **+1.28% / +0.76% / +0.40%** |
| 31-case total | **47.2171 ms -> 47.1290 ms (+0.183%)** |

All five aggregate samples improved: `+0.1948%, +0.1882%, +0.1833%,
+0.1492%, +0.1194%`. An earlier full run measured `+0.257%` median with one
negative aggregate sample caused by a `-1.19%` outlier on an unchanged
B32/64K kernel; the accepted repeat removed that outlier and confirmed the
selected-shape improvements.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`567bb7984211b8f1ddc2fdf0f2d124746e6c4055aceb10be5442385a008b3956`.
