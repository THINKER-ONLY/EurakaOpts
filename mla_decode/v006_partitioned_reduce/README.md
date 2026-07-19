# MLA Decode v006: Partitioned Reduce

## Objective

Reduce the low-batch tail latency of the split-K output merge after v005's
adaptive main-kernel split policy.

## Change

- flatten the reduction grid from `(head, batch)` to
  `(head * output_parts, batch)`;
- let each reduction CTA merge one 128-column output partition instead of all
  512 columns;
- reduce partial/output fragment sizes from 512 to 128 elements per CTA;
- retain the same LSE normalization, partial reads, output writes, main
  attention kernel, and v005 adaptive split policy.

LSE scalar work is repeated for each of the four partitions, but the output
merge exposes four times as many CTAs and uses fewer registers per CTA.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00038418` and `0.00028522`.
- All 31 published MLA shapes are bitwise identical to v005.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 published shapes:

| Shape/family | Median paired improvement over v005 |
| --- | ---: |
| B1, context 2K / 4K / 8K | **+7.19% / +5.35% / +2.87%** |
| B1, context 16K / 32K / 64K | **+2.97% / +0.92% / +0.90%** |
| B2, context 2K / 8K / 16K | **+5.05% / +3.53% / +2.78%** |
| B2, context 32K / 64K | **+1.61% / +1.06%** |
| 31-case total | **47.2053 ms -> 47.0856 ms (+0.273%)** |

All five aggregate samples improved: `+0.1540%, +0.2919%, +0.2861%,
+0.2733%, +0.1938%`. High-batch cases are dominated by the main attention
kernel and remain within sub-percent event-timing noise.

A three-dimensional reduction grid was rejected because the TileLang frontend
cannot register a second `blockIdx.z` launch axis in the same prim_func; the
accepted two-dimensional flattened grid is equivalent and compiles cleanly.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`d03c04f5162d30ef179bd738676f89e54821819c84d9344033a4054e4a7c61e8`.
