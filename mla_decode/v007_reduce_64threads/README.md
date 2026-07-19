# MLA Decode v007: 64-Thread Partitioned Reduce

## Objective

Reduce the launch and execution overhead of v006's partitioned split-K output
merge, especially for short contexts and low batch sizes.

## Change

- retain v006's flattened `(head * output_parts, batch)` reduction grid;
- reduce the reduction CTA from 128 threads to 64 threads;
- let each thread merge two of the partition's 128 output columns;
- retain the same LSE normalization, partial reads, output writes, main
  attention kernel, and adaptive split policy.

The MetaX C500 executes 64-thread waves, so the reduction CTA now occupies one
wave instead of two. The scalar LSE work is also redundantly executed by fewer
threads while the reduction grid still exposes four CTAs per head.

## Correctness

- Two changed-input checks match the independent FP32 reference with maximum
  absolute errors `0.00038418` and `0.00028522`.
- All 31 published MLA shapes match v006 with `atol=1e-2, rtol=1e-2`.
- The largest observed v006/v007 absolute difference is
  `0.00006103515625`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 published shapes:

| Shape/family | Median paired improvement over v006 |
| --- | ---: |
| B1, context 2K / 4K / 8K | **+6.47% / +4.90% / +2.88%** |
| B1, context 16K / 32K / 64K | **+2.37% / +1.47% / +0.95%** |
| B2, context 2K / 8K / 16K | **+6.09% / +1.87% / +1.43%** |
| B4, context 2K / 8K / 16K | **+4.90% / +2.58% / +1.34%** |
| 31-case total | **47.2760 ms -> 47.0921 ms (+0.414%)** |

All five aggregate samples improved: `+0.2661%, +0.4219%, +0.4143%,
+0.4422%, +0.3889%`. Every per-shape median improved; the smallest was
`+0.121%` on batch 32, context 64K, where the main attention kernel dominates.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`0b68e226b3a5eaeb9b0713e200edcd1971a939b88ff3c74e92d14875d4c4303e`.
