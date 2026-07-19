# MLA Decode v012: Prefix-Scaled Weights

## Objective

Remove v011's standalone logits-normalization kernel and its full read/write
pass over the materialized score tensor.

## Change

- the QK plus output-part-0 kernel now stores FP16 exponential weights
  relative to the running prefix maximum for each 16-head by 32-token tile;
- store one FP16 scaled prefix maximum per head and score tile;
- remove the standalone normalization kernel;
- in output parts 1-3, combine the saved prefix maximum with the split LSE to
  apply one scale per head, then multiply the score tile before the PV GEMM;
- retain v011's direct path, split policy, output partitioning, materialization
  threshold, and final split-K reduction.

For the largest official shape, the existing materialized score tensor is 64
MiB and the new prefix-maximum metadata is 2 MiB. Both are invocation-local,
are recomputed from current inputs, and are not cached across calls.

## Correctness

- All 31 official outputs match v011 with maximum absolute difference
  `0.00036621` and maximum mean absolute difference `0.00002565`.
- Direct comparisons against the PyTorch reference using the official
  `rtol=2e-3, atol=2e-3` threshold pass for B4/2K and B32/2K. Their maximum
  absolute errors are `0.00017986` and `0.00051752`, respectively.
- The standard two-input correctness checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape/group | v011 | v012 | Improvement |
| --- | ---: | ---: | ---: |
| B2, context 8K | 0.11738 ms | 0.10670 ms | **+9.08%** |
| B4, context 2K | 0.08906 ms | 0.07990 ms | **+10.21%** |
| B8, context 2K | 0.11359 ms | 0.10324 ms | **+8.53%** |
| B32, context 32K | 3.78875 ms | 3.44443 ms | **+9.09%** |
| B32, context 64K | 7.24659 ms | 6.82319 ms | **+5.82%** |
| All B2 shapes | 1.16736 ms | 1.07725 ms | **+7.72%** |
| All B4 shapes | 2.01001 ms | 1.85966 ms | **+7.48%** |
| All B8 shapes | 3.67711 ms | 3.41842 ms | **+7.04%** |
| All B16 shapes | 7.07282 ms | 6.59876 ms | **+6.70%** |
| All B32 shapes | 14.58831 ms | 13.55087 ms | **+7.11%** |
| 31-case total | 29.88308 ms | 27.85784 ms | **+6.78%** |

All five aggregate samples improved: `+6.4230%, +6.7815%, +6.8458%,
+6.8015%, +6.7638%`. Every per-shape median was non-negative; materialized
shapes improved by approximately 5.8% to 10.2%.

Decision: **accepted as the local baseline**. The two requested v001 online
probes were canceled while still pending; v012 was not submitted online.

The archived submission SHA-256 is
`a67fd99d352a8b79bab779924b9dc84b40fc00a72411caeb8983fef87ff8a778`.
