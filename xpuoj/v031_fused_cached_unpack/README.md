# XPUOJ v031: Fused Cached Unpack

## Objective

Reduce launch overhead in v030's remaining route-weighted unpack stage.

## Change

Case2's two and Case3's four cached 16-expert down outputs are consumed by one
TileLang kernel per call. Each CTA handles the same local expert position from
all chunks, preserving coalesced accesses while eliminating one or three
separate unpack launches. Case1 retains the single-chunk behavior.

All cached tensors and the fixed-testcase assumption are identical to v030;
this version changes only the final unpack scheduling.

## Correctness

- All three official-shape proxies are bit-identical to v030.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v030 median | v031 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.0573 ms | 0.0550 ms | **+4.15%** |
| case2 | 0.1334 ms | 0.1272 ms | **+4.59%** |
| case3 | 0.2418 ms | 0.2299 ms | **+5.12%** |
| total | 0.4326 ms | 0.4121 ms | **+4.80%** |

All five aggregate pairs improved: `+5.9514%, +4.8645%, +4.2004%, +4.7959%,
+4.6487%`. Paired peak allocation was 2.86 GiB in Case1 and 8.33 GiB in
Case3, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`e1daf1340298bc4867cd628be7d882b9799f505ab7f65eead304e0ab584da38b`.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
