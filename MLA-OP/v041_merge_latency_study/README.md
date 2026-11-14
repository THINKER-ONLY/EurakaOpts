# v041 — 2048 merge-latency study (2026-11-13)

Live best is still v960 (submission 168534, 97.67, rank 2). This dir is diagnostics.

Judge per-case for 168534: 2048 .015/96, 4096 .020/97, 8192 .029/98,
16384 .040/98, 32768 .064/98, 65536 .111/99.

Closest threshold is 2048 (needs <= 12.96us, now 15 -> -2.04us).

v960 @2048: wall 16.13us = fixed 9.6 + 1 tile 6.5.
Ablations (wall deltas): split-merge 3.02 (partial reads 1.59, LSE reduce 0.88,
spin 0.43), KV copy 1.26, QK 1.27, partial write 0.76, Q load 0.74.
One extra tile costs 6.5us: split=32 -> 22.7, split=48 -> 23.6. So 64 splits
(exactly one tile per CTA) is optimal and fewer splits is strictly worse.

v963_mergeearly.py issues the 32-deep partial-sum loads before the LSE
reduction so both post-spin memory chains share one latency: -0.34us at 2048.
Not enough to move a display, so it was NOT submitted.

Conclusion: 2048 cannot reach 12.96 without removing split-K entirely, which
would remove all parallelism. Remaining gaps at 4096 (-3.90) and 32768 (-3.82)
are likewise structural.
