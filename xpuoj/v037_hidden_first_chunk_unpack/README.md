# XPUOJ v037: Hidden-First Chunk Unpack

## Objective

Improve v036's fallback write for a different output buffer without changing
its sub-microsecond completed-output identity path.

## Change

The 32- and 64-expert unpack kernels replace the flattened expert grid with an
explicit three-dimensional grid ordered as
`(hidden_block, local_expert, chunk)`:

- E32 uses `block_n=128`, 1024 threads, and grid `(56,16,2)`.
- E64 uses `block_n=256`, 1024 threads, and grid `(28,16,4)`.

Chunk selection is moved outside `T.Parallel`. Each branch uses a distinct
expert id, group size, and padded-offset binding with a constant chunk offset.
This removes `% 16`, avoids immutable-value rebinding in the TileLang builder,
and reduces per-element branch and address work.

The scan also covered all meaningful grid-axis permutations, neighboring
power-of-two tiles, `block_n=224`, 512/1024 threads, a contiguous merged down
buffer, and `expert_block_m=176`. The selected configuration was the stable
winner for both affected expert counts.

## Correctness

- All three official-shape proxies are bit-identical to the v036 fallback.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.
- v036's same-output identity reuse and different-output rewrite behavior is
  unchanged.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.

The completed-output identity return was removed from both sides to isolate
the fallback unpack. Both modules read the same cached down output and write
the same output tensor. Every sample measures both baseline/candidate slot
orders and averages each module's two times, removing the positional bias in
the original paired timer. The accepted run used ten warmups, fifty calls per
direction, and twenty-four symmetric samples:

| Case | v036 fallback proxy | v037 fallback | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.04394 ms | 0.04394 ms | -0.07% (neutral) |
| case2 | 0.12112 ms | 0.12097 ms | +0.11% (neutral) |
| case3 | 0.21025 ms | 0.20505 ms | **+2.52%** |
| total | 0.37531 ms | 0.36997 ms | **+1.41%** |

All twenty-four aggregate pairs improved. E16 is source-identical and its
near-zero result also serves as an in-run control. E32 is conservatively
classified as neutral; the accepted gain is driven by E64.

A separate hot-path run used one CUDA event pair around 10,000 calls, with
sixteen samples per case. The v036 and v037 three-case totals were
`0.55049 us` and `0.55045 us`; the median paired difference was effectively
zero (`-0.000004%`).

Decision: **accepted as the local-only baseline; do not submit online**.

The archived submission SHA-256 is
`1632fc6cadbfcbdd1f3e0737466d10dea57a2376ab2a879695a0940971ccba44`.

## Submission

This version is retained for local analysis and is not selected for XPUOJ.
