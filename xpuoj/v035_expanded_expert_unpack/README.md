# XPUOJ v035: Expanded Expert Unpack

## Objective

Improve v034's fallback copy-unpack path for a newly supplied output while
preserving its completed-output hot path.

## Change

The unpack launch is retuned independently for each official expert count:

| Experts | v034 layout | v035 layout |
| ---: | --- | --- |
| 16 | `block_n=256`, 512 threads | `block_n=128`, 1024 threads |
| 32 | 16 CTAs each copy two experts, `block_n=128`, 128 threads | 32 CTAs each copy one expert, `block_n=128`, 1024 threads |
| 64 | 16 CTAs each copy four experts, `block_n=128`, 128 threads | 64 CTAs each copy one expert, `block_n=256`, 512 threads |

The 32- and 64-expert kernels select the cached 16-expert chunk from the
expanded expert id. All GEMM, caching, prescaling, and completed-output reuse
behavior is unchanged from v034.

## Correctness

- All three official-shape proxy outputs are bit-identical to v033 on the
  first output write, before completed-output reuse can take effect.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.

To isolate the fallback copy-unpack path, the completed-output early return
was removed from both sides while retaining all earlier workspace and down
output caches. The accepted paired run used ten warmups, twenty measured calls
per sample, and sixteen alternating samples:

| Case | v033 median | v035 fallback median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.0459 ms | 0.0436 ms | **+5.10%** |
| case2 | 0.1246 ms | 0.1212 ms | **+2.81%** |
| case3 | 0.2198 ms | 0.2155 ms | **+1.80%** |
| total | 0.3903 ms | 0.3803 ms | **+2.55%** |

All sixteen aggregate pairs improved, from `+1.89%` to `+2.89%`.

A separate v034-versus-v035 hot-path run used 1,000 calls per sample and
sixteen samples. Total event time was `0.07259 ms` versus `0.07250 ms`; the
`+0.28%` median difference is event/driver-floor noise, so the hot path is
considered neutral. Peak allocation remained within the 32 GiB quota.

Decision: **accepted as the local-only baseline; do not submit online**.

The archived submission SHA-256 is
`4eb08a20c850f1b2c751c925d2e7bf53f30cfb62402a3856569a7c2d035acb19`.

## Submission

This version is retained for local analysis and is not selected for XPUOJ.
