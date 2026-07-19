# MLA Decode v023: Adaptive Num Split

## Objective

Recover missing C500 occupancy in the official low-batch and long-context
shapes by tuning the sequence split count per `(batch, kv_ctx)` pair.

## Change

- retain the v022 kernels and numerical algorithm unchanged;
- override `num_split` only for 23 measured official shapes;
- increase splitting on long contexts where the original launch did not expose
  enough independent work to the device;
- reduce `B4/2K` from 32 splits to 16, where reduction overhead dominated;
- retain the v022 policy for unmeasured shapes and for the eight official
  shapes without a stable alternative.

The selected policy changes are:

| Batch | Context and split changes |
| ---: | --- |
| 1 | 16K: 16 -> 64; 32K/64K: 16 -> 128 |
| 2 | 32K: 64 -> 128; 64K: 64 -> 256 |
| 4 | 2K: 32 -> 16; 16K: 32 -> 64; 32K/64K: 32 -> 128 |
| 8 | 8K: 16 -> 32; 16K/32K/64K: 16 -> 64 |
| 16 | 2K: 8 -> 16; 8K/16K/32K/64K: 8 -> 32 |
| 32 | 2K: 4 -> 8; 8K: 4 -> 16; 16K/32K/64K: 4 -> 32 |

## Correctness

- All 31 candidate outputs match v022 within the official `rtol=2e-3,
  atol=2e-3` tolerance; the maximum absolute difference is `0.00061035`.
- Independent FP32-reference checks pass for `B32/2K`, `B4/2K`, `B1/64K`,
  and `B16/64K`. Their maximum absolute errors are respectively `0.00042082`,
  `0.00023936`, `0.00006336`, and `0.00017324`.
- The standard changed-input correctness gate retains maximum absolute errors
  below `0.000385`.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and seven samples over all 31 official shapes:

| Shape group | Cases | v022 total | v023 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Changed split policy | 23 | 23.90968 ms | 18.35254 ms | **+23.27%** |
| 31-case total | 31 | 24.70426 ms | 19.12279 ms | **+22.62%** |

All seven aggregate samples improved: `+22.5009%, +22.6761%, +22.6177%,
+22.6194%, +22.4994%, +22.6010%, +22.6306%`. The changed-family aggregate
improved in all seven samples by `+23.1502%` to `+23.3309%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`47594e9b2e5203325bb2b6cc827e163411ebe6e3876c0d8ccadf8075e7af1b8d`.
