# NSA v003: Serial Shared Reuse

## Objective

Reduce shared-memory pressure and pipeline overhead in the sparse-block loop,
especially for the 64K sequence stress case.

## Change

- use 256 threads for `D >= 64, block_size >= 64`;
- express the selected-block loop as `T.serial`;
- reuse the K shared tile as the V tile after QK completes;
- write the output fragment directly to the caller's output buffer.

The serial loop is required for the K/V alias: TileLang's pipeline planner
correctly rejects overlapping K and V writes inside `T.Pipelined`. QK consumes
the K tile before V overwrites it, and PV consumes V before the next iteration
loads K, so the explicit serial order preserves the algorithm.

Every call reads its current Q, K, V, block indices, and output buffer. Only
compiled kernels are cached.

## Correctness

- The standard two-input-set gate matches the FP32 reference with maximum
  absolute errors `0.00159550` and `0.00134659`.
- All five proxies match v002 with `atol=1e-2, rtol=1e-2`.
- A separate `D128/BS64` test, which v002 cannot launch due to 71,680 bytes of
  shared memory, passes in v003 with maximum errors `0.00169826` and
  `0.00154328` on two changed input sets.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples:

| Case | v002 median | v003 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| short S1: B1, L1024, D64, BS16 | 0.0380 ms | 0.0317 ms | **+16.34%** |
| long S1: B1, L16384, D64, BS16 | 0.1926 ms | 0.1152 ms | **+40.10%** |
| multi S4: B4, L1024, D64, BS16 | 0.1302 ms | 0.0818 ms | **+37.20%** |
| wide: B2, L1024, D128, BS32 | 0.1774 ms | 0.0694 ms | **+60.48%** |
| 64K stress: B1, L65536, D64, S16, BS64 | 49.6076 ms | 12.4069 ms | **+74.99%** |
| proxy total | 50.1457 ms | 12.7051 ms | **+74.66%** |

All five aggregate samples improved: `+74.6613%, +74.6630%, +74.6605%,
+74.6654%, +74.6564%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`6529d3f166fb5ce6632492fa5190d542324f110eb5888b6ec9b73103963a956b`.
