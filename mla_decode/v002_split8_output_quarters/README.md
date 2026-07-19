# MLA Decode v002: Split-8 Output Quarters

## Objective

Expose enough independent work to occupy the C500 on MLA shapes with only 16
query heads, while keeping every API call dependent on its current inputs.

## Change

The baseline launches one attention CTA per batch item. This version applies
two compile-time decompositions:

- split the KV context into eight independent online-softmax partitions;
- split the 512-column output into four 128-column partitions.

Each output partition recomputes QK and online softmax, but only accumulates its
own PV columns. The smaller accumulator avoids the MACA register spill that
made the official split kernel exceed the 64 KiB shared-memory limit. A second
kernel combines the eight partial softmax states and output slices.

The implementation only caches compiled kernels. It does not cache Q, Q-PE,
KV, K-PE, partial results, or completed outputs across calls.

## Correctness

- Two independently generated input sets at the same shape match the FP32
  reference with maximum absolute errors `0.00046025` and `0.00028783`.
- All five performance proxies match v001 with `atol=1e-2, rtol=1e-2`.
- The largest observed v001/v002 difference is `0.00018310546875`.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted paired run used five warmups, ten
alternating measured calls per sample, and five samples:

| Case | v001 median | v002 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| batch 1, context 8192 | 1.5073 ms | 0.1770 ms | **+88.26%** |
| batch 8, context 8192 | 1.8525 ms | 0.4824 ms | **+73.97%** |
| batch 32, context 8192 | 1.8795 ms | 1.5684 ms | **+16.46%** |
| batch 1, context 65536 | 11.8375 ms | 1.1729 ms | **+90.09%** |
| batch 32, context 65536 | 14.8078 ms | 12.3051 ms | **+16.90%** |
| proxy total | 31.8846 ms | 15.7058 ms | **+50.75%** |

All five aggregate samples improved: `+50.7456%, +50.7300%, +50.7597%,
+50.6506%, +50.7477%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`b49981f57acc83743f8750b2c94ff951b5d1aec0dc68ee2c130ad074d6c60fd6`.
