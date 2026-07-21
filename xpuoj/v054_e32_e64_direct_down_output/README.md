# XPUOJ v054: E32/E64 Direct Down Output

## Objective

Extend v053's direct down-projection output placement to E32, removing the
workspace write and copy for every expert with at least one full 128-row
block.

## Change

The E32 and E64 paths now build a down pointer table.  A large expert's down
GEMM writes directly into that expert's padded segment of the caller's `out`;
an expert with fewer than 128 valid rows uses the private down workspace and
the guarded unpack fallback.  The E32 two-chunk unpack was changed to skip
large experts, matching the existing E64 four-chunk behavior.  Invalid
activation rows are cleared before direct writes.

E16 is unchanged.  Every invocation still reads the current activation,
weights, route, and metadata, executes FC1/SwiGLU/down, and writes the current
output.  No input-dependent result or output buffer is cached.

## Correctness

- The v053 comparison passed for all three official proxy cases with
  `max_abs=0` on valid and padded rows.
- FP32 oracle errors remained `0.000570536` for `(2048, 8192)` and
  `0.000950813` for `(7168, 2048)`.
- Two different activation/route/output graph keys were captured and replayed
  alternately; both E32 outputs matched v053 bit-for-bit.
- After graph capture, swapping E32's smallest and largest expert counts and
  updating all metadata in place still matched v053 bit-for-bit; padded rows
  remained zero.
- Static submission check returned `status: pass`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Single-module processes used ten warmups, thirty calls per
sample, and twelve samples. E32 was repeated in the reverse process order to
remove module-slot bias; reported E32 values are geometric means.

| Case | v053 corrected | v054 corrected | Improvement |
| --- | ---: | ---: | ---: |
| E16 | 1.91272 ms | 1.91031 ms | +0.13% |
| E32 | 3.01670 ms | 2.94667 ms | **+2.32%** |
| E64 | 6.01079 ms | 6.00423 ms | +0.11% |
| total | 10.94022 ms | 10.86121 ms | **+0.72%** |

The E16/E64 differences are within normal process variation; the E32 gain is
stable across both independent process orders.

Decision: **accepted as the local full-compute baseline; not submitted to
XPUOJ**.

Submission SHA-256:
`1b61e74216ce23de57f1b9f63b9f9632ceb481dcebdb8a221f473dea00032b29`.
