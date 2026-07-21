# XPUOJ v053: E64 Direct Down Output

## Objective

Remove the final copy from the E64 down-projection path for experts whose
group has at least one full 128-row block, while preserving the v052
full-recomputation semantics.

## Change

The E64 path builds a device pointer table for the down batched GEMM.  For a
large expert, the GEMM output pointer targets that expert's padded segment in
the caller-provided `out` tensor directly.  Experts with fewer than 128 valid
rows continue to use a private down workspace and the existing guarded unpack
copy.  The E64 SwiGLU kernel explicitly clears invalid rows before a direct
write, so padded output rows remain zero.  E32 and E16 retain the v052 path.

This is a normal pointer-batched GEMM and output-placement optimization.  Each
call still repacks the current activation, consumes the current route and
metadata, executes FC1/SwiGLU/down, and writes the current output.  It does not
cache input-dependent results, inspect input values, or branch on tensor
identity.

## Correctness

- v052 comparison passed for all three official proxy cases; valid and padded
  rows were bit-identical (`max_abs=0`).
- FP32 oracle errors were unchanged: `0.000570536` for `(2048, 8192)` and
  `0.000950813` for `(7168, 2048)`.
- Metadata permutation and alternating CUDA-graph replay tests passed for E32
  and E64 with independent activation, route, metadata, and output tensors.
- Static submission check returned `status: pass`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each direction used ten warmups, thirty calls per sample, and
twelve symmetric paired samples. The two directions reverse module load order
to correct the measurable allocation-slot bias.

| Case | v052 corrected | v053 corrected | Improvement |
| --- | ---: | ---: | ---: |
| E16 | 1.92433 ms | 1.92447 ms | -0.01% |
| E32 | 3.11403 ms | 3.11257 ms | +0.05% |
| E64 | 6.08493 ms | 5.94559 ms | +2.29% |
| total | 11.12330 ms | 10.98263 ms | **+1.26%** |

The E64 gain is stable in both load orders. The E32 and E16 differences are
within measurement noise after correction.

Decision: **accepted as the local full-compute baseline; not submitted to
XPUOJ**.

Submission SHA-256:
`484dc09c4d71b5832707b32ada5f3479acdbb5729711a572982b347721ca4d4c`.
