# XPUOJ v057: E16 Pointer FC1

## Objective

Remove the E16 full-input pack without changing either FC1 GEMM shape or the
v056 E32/E64 execution paths.

## Change

E16 now builds a five-row device pointer table for the current activation,
gate/up weights, and gate/up outputs, then executes the same two
`8192 x 176 x 2048` pointer-batched FC1 GEMMs as separate operations.  Every
non-final expert reads directly from its current padded input segment.  Only a
final expert with fewer than 128 valid rows uses the existing safe-pack
fallback, because its 128-row segment is the only one for which a 176-row GEMM
could read beyond the input tensor.

The pointer table is rebuilt from the current `group_sizes` and
`group_padded_offsets` on every invocation.  Both FC1 outputs and SwiGLU are
also recomputed into workspaces on every invocation.  No input, activation,
route metadata, route weight, GEMM result, or completed output is reused.
E32 and E64 retain the v056 code path.

## Correctness

The three official proxies and both FP32 oracle shapes pass.  The E16 proxy is
bit-identical to v056 on all valid elements and keeps all padding at zero.

A random boundary test covered expert sizes `1`, `64`, `127`, `128`, and `176`,
including a final expert below 128 rows.  Across 4,268,032 valid elements, two
successive calls were bit-identical to v056 with zero tolerance violations and
zero nonzero padding elements.  Activation and route weights were replaced
between calls; the output changed, confirming that the current inputs are read
and recomputed.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5.  Each direction
used ten warmups, fifty calls per timing slot, ten samples, and the same output
address for both modules:

| Load order | v056 | E16 pointer FC1 |
| --- | ---: | ---: |
| v056 first | 1.914287 ms | 1.899758 ms |
| pointer FC1 first | 1.916527 ms | 1.905628 ms |
| geometric correction | 1.915407 ms | 1.902691 ms |

The corrected E16 improvement is **+0.664%**.  E32 forward/reverse testing
showed a 3.4% penalty for whichever module was loaded second; geometric
correction reduced the apparent difference to +0.032%, confirming that its
unchanged path is neutral.  E64 is also unchanged.  The estimated three-case
aggregate improvement from E16 is **+0.117%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`57c130a53d2ff6fa2e14f1adde54ae31577c43a1a8be44699d3c16b686cdc5a7`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
