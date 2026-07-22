# XPUOJ v058: E32/E64 Prepare Fusion

## Objective

Reduce E32/E64 launch overhead by combining the remaining per-invocation
metadata preparation without changing any GEMM, SwiGLU, routing, or output
semantics.

## Change

E32/E64 now use one TileLang kernel to perform three operations that v057
launched separately:

- conditionally safe-pack a final expert with fewer than 128 rows;
- build the FC1 device pointer table;
- build the down-projection device pointer table.

The fused kernel launches one 256-thread block per 256 hidden columns.  The
first block builds both pointer tables while all blocks participate in the
guarded final-expert copy when it is required.  The E16 path is unchanged.

Every invocation still reads the current activation, weights, routing
metadata, route weights, and output pointer.  All pointer tables and computed
results are rebuilt; no input-dependent result or completed output is cached.

## Correctness

- All three official proxies matched v057 bit-for-bit on valid elements and
  kept every padded element at zero.
- The FP32 oracle checks passed with maximum absolute errors `0.000570536` for
  `(2048, 8192)` and `0.000950813` for `(7168, 2048)`.
- After graph capture, E32 and E64 metadata was changed in place to move a
  119-row expert to the final position.  Both cases remained bit-identical to
  v057, covering the conditional safe-pack path with current metadata.
- The static submission check returned `status: pass`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.  Each timing slot used ten warmups and thirty calls.  E32 was
repeated with ten samples in both module-load orders.  Both modules wrote to
the same output address during timing; geometric means remove the persistent
penalty on the module loaded second.

| E32 load order | v057 | v058 |
| --- | ---: | ---: |
| v057 first | 2.969178 ms | 3.044742 ms |
| v058 first | 3.077013 ms | 2.979253 ms |
| geometric correction | 3.022615 ms | 3.011820 ms |

The corrected E32 improvement is **+0.357%**.  The corresponding E64
bidirectional measurement was `5.930223 -> 5.925370 ms`, or **+0.082%**.
Using v057's unchanged E16 measurement gives an estimated three-case total of
`10.855528 -> 10.839880 ms`, or **+0.144%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`eee7bba5a77b778ff7a818976cf910abab7ff43d9903db3734483f7161272bc3`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
