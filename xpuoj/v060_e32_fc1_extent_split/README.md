# XPUOJ v060: E32 Split FC1 Extent

## Objective

Reduce E32's invalid-row FC1 work while preserving v059's complete per-call
computation and support for current route metadata.

## Change

The disclosed E32 proxy's first 16 experts contain at most 157 valid rows,
while the second half contains the 170-row maximum.  v060 therefore changes
the two pointer-batched combined FC1 calls from `M170/M170` to `M158/M170`.

A guarded TileLang fallback runs on the first FC1 stream after the `M158`
mcBLAS call.  It reads the current `group_sizes` and recomputes rows
`[158, group_size)` whenever an expert in the first half grows past the main
extent.  In the standard proxy all 32 fallback column blocks only perform the
metadata check.  The fallback covers the same global `M170` contract as v059;
it does not discard rows or assume that metadata remains unchanged.

E16, E64, SwiGLU, down projection, direct output placement, routing, and graph
behavior are unchanged.  Every invocation still reads the current inputs,
weights, metadata, route weights, and output pointer and recomputes the full
result.

## Correctness

- All three official proxies matched v059 bit-for-bit on valid and padded
  elements.
- FP32 oracle errors remained `0.000570536` for `(2048, 8192)` and
  `0.000950813` for `(7168, 2048)`.
- After graph capture, the 170-row E32 expert was exchanged with expert 0 so
  that the fallback computed twelve tail rows.  v060 remained bit-identical to
  v059, and every padded output element remained zero.
- The static submission check returned `status: pass` with maximum absolute
  error `0.001953125`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.  Each direction used ten warmups, thirty calls per timing
sample, and nine paired samples.  The two directions reversed module
initialization order to remove the known multi-stream allocation bias.

| Load order | v059 | v060 |
| --- | ---: | ---: |
| v059 first | 2.978935 ms | 3.044898 ms |
| v060 first | 3.061402 ms | 2.975181 ms |
| geometric correction | 3.019887 ms | 3.009838 ms |

The corrected E32 improvement is **+0.333%**.  Applying the unchanged E16 and
E64 measurements from v059 gives an estimated three-case total of
`10.830515 -> 10.820466 ms`, or **+0.093%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`cf31769de78788a26a8f4e6c7907a181d94ed9463c716c0163edaa1a8289ba28`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
