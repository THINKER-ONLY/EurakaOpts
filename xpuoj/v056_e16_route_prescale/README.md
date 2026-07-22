# XPUOJ v056: E16 Route Prescale

## Objective

Reduce the E16 routed-output epilogue without caching inputs or results and
without changing the E32/E64 paths inherited from v055.

## Change

For E16, multiply each valid SwiGLU row by its current FP16 route weight before
the down projection.  Linearity moves the existing route multiplication from
the final unpack to the already-running SwiGLU kernel.  The final E16 unpack is
therefore a pure FP16 copy.

Every invocation still packs the current activation, executes both FC1 GEMMs,
recomputes SwiGLU, executes the down GEMM, reads the current route weights and
metadata, and writes the current output.  No input, activation, down result, or
completed output is reused.  E32 and E64 are unchanged from v055.

## Correctness

All three official proxies are bit-identical to v055 on the constant benchmark
inputs, including padded rows, and both FP32 oracle shapes pass.

A full E16 random test used activation standard deviation 0.5, weight standard
deviation 0.02, and route weights sampled in `[0.05, 1.0]`.  Across 4,653,056
valid output elements, v056 differed from v055 by at most `0.00048828125`, with
zero `atol=1e-2, rtol=1e-2` violations.  Three experts were also checked
directly against an FP32 oracle; v056's maximum absolute error was below
`0.000618`, again with zero tolerance violations.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5.  To remove the
module-position bias, the control and prescale variants were loaded from the
same source file and selected by a compile-time module flag.  Each direction
used ten warmups, thirty calls per timing slot, and twelve samples:

| Load order | Control | Route prescale |
| --- | ---: | ---: |
| control first | 1.929954 ms | 1.913216 ms |
| prescale first | 1.922692 ms | 1.912550 ms |
| geometric correction | 1.926320 ms | 1.912883 ms |

The corrected E16 improvement is **+0.698%**.  E32 and E64 retain v055, so the
estimated three-case aggregate improvement is **+0.124%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`1a88c69a5d5a7dd4950d57f61288a9b3094d677f01963904afb1d7e287c21dd5`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
