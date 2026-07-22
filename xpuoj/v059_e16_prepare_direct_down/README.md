# XPUOJ v059: E16 Prepare Fusion and Direct Down Output

## Objective

Remove E16 preparation and output-copy overhead while preserving v058's
full-compute behavior and its E32/E64 paths.

## Change

E16 now fuses three per-invocation preparation operations into one TileLang
kernel:

- conditionally safe-pack a final expert with fewer than 128 rows;
- build the five-row FC1 device pointer table;
- build a three-row down-projection device pointer table.

The down projection uses pointer-batched mcBLAS.  Experts with at least 128
rows write directly into their current padded segment of `out`; smaller
experts write to the private workspace and retain the guarded unpack fallback.
SwiGLU clears invalid activation rows for direct-output experts.  E32/E64 are
unchanged from v058.

Every call rebuilds both pointer tables, recomputes both FC1 GEMMs, SwiGLU, and
the down GEMM, and reads the current activation, route metadata, route weights,
and output pointer.  No computed result or completed output is reused.

## Correctness

- The E16 official proxy matched v058 bit-for-bit on valid elements and kept
  every padded element at zero.
- FP32 oracle errors remained `0.000570536` for `(2048, 8192)` and
  `0.000950813` for `(7168, 2048)`.
- Metadata tests covered both the standard 158-row final expert and a mutated
  117-row final expert.  Replacing activation and route weights twice while
  keeping the small-final metadata also remained bit-identical to v058, with
  zero nonzero padding elements.
- The static submission check returned `status: pass`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.  Each direction used ten warmups, thirty calls per timing
slot, seven samples, and the same output address for both modules.

| Load order | v058 | v059 |
| --- | ---: | ---: |
| v058 first | 1.908762 ms | 1.896418 ms |
| v059 first | 1.901137 ms | 1.894741 ms |
| geometric correction | 1.904946 ms | 1.895580 ms |

The corrected E16 improvement is **+0.492%**.  Applying the unchanged E32 and
E64 measurements from v058 gives an estimated three-case total of
`10.839880 -> 10.830515 ms`, or **+0.086%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`1023b8ae621d83a854ddd4a2c5059e76d2513bf7d7d95339d3bc4206e7e6a4b1`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
