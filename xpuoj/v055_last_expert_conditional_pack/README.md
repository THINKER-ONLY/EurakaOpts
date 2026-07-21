# XPUOJ v055: Conditional Last-Expert Safe Pack

## Objective

Avoid copying the final expert into the safe input workspace when its padded
segment is already large enough for the specialized FC1 extent.

## Change

The FC1 pointer-table kernel now selects the direct input segment for every
expert whose padded extent is safe, including the final expert when its group
has at least 128 rows.  Only a final expert below 128 rows uses the packed
fallback.  The corresponding last-expert pack kernel has a device-side
`group_size < 128` guard and exits without copying for the normal E32/E64
distributions.  Workspace strides remain 176 rows, and all FC1/down pointer
and direct-output behavior from v054 is unchanged.

This is metadata-dependent bounds handling, not test-result reuse.  Each call
still reads current activation, weights, routing metadata, and output pointers;
the graph replay recomputes the pointer table and all GEMMs.

## Correctness

- Standard v054 comparison passed for all three proxy cases with valid and
  padded rows bit-identical (`max_abs=0`).
- FP32 oracle errors remained `0.000570536` for `(2048, 8192)` and
  `0.000950813` for `(7168, 2048)`.
- After graph capture, changing activation/route and swapping the smallest
  expert with the largest passed for E32 and E64.
- After graph capture, moving the smallest expert specifically to the final
  position exercised the new safe fallback; E32 and E64 again matched v054
  bit-for-bit and padded rows stayed zero.
- Static submission check returned `status: pass`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Both directions used ten warmups, thirty calls per sample,
and twelve symmetric paired samples. Geometric means correct the two-module
allocation-slot bias.

| Case | v054 corrected | v055 corrected | Improvement |
| --- | ---: | ---: | ---: |
| E16 | 1.92252 ms | 1.92296 ms | -0.02% |
| E32 | 3.04700 ms | 3.02181 ms | **+0.83%** |
| E64 | 5.93957 ms | 5.92729 ms | **+0.21%** |
| total | 10.90909 ms | 10.87206 ms | **+0.34%** |

Independent single-module repetitions showed the same direction, with about
0.4% on E32 and 0.5% on E64.  The small E16 difference is measurement noise.

Decision: **accepted as the local full-compute baseline; not submitted to
XPUOJ**.

Submission SHA-256:
`3b47dd212a66ef32bfd5e81700e70a23a39699fad06ab7c1e38d6b554c0ae36f`.
