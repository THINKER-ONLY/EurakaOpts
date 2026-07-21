# XPUOJ v047: E64 Stream Pack

## Objective

Remove the remaining global pack barrier from the E64 full stream pipeline and
retune the smaller per-stream copy without changing v046's computation.

## Change

E64 replaces its single 64-expert `BN128,T512` pack with two 32-expert
`BN256,T256` packs. Each pack runs at the head of the stream that consumes its
half of the experts:

```text
32-expert pack -> fused FC1 -> 2 x (16-expert SwiGLU -> down BMM)
```

This lets one half enter FC1 without waiting for the other half's input copy.
An isolated two-stream scan reduced concurrent pack time from 0.27381 ms for
`BN128,T512` to 0.22672 ms for `BN256,T256`. E16 and E32 retain v046's global
pack and otherwise follow byte-equivalent hot paths.

Every invocation still copies the current valid input rows and recomputes FC1,
SwiGLU, down projection, route multiplication, and output. No result is cached.

## Correctness

Random tests using all three disclosed expert distributions passed against the
FP32 oracle:

| Case | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| E16 | 0.001953125 | 0.0001468057 |
| E32 | 0.001953125 | 0.0001410794 |
| E64 | 0.002197265625 | 0.0001427663 |

After CUDA Graph capture, activation and route weights were modified in place
for E32 and E64. Graph replay and a fresh eager execution were bit-identical
for both distributions (`max_abs=0`). Constant-data outputs are bit-identical
to v046, including every padded row.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each independent process used ten warmups, thirty calls per
timing slot, and twelve samples. To avoid the known two-module graph allocation
bias, structurally identical controls compiled both pack variants and differed
only in whether E64 executed the global or per-stream pack.

Three fresh processes for each mode produced:

| Mode | Process medians | Median of processes |
| --- | --- | ---: |
| global pack | 6.33113, 6.31351, 6.33018 ms | 6.33018 ms |
| stream pack | 6.30216, 6.31445, 6.29484 ms | 6.30216 ms |

The E64 improvement is **+0.443%**. E16 and E32 retain the same operations;
weighting E64 by v046's three-case runtime gives an estimated aggregate
improvement of **+0.25%**. Standalone peak allocation was 9.67 GiB.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`1439daf3d6753cdc41db56bed1804b15136f933a1141c355a23eae1a0621ea14`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
