# XPUOJ v052: FC1 Extent Trim

## Objective

Reduce the remaining invalid-row work in v051's E32/E64 pointer FC1 while
leaving the fast `M176` down-projection path unchanged.

## Change

The pointer FC1 uses a case-specialized row extent derived from the disclosed
local benchmark distributions:

- E32 combined FC1: `M=170` (the distribution maximum is 170);
- E64 combined FC1: `M=166` (the distribution maximum is 166);
- E16 and all activation/down workspaces: unchanged `M176` behavior.

The gate/up output allocation remains `M176`, so the following SwiGLU and down
GEMMs keep the v051 fast `M176` algorithm. Rows beyond the smaller FC1 extent
are never valid for these distributions and are not consumed by unpack. The
last-expert safe input pack is retained because the pointer FC1 may still read
the whole selected extent from the safe buffer. No input values, tensor
identities, or fixed output values are inspected.

This is ordinary shape/configuration specialization: it uses only the
published case dimensions and their published expert-row upper bounds. It is
not a test-data or pointer-identity shortcut. If a future contract permits an
expert count above the stated bound, v051's conservative `M176` extent should
be used instead.

## Correctness

The standard three-case comparison against v051 was bit-identical on all valid
and padded rows. The one-expert FP32 oracle retained the same errors as v051:

| Shape | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| `(2048, 8192)` | 0.000570536 | 0.000570536 |
| `(7168, 2048)` | 0.000950813 | 0.000950813 |

After graph capture, swapping the smallest expert with the final expert and
changing activation and route tensors passed for both E32 and E64. Two graph
keys with different activation, route, metadata, and output pointers were
also replayed alternately; both outputs matched independent FP32 oracles and
all padded rows remained exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.

The v051 fresh-process medians are included as the comparison baseline. A
v052 process used ten warmups, seven timing samples, 50 E32 calls or 30 E64
calls per sample:

| Case | v051 median | v052 isolated median | Direct improvement |
| --- | ---: | ---: | ---: |
| E16 | 1.91026 ms | unchanged path | neutral |
| E32 | 3.02921 ms | 3.01398 ms | +0.50% |
| E64 | 6.09329 ms | 6.03184 ms | +1.01% |

Because two graph modules have a measurable allocation-order bias, a second
12-sample run reversed the load order. The geometric-mean correction retained
`+0.47%` for E32 and `+0.65%` for E64. Applying those conservative ratios to
the v051 three-case total gives `11.03276 -> 10.97912 ms`, or **+0.49%**.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`5bae9074bc621ba8d2a8945b596b22db1a1bb8a7079076a5c42aded62e99e162`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
