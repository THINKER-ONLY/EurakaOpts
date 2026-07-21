# XPUOJ v050: Route Precision Retune

## Objective

Retune route scaling and the now copy-only E64 unpack after v049 moved route
multiplication ahead of the down projection, while keeping complete
recomputation and graph input sensitivity.

## Change

This version combines four small, independently measured changes:

- E16 enables TileLang fast math for SwiGLU and explicitly casts each FP32
  route scalar to FP16 in the final unpack.
- E64 explicitly casts each FP32 route scalar to FP16 in its routed SwiGLU.
- E64's copy-only four-chunk unpack changes from `BN256,T256` to
  `BN256,T1024`.
- E32 retains v049's FP32 route expression and `BN256,T1024` unpack because
  an FP16 route cast was neutral to slightly slower in complete-graph timing.

The FP16 route casts introduce one deliberate rounding step before the FP16
output/activation. Every invocation still reads the current route weights,
packs the current input, and recomputes FC1, SwiGLU, down projection, and all
valid output rows. No result or input identity is cached.

Isolated medians changed as follows:

| Kernel | v049 | v050 | Improvement |
| --- | ---: | ---: | ---: |
| E16 SwiGLU | 0.10566 ms | 0.10484 ms | +0.78% |
| E16 unpack | 0.02026 ms | 0.01996 ms | +1.51% |
| E64 unpack | 0.18970 ms | 0.18442 ms | +2.78% |

## Correctness

Random tests using all three disclosed expert distributions passed against the
FP32 oracle:

| Case | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| E16 | 0.001953125 | 0.0001533302 |
| E32 | 0.001953125 | 0.0001413113 |
| E64 | 0.002197265625 | 0.0001495487 |

After graph capture, activation and route weights were modified in place for
E32 and E64. Graph replay and a fresh eager execution were bit-identical for
both distributions (`max_abs=0`). Padded-row checks also passed. E32's
oracle statistics are identical to v049.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each result below is a process median; each process loaded
only one version. E16 used 100 calls per timing slot and E64 used 50.

| Case | v049 process medians | v050 process medians | Median paired improvement |
| --- | --- | --- | ---: |
| E16 | 1.91118, 1.91310, 1.91069 ms | 1.90863, 1.90940, 1.90900 ms | **+0.133%** |
| E32 | 3.10862 ms | 3.10978 ms | -0.037% (neutral) |
| E64 | 6.27690, 6.28445, 6.29556 ms | 6.26744, 6.27499, 6.29435 ms | **+0.150%** |

Weighting the E16 and E64 median improvements and treating E32 as unchanged
gives an estimated three-case improvement of **approximately +0.10%** over
v049. This is a small optimization, but its component kernels and independent
process ratios agree on direction.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`561ffc9e7c4d55273b674da8ea01065a84e21253be073507d395a66645848421`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
