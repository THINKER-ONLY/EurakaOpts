# MLA Decode v001: Official Baseline

## Objective

Archive the official XPUOJ TileLang template and establish a reproducible local
MetaX C500 baseline before changing the kernel.

## Correctness

The local gate runs the same compiled shape with two independently generated
sets of Q, Q-PE, KV, and K-PE tensors. Both calls match an FP32 PyTorch
reference with `atol=1e-2, rtol=1e-2`:

| Input set | Maximum absolute error |
| --- | ---: |
| initial | 0.00013395 |
| changed, same shape | 0.00013334 |

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. Each case used five warmups, ten measured calls per
sample, and five samples:

| Case | Median latency |
| --- | ---: |
| batch 1, context 8192 | 1.4833 ms |
| batch 8, context 8192 | 1.8266 ms |
| batch 32, context 8192 | 1.8556 ms |
| batch 1, context 65536 | 11.8023 ms |
| batch 32, context 65536 | 14.7959 ms |
| proxy total | **31.7636 ms** |

The race branch's historical result for batch 1/context 8192 is 4.6272 ms.
The current full-device allocation is substantially faster, so only paired
measurements from this environment will be used for optimization decisions.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`59c8f757122da36fe101b495875d546241c710bbe4c6994e367a043c566b6951`.
