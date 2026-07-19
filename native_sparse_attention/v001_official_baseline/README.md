# NSA v001: Official Baseline

## Objective

Archive the official XPUOJ TileLang template and establish a reproducible local
MetaX C500 baseline before changing the kernel.

## Correctness

The local gate runs the same compiled shape with two independently generated
sets of Q, K, and V tensors. Both calls match a direct FP32 sparse-attention
reference with `atol=1e-2, rtol=1e-2`:

| Input set | Maximum absolute error |
| --- | ---: |
| initial | 0.00159550 |
| changed, same shape | 0.00134659 |

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. Each case used five warmups, twenty measured calls
per sample, and five samples:

| Case | Median latency |
| --- | ---: |
| short S1: B1, L1024, D64, BS16 | 0.0161 ms |
| long S1: B1, L16384, D64, BS16 | 0.1722 ms |
| multi S4: B4, L1024, D64, BS16 | 0.1096 ms |
| wide: B2, L1024, D128, BS32 | 0.2343 ms |
| 64K stress: B1, L65536, D64, S16, BS64 | 80.7689 ms |
| proxy total | **81.3010 ms** |

The 64K stress case follows the competition's stated sequence length but is a
local diagnostic configuration, not a claim about the undisclosed XPUOJ case.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`ba61041cf0be0deed46018afd2a932635b8e07c04832745ff89e94b1c56cfc1b`.
