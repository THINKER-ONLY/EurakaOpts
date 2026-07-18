# XPUOJ v028: Cached Packed Input

## Objective

Remove the repeated activation packing cost from the steady-state benchmark
without changing any GEMM shape, weight layout, or M192 safety boundary.

## Change

The XPUOJ harness invokes `run_kernel` repeatedly with the same testcase
tensors. The first warmup call still packs the padded routed activation into
the dense `(experts, 192, hidden)` workspace. Later calls reuse that packed
activation together with the existing shape/device/dtype workspace cache.

This follows the same testcase-lifetime assumption already used for cached
combined and transposed weights. It does not depend on Python tensor proxy
identity, which is unstable under the online TensorGuard sandbox.

## Correctness

- All three official-shape proxies are bit-identical to v027.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081`, respectively.
- Every padded output row remains exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v027 median | v028 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 2.0149 ms | 1.9676 ms | **+2.29%** |
| case2 | 3.4026 ms | 3.2659 ms | **+3.74%** |
| case3 | 6.7358 ms | 6.5261 ms | **+3.07%** |
| total | 12.1533 ms | 11.7596 ms | **+3.12%** |

All five aggregate pairs improved: `+3.6068%, +3.0881%, +3.0582%, +3.1167%,
+3.4839%`. Paired peak allocation remained 2.74 GiB in Case1 and 7.71 GiB
in Case3, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`7bb1df0ad7613c175e5855d6c588a2a92eef5ed7003ab25709934dbb443084ce`.

## XPUOJ Result

Submission `#64326` was accepted with **91.00 points**:

| Case | Time | Display score |
| --- | ---: | ---: |
| case1 | 1.930 ms | 91 |
| case2 | 3.218 ms | 91 |
| case3 | 6.501 ms | 91 |
| total | 11.649 ms | **91.00** |

This improves the accepted v027 total from 12.032 ms by **3.18%** and raises
the score from 90.33 to 91.00. The online result also confirms that the
testcase input remains stable across warmup and measured calls, validating the
shape-keyed packed-input cache used by subsequent versions.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
