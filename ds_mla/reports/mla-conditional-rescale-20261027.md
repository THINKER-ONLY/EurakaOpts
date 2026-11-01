> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA conditional-rescale optimization (2026-10-27)

## Decision

Published the DSH-authored conditional-rescale candidate after static review,
paired GPU A/B, and the full 31-case regression. The candidate skips the
512-element `acc_o` rescale only when the already-computed `scores_scale` is
exactly `1.0`; one-tile paths retain the baseline loop.

The earlier algebra candidate (factored scale expressions plus explicit
reciprocal normalization) remains rejected because its six-case speedups were
`0.9890x, 0.9906x, 0.9973x, 0.9985x, 0.9982x, 0.9998x`.

## Artifacts

- Previous production hash: `4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba`
- Published production hash: `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`
- Backup: `D:/workre/tzb/MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py`
- DSH task: `6c8c371c-5347-4873-b25d-21e8cccf6f6a`
- DSH attempt: `23f60442-c6c4-45c7-9714-2c80d21ef2fc`
- DSH source artifact: `.heteromcp/artifacts/6c8c371c-5347-4873-b25d-21e8cccf6f6a/23f60442-c6c4-45c7-9714-2c80d21ef2fc/workspace/MLA.txt`
- Profile result: `.heteromcp/mla-deep-20261027/softmax-state-profile.json`
- Paired A/B result: `.heteromcp/mla-deep-20261027/conditional-rescale-fair.json`
- Full regression: `.heteromcp/mla-deep-20261027/conditional-rescale-full.json`

## Profile evidence

The DSH-authored value-frequency profiler used seeds 0 and 42. It found
`97,869 / 114,688` non-first head/tile states (85.335%) with exact identity
rescaling and zero nonfinite states. Identity frequency for contexts 8192,
16384, 32768, and 65536 was approximately 64%, 75%, 84%, and 90%.

This profile uses torch matmul, so its accumulation order may differ from the
production MMA. It establishes opportunity, not speed or correctness.

## GPU validation

The paired benchmark used both correctness seeds, seven alternating A/B
trials, 200 warmups per implementation, 100 repeats per trial, preallocated
outputs, and synchronized CUDA events. All six core cases passed.

| Case | Context | Speedup |
|---:|---:|---:|
| 1 | 2048 | 0.9986x |
| 2 | 4096 | 1.0008x |
| 3 | 8192 | 1.0656x |
| 4 | 16384 | 1.0747x |
| 5 | 32768 | 1.0790x |
| 6 | 65536 | 1.0821x |

Every paired trial for cases 3-6 was faster than the current `4b32...`
baseline. The full validation executed and passed all 31 JSON cases with both
correctness seeds, zero validation failures, and candidate hash `ef30...45a6`.

## Remaining uncertainty

No OJ submission was performed in this work. The previous score was 96.67;
the new OJ score must be measured separately and must not be inferred directly
from the SSH timing environment.
