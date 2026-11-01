# 源码快照索引

所有Python快照保留原字节。文件名后缀可能是任务或历史标签，实际SHA以此表和来源清单为准。
优先选用 `MLA.py` 最佳基线；失败版本只用于复盘。`experiments/` 保留各实验目录中的历史副本。

| 文件 | 实际SHA前缀 | 状态 | 可匹配的真实任务 |
|---|---|---|---|
| [MLA.baseline96p83-20261028.py](MLA.baseline96p83-20261028.py) | `5e3efcc1da` | 最佳基线；用户反馈OJ96.83 / 474μs | 未匹配终态artifact；以来源清单为准 |
| [MLA.before-oj-20261026-3012d1240c.py](MLA.before-oj-20261026-3012d1240c.py) | `2e16438207` | 历史阶段快照，见优化过程/原报告 | `d5d02bcc-b021-4fac-a427-ccb2a1ac1b1d` |
| [MLA.before-ssh-20261026-a57c1c819a.py](MLA.before-ssh-20261026-a57c1c819a.py) | `bac9755706` | 历史阶段快照，见优化过程/原报告 | `542e3769-c063-458b-a2b4-ee1e9c4e74ce` |
| [MLA.before-validated-20261026-67f9c320b3.py](MLA.before-validated-20261026-67f9c320b3.py) | `d7770ae66a` | 历史阶段快照，见优化过程/原报告 | `a7b2de1e-1e8e-4397-8c34-a8717a0eaaf9` |
| [MLA.candidate-deferredsum-20261030.py](MLA.candidate-deferredsum-20261030.py) | `3b20aa59e4` | 独立候选；原31点通过；OJ待测 | `793a4e11-a7da-49e1-9bfa-21fce6722476` |
| [MLA.candidate-pvstream-B4-20261030.py](MLA.candidate-pvstream-B4-20261030.py) | `01a2c12bc5` | 独立候选；原31点通过；OJ待测 | `774c955a-a973-49b0-abcf-c97313d315cf` |
| [MLA.candidate-qkchunk128-20261028.py](MLA.candidate-qkchunk128-20261028.py) | `9699e7d193` | OJ96.17 / 712μs；否决 | `4676c881-fb07-43b4-95be-d3e7557cd97c` |
| [MLA.diagnostic-backend-unroll-hint-20261028-8465065409.py](MLA.diagnostic-backend-unroll-hint-20261028-8465065409.py) | `8465065409` | 诊断版本；SSH无性能收益 | `fa977b4b-fe28-4210-8a81-2e26d3374996` |
| [MLA.OJ-failed-global-unroll-20261028-5f6c03f884.py](MLA.OJ-failed-global-unroll-20261028-5f6c03f884.py) | `5f6c03f884` | OJ显式展开编译失败 | `b39e35ad-54c7-41df-8d99-7edd9836f4a2` |
| [MLA.OJ-failed-local-unroll-20261028-7f5f1fdfa0.py](MLA.OJ-failed-local-unroll-20261028-7f5f1fdfa0.py) | `7f5f1fdfa0` | OJ局部展开编译失败 | `4560128a-dab1-412f-b6c2-1f7507fab857` |
| [MLA.optimized-source-pv-20261028-def723b6f9.py](MLA.optimized-source-pv-20261028-def723b6f9.py) | `def723b6f9` | 历史静态PV；OJ96.67 / 483μs | `cf1e3f80-dd19-41fc-be1f-03858661ca2f` |
| [MLA.original-20261026.py](MLA.original-20261026.py) | `8ddf480f34` | 最初版本 | 未匹配终态artifact；以来源清单为准 |
| [MLA.pre-9683-baseline-20261028-def723b6f9.py](MLA.pre-9683-baseline-20261028-def723b6f9.py) | `def723b6f9` | 历史静态PV；OJ96.67 / 483μs | `cf1e3f80-dd19-41fc-be1f-03858661ca2f` |
| [MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py](MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py) | `4b32b1ccfb` | 历史merge64；用户确认OJ96.67 | `fefaedc0-7175-49e0-8dd6-09e6152a7e62` |
| [MLA.pre-source-pv-20261028-ef30c4edde.py](MLA.pre-source-pv-20261028-ef30c4edde.py) | `ef30c4edde` | 历史conditional-rescale；用户确认OJ96.67 | `6c8c371c-5347-4873-b25d-21e8cccf6f6a` |
| [MLA.pre-split256-20261028-def723b6f9.py](MLA.pre-split256-20261028-def723b6f9.py) | `def723b6f9` | 历史静态PV；OJ96.67 / 483μs | `cf1e3f80-dd19-41fc-be1f-03858661ca2f` |
| [MLA.py](MLA.py) | `5e3efcc1da` | 最佳基线；用户反馈OJ96.83 / 474μs | 未匹配终态artifact；以来源清单为准 |
| [MLA.rejected-OJ503-20261028-ac9cdc1b08.py](MLA.rejected-OJ503-20261028-ac9cdc1b08.py) | `ac9cdc1b08` | OJ503μs；否决 | `d835fd3b-34cb-4f93-b29f-780577db412e` |
| [MLA.score71p5-20261026-a43bf064bd.py](MLA.score71p5-20261026-a43bf064bd.py) | `ff56144e53` | 历史字面量兼容；OJ71.5 | `a6355add-eabc-4767-9ffc-10b98cd9c8d4` |
| [MLA.score96p17-20261026-50119b5426.py](MLA.score96p17-20261026-50119b5426.py) | `3ee71019b6` | 历史shape split；OJ96.17 | `bc8f150f-fa66-4633-b75b-15533dd1f8ff` |
| [MLA.score96p67-20261027-189eac4be0.py](MLA.score96p67-20261027-189eac4be0.py) | `db89ab8fe6` | 历史hybrid merge；OJ96.67 | `a3bae057-e6e9-4663-b030-4825b6d015d2` |
| [MLA.score96p67-pre-unroll-20261028-ef30c4edde.py](MLA.score96p67-pre-unroll-20261028-ef30c4edde.py) | `ef30c4edde` | 历史conditional-rescale；用户确认OJ96.67 | `6c8c371c-5347-4873-b25d-21e8cccf6f6a` |

完整跨目录溯源见 [SOURCE_MANIFEST](../SOURCE_MANIFEST.json)，任务状态见 [hetemcp_tasks](../evidence/hetemcp_tasks.json)。
