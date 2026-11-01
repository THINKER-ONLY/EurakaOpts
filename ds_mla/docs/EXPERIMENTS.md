# 实验结论表

状态优先级：本表覆盖旧报告的时点描述。OJ反馈均为历史用户报告，未独立取得提交后台记录。
“本地通过”仅覆盖指定shape、seed和容差，不表示所有API形状正确。

| 机制 | 实现/证据 SHA 前缀或报告 | 本地 / 编译观测 | 最终决策 |
|---|---|---|---|
| shape split | `3ee71019b6` | 原31点通过 | OJ96.17，历史接受 |
| hybrid merge | `db89ab8fe6` | 原31点通过 | OJ96.67，历史接受 |
| merge64 | `4b32b1ccfb` | 六点1.01946×，31点通过 | OJ仍96.67 |
| conditional rescale | `ef30c4edde` | 六点1.04950×，31点通过 | 后续用户确认96.67，已整合 |
| Q cache / single tile / N16 / algebra | [历史](../reports/MLA_optimization_history_and_MCP_trace_20261027.md) | 编译或paired A/B无有效收益 | 否决 |
| 全局显式展开 | `5f6c03f884` | SSH快、31点通过 | OJ编译失败，否决 |
| PV局部显式展开 | `7f5f1fdfa0` | SSH快、31点通过 | OJ同阶段失败，否决 |
| backend unroll hint | `8465065409` | SSH六点几何平均0.97917× | 仅诊断，不发布性能候选 |
| PV窄fragment、QK/PV分离等 | [架构历史](../reports/MLA_architecture_optimization_20261028.md) | 无有效收益 | 否决 |
| 源码静态PV | `def723b6f9` | 31点1.504439×；private252→0 | OJ通过96.67 / 483μs，历史接受 |
| short target64 | `5e3efcc1da` | 用户给定完整最佳源码 | OJ96.83 / 474μs，当前最佳 |
| split256 | `ac9cdc1b08` | SSH约11%快 | OJ503μs，否决 |
| QK/PE576融合、merge32 | [483后续](../reports/MLA_OJ483_investigation_20261028.md) | 无有效收益 | 否决 |
| QK128分块 | `9699e7d193` | 第4点1.173853×；31点通过；shared57344→53248 | OJ96.17 / 712μs，否决 |
| N64/D256、两CTA输出 | `32d2924a6a` | 六点通过，第4点约124→212μs | SSH性能否决；未跑31点/OJ |
| PV stream B4 | `01a2c12bc5` | 核心及31点双seed通过；mtreg244→218；六点1.014237× | **OJ待测**，独立候选 |
| shared bridge warp softmax | `b5ea3c3040` | AllReduce128→64，但barrier6→9，shared57344→59456；中长约慢2% | 否决；初版ea90cf布局失败也保留 |
| 单warp完整512输出 | `3a835d00d7` | private496，第4点123→319μs，最长435→1175μs | 否决 |
| 两阶段完整KV双缓冲 | `15af3e0786` | 6次compile/export成功；dynamic shared94208超过65536预算 | 资源否决，未执行计时 |
| QK32 MFMA源码静态展开 | `4f046b2d70` | mtreg256/private104，最长435→635μs | 否决 |
| QK static + B4修正 | `d8568d9522` | private104→20，但最长435→593μs | 否决 |
| QK lookahead | `422e8313cb` | 资源与基线相同，最长436→515μs | 否决 |
| deferred denominator sum | `3b20aa59e4` | 核心及31点双seed原容差通过；31点1.014005× | **OJ待测**，独立候选 |

## 可复核的数据

- [核心60行CSV](../data/ssh_core.csv)：10个实验×6点，含每行baseline/candidate SHA；不同实验的baseline并不相同。
- [OJ观察值](../data/oj_observations.json)：缺失耗时填null，没有用本地数据补写OJ。
- [任务摘要](../evidence/hetemcp_tasks.json)：保留失败/超时终态；未伪装成成功任务。
- [来源清单](../SOURCE_MANIFEST.json)：完整源路径、原SHA、归档SHA和是否脱敏。

## 明确的限制

原31点之外，早期补充shape存在旧基线即失败的记录，包括短context、不规则heads和dim256。
有的补充作业外层状态成功但内部exec退出1，不能声称补充集合全部通过。
完整历史见 [源码PV报告的额外覆盖](../reports/MLA_source_explicit_pv_20261028.md)。
SSH为TileLang0.1.9tir、MACA3.5.3.20、50%sGPU C500；OJ日志为TileLang0.1.10，其他配置未全部独立确认。
没有有效硬件带宽、occupancy或spill counter，资源元数据仅为编译观察。
不重复已否决的常量扫描或结构，除非新profile/编译器/硬件证据改变了前提。
