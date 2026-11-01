# 优化过程与里程碑

## 任务与计算范围

任务是在 MetaX C500 / MACA 上优化纯 TileLang MLA decode。正式集合为31个 batch/context 组合，
heads16、kv_heads1、dim512、pe_dim64，核心六点为 batch1、context2048/4096/8192/16384/32768/65536。
接口保持 `flashattn`、`_get_kernel`、`run_kernel`，原地写输出；生产实现不引入 torch/numpy/triton。
参考与验证脚本单独使用 torch。原始文件及其 SHA 见 [来源清单](../SOURCE_MANIFEST.json)。

## 演进总表

| 时间 / 阶段 | 源码 SHA 前缀 | 主要变化 | 证据及决策 |
|---|---|---|---|
| 10-26 原始 | `8ddf480f34` | 初始单文件 | [原件](../code/MLA.original-20261026.py) |
| 10-26 理论候选 | `bac9755706` | balanced split、tail mask、FP32 LSE、减少冗余搬运 | 静态通过；不等于 GPU 通过 |
| 10-26 首次 GPU 修复 | `2e16438207` | 真实编译失败后，调整合法128线程、pipeline stages0 | 原31点通过；相对初始基线几何平均2.32345×，仅该轮对照有效 |
| 10-26 OJ loader 兼容 | `ff56144e53` | `8 * 1024 * 1024` 改为等值字面量 `8388608` | 用户反馈71.5分 |
| 10-26 shape split | `3ee71019b6` | shape 驱动 CTA/split 选择 | 用户反馈96.17分 |
| 10-27 hybrid merge | `db89ab8fe6` | split32—128并行 tiled merge，小split保留串行 | 用户反馈96.67分 |
| 10-27 merge64 | `4b32b1ccfb` | merge线程128→64 | 本地六点1.01946×；用户反馈仍96.67分 |
| 10-27 conditional rescale | `ef30c4edde` | scale==1时跳过512维乘法 | 本地六点1.04950×；后续用户确认96.67分 |
| 10-28 全局/局部显式展开 | `5f6c03f884` / `7f5f1fdfa0` | 在IR阶段复制循环体 | 本地显著快但OJ编译失败；不能作为可用提交 |
| 10-28 源码静态 PV | `def723b6f9` | 固定 local 分配、静态加载/MFMA，保留serial ki2 | 本地31点通过；用户反馈OJ96.67 / 483μs |
| 10-28 最佳参数固定 | `5e3efcc1da` | short CTA target128→64；用户给定完整源字节 | 用户反馈OJ96.83 / 474μs；最佳基线 |
| 10-28—30 QK128分块 | `9699e7d193` | 小块QK后重载完整KV供PV | 本地快，OJ96.17 / 712μs，否决 |
| 10-30 结构探索 | 多版本 | N64、warp-local、pipeline、QK static/lookahead | 编译资源或本地性能否决，详见实验表 |
| 10-30 独立候选 | `01a2c12bc5` / `3b20aa59e4` | PV流式B4 / 延迟softmax分母归约 | 双seed31点通过，小幅本地收益；OJ待测 |

这些相对加速来自不同基线、不同轮次；不能连乘，也不能用来推算 OJ 分数。
备份文件名中的后缀并不总是该文件实际 SHA，必须使用来源清单或重新计算哈希。

## 三次关键转折

1. 从理论布局到真实 MetaX 编译：256线程触发 MMA layout inference 问题，128线程配置才落地。Python AST正确并不保证 kernel 可编译。
2. 从编译器展开到源码静态化：全局与局部显式展开都在OJ编译阶段失败；源码显式PV通过了后续真实OJ。旧失败的具体终端断言未完整取得，相关编译器修复只属于强线索。
3. 从“SSH快”到双环境验收：QK128的共享内存与mtreg下降，SSH第四点1.174×；但OJ第四点64→96μs，合计474→712μs，最终否决。

## 历史文档与当前状态

初期完整叙事：[历史与MCP留痕](../reports/MLA_optimization_history_and_MCP_trace_20261027.md)。
编译兼容路线：[源码静态PV](../reports/MLA_source_explicit_pv_20261028.md)。
最佳基线与QK失败：[96.83后续探索](../reports/MLA_9683_optimization_20261028.md)。
最后候选：[PV stream](../reports/MLA_PVstream_20261030.md)、[Deferred sum](../reports/MLA_DeferredSum_20261030.md)。

保留旧报告是为了追溯假设与决策，不应直接照旧交接文件恢复服务器路径或替换最佳文件。
