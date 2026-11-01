> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

---
name: heteromcp-mla-handoff
description: MLA优化跨窗口交接；Codex负责架构与验收，真实HeteMCP委派DSH实现。
updated: 2026-10-28
---

# 新窗口从这里继续

目标：修复优化版在OJ的编译错误，保留加速并突破97分。**回退正常版不算修复；SSH通过不等于OJ通过。** 主对话只保留决策和关键证据，细节写报告。

## 最少阅读

1. 本文件。
2. `D:/workre/tzb/log_report/MLA_compile_rootcause_investigation_20261028.md`：最新结论、失败实验、版本证据；优先级高于所有旧skill状态。
3. `D:/workre/tzb/MLA.py`：真实当前代码，先核对SHA。
4. `D:/workre/HeteMCP/.heteromcp/supervise-mla.mjs`：真实MCP调度器。

按需读取，不要开局灌入全部历史：

- 架构/性能历史：`D:/workre/tzb/log_report/MLA_architecture_optimization_20261028.md`。
- 初始至96.67的留痕：`D:/workre/tzb/log_report/MLA_optimization_history_and_MCP_trace_20261027.md`。
- 原始角色约束（原位置已归档）：`D:/workre/HeteMCP/.heteromcp/archive/2026-10-28/skill/MLA_skill.md`。
- OJ五版AST差分：`D:/workre/HeteMCP/.heteromcp/mla-unroll-differential-20261028/variant-comparison.json`。
- 原始证据根目录：`D:/workre/HeteMCP/.heteromcp/mla-compile-rootcause-20261028/`。

旧 `MLA_optimization_skill_current_20261028.md` / `MLA_skill_OJ_update_20261028.md` 有“尚未OJ提交”等过时状态，不能作为最新事实。

## 当前事实

- 当前生产 `D:/workre/tzb/MLA.py` SHA256：`ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`（conditional-rescale稳定版）。新窗口若不同，先检查用户修改，不自动覆盖。
- 正常备份：`D:/workre/tzb/MLA.score96p67-pre-unroll-20261028-ef30c4edde.py`。
- 真实OJ96.67；六点20/34/40/65/116/215μs，总490μs。97分尚未实现。
- OJ：**TileLang0.1.10，镜像描述marco torch2.8.0+metax**。不是已取得的完整镜像地址。
- SSH：TileLang0.1.9 MACA开发版，tir；OJ栈为tirx。不能当同环境复现。
- 上游v0.1.10 tag `69bc43e2cfbc38976b0c5e1c4b2299b66e8970ee` 无UnrolledBodyDefFreshener；FlattenBuffer保留DeclBuffer；SplitHostDevice279行，OJ280行。
- 上游PR2290/commit `ac2e1d7918ce25744959cecfed34e5ee24bd8b08` 明确修复：显式展开复用Buffer定义身份，导致VarUseDefAnalyzer在SplitHostDevice崩溃。与本例高度匹配；MetaX包是否同源码、最终断言仍未知，**不能写已完全证实或已修复**。
- 全部DSH/GPU任务已结束；本轮没有发布新候选。

| 候选SHA前缀 | 实质变化 | 已知结论 |
|---|---|---|
| 5f6c03f884 | 全局explicit unroll，extent16 | SSH大幅加速；OJ SplitHostDevice编译失败 |
| 7f5f1fdfa0 | PV局部maxstep256，仍explicit | SSH大幅加速；OJ同阶段失败 |
| 8465065409 | 改后端#pragma提示 | 六点正确，平均0.979×，否决 |
| 7d21c0093a | PV展开depth1保留ki | 31点双seed过，平均2.529×；内层加载仍有定义风险，未OJ验证 |
| d358c19a76 | PV加载移出attr，fragment/fragment GEMM | 六点正确，平均0.946×，否决 |
| bcede14049 | 16个PV32片段，无新增展开配置 | 六点正确，平均0.982×，否决；j消失但local_id4仍动态 |

候选完整SHA、task/attempt、路径见最新排查报告。别重新跑上述无变化实验。后续需要**PV加载索引静态化且Buffer定义唯一**，或取得实际MetaX0.1.10环境验证；只缩范围/限制深度不能宣称修复。

## HeteMCP→DSH：直接复用

工作目录 `D:/workre/HeteMCP`。真实链路：MCP SDK stdio client→`node dist/index.js`→Orchestrator→TaskExecutor→AgentHarness→DSH。不要直接调用模型冒充MCP。

```powershell
node .heteromcp\supervise-mla.mjs submit <TaskSpec.json路径> <本次唯一幂等键>
node .heteromcp\supervise-mla.mjs read <taskId>
```

`submit`自动等待终态并将快照保存 `.heteromcp/runs/<taskId>.json`。同一次提交重发沿用幂等键；有意新实验用新键。终态前不要发布仍在写入的产物。若因Docker EPERM失败，走实际工具提权重试，不绕开沙箱。

调度器内部复用 `heteromcp-postgres`，运行时读取数据库密码，不打印；向stdio子进程显式传递：

```text
HETEMCP_DATABASE_URL=postgresql://heteromcp:<runtime secret>@127.0.0.1:55432/heteromcp
HETEMCP_DSH_DESKTOP=D:/dsh/dddd
HETEMCP_CREDENTIALS_PATH=<USER_HOME>/.dsh/.credentials.yaml
HETEMCP_ALLOW_UNMETERED=1
```

禁止重建/删除数据库和卷。不要用测试compose替代。缺数据库配置可能退回内存，缺DSH配置可能选择fake provider。复用上述调度器，不用尚未核对配置转发的其他入口。

TaskSpec模板（填入固定基线完整源码及SHA、单一机制、允许差分、禁止项；不含凭据）：

```json
{
  "version": 1,
  "objective": "DSH只写完整MLA.txt；附本轮设计、约束及完整源码。禁止shell/network/sidecars。写完直接结束。",
  "allowedPaths": ["MLA.txt"],
  "excludedPaths": [],
  "acceptance": ["generated-file-scope"],
  "workerProfile": "deepseek-file-generation",
  "budget": {"maxInputTokens": 44000, "maxOutputTokens": 24000, "maxCostMicros": 3000000},
  "timeoutMs": 600000,
  "context": {"summary": "固定基线与单一实验", "decisions": [], "evidence": [], "openQuestions": []}
}
```

允许单文件扩展名限`.cpp/.txt/.md`，上限128KiB；生产Python写`MLA.txt`，审阅后原字节复制为`.py`。诊断也由DSH写`inspect.txt`等。生成成功仅证明文件范围验收，不证明编译/数值/性能。

产物：`.heteromcp/artifacts/<taskId>/<attemptId>/workspace/MLA.txt`；外层`report.json`是范围验收。快照里的artifact SHA必须核对。`.heteromcp`被Git忽略，数据库不保存完整大文件，保留所有日志和失败产物。

## 分工与验收

- Codex：架构、调度、证据审阅、编排脚本、日志和原样发布。**DSH写所有生产实现/修复及profiling/validation代码，Codex不手改产物。**
- 纯TileLang生产代码；禁止torch/numpy/triton。保留`flashattn/_get_kernel/run_kernel`签名、布局、原地输出、kv_heads=1、mask、FP32 accumulation/LSE。模块级赋值须`ast.literal_eval`兼容。
- 单轮固定SHA和单一机制。先静态差分，再真实编译、核心六点双seed paired A/B；有收益才完整31点。失败/超时留痕，有具体调整才重试。
- 使用原参考/原容差，不改测试取巧，不缓存答案，不用环境检测/异常fallback掩盖编译失败，不改编译器或驱动冒充源码修复；不擅装大型编译环境。
- 发布前备份、核对目标未被并发修改、原样复制、发布后比SHA。目标在工作区外时按实际工具权限写入；用户已授权的工作无需重复问意向。

## SSH/GPU复用

连接：`ssh <SSH_USER>@<SSH_HOST> -p 32222`。用户已授权SSH/GPU工作；**本文件不保存密码**。新窗口拿不到原对话凭据时由用户重新提供，别读敏感凭据文件猜测。

```powershell
python .heteromcp\run-remote-secure.py <job.json>
```

runner通过无回显getpass输入密码，固定服务器指纹；作业执行/上传/下载均留痕 `.heteromcp/mla-ssh-20261026/`。可参考`pv32-core-ir-job.json`结构，别直接重跑已有独占输出作业。逐step检查exit_code；外层status成功可能包含allow_failure步骤失败。

- GPU：MetaX C500，MACA3.5.3.20，50%sGPU。服务器当前没有docker命令；不干预其他分片进程。
- stage：`/data/tilelang-metax/race_tests/mla/codex-20261026-a7b2de1e-a57c1c819a`。
- Python：`/data/mla-validation-env-20261026-a57c1c819a/bin/python`。
- 环境：`MACA_PATH=/opt/maca USE_MACA=ON PYTHONPATH=/data/tilelang-metax`；PATH/LD_LIBRARY_PATH完整设置直接沿用已有job。
- 服务器生产MLA.py仍4b32历史基线，比较必须显式传stage里的`MLA.conditional-rescale-ef30c4edde.py`，不能误用服务器生产文件。
- harness本地：`D:/workre/HeteMCP/.heteromcp/mla-score96-20261026/fair-benchmark.py`；stage同名。原参考`test_tilelang_mla.before.py`；cases `test_cases_mla_batch_ctx.json`。
- 核心`--ids 1,2,3,4,5,6`，7轮×100次、预热200，seed0/42；rtol2e-4、atol1e-4、mismatch_ratio0.01，双方预分配输出。完整31点`--ids 1,...,31 --trials 5 --repeats 20 --warmup 20`。GPU作业串行。
- 旧`MLA-validation.py`要求baseline.ref_program，不能直接配纯算子baseline。生成代码/IR文本不等于硬件spill/吞吐证据。
- IR观测用DSH `inspect-ir-builtin.dsh.py`：开启JIT内置`tl.enable_dump_ir/tl.dump_ir_path`；**外层PassInstrument会被JIT自己的PassContext屏蔽，不再采用**。编译/导出不执行kernel，但仍不同于真实OJ编译。

开局操作：核对生产SHA→读最新报告→选一个有证据支持的修复设计→真实MCP委派DSH→审阅验证；不要恢复历史失败版本作为“修复”。
