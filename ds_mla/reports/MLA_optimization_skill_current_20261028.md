> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

---
name: heteromcp-mla-optimization-current
description: 从2026-10-28静态MMA片段索引方案续接MLA优化；保留真实HeteMCP/DSH链路及完整GPU验收。
updated: 2026-10-28
---

# MLA 优化续接 Skill（2026-10-28）

本文件承接同目录20261027历史报告及skill，并更新当前状态。每次继续前核对文件哈希，不把状态快照当作覆盖用户后续改动的授权。

## 当前候选与依据

- 本轮验收候选SHA-256：`5f6c03f884eb92dab1dad8eea47411facce93f62704bcbd52f4e99af35b11a70`。
- 生产目标：`D:/workre/tzb/MLA.py`；实际是否已发布，以目标哈希和 `.heteromcp/mla-architecture-20261028/publication.json` 为准。
- 发布前基线：`ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`。
- 计划独立备份：`D:/workre/tzb/MLA.score96p67-pre-unroll-20261028-ef30c4edde.py`。
- DSH task：`b39e35ad-54c7-41df-8d99-7edd9836f4a2`；attempt：`b175d533-66a4-4057-896f-d12d8ba73908`。
- 原始产物：`D:/workre/HeteMCP/.heteromcp/artifacts/b39e35ad-54c7-41df-8d99-7edd9836f4a2/b175d533-66a4-4057-896f-d12d8ba73908/workspace/MLA.txt`。
- 唯一生产改动：在flashattn的JIT pass_configs中保留FAST_MATH，增加 `"tl.UnrollLoop": {"auto_max_extent": 16, "explicit_unroll": True}`。所有算子函数主体、数值公式、split、merge及host包装AST保持原样。
- 这是执行结构的循环展开，使片段索引静态化的实验；没有修改编译器、驱动、资源上限，也没有近似数学或乘法替换加法。
- 核心六点双seed通过；7轮×100次、预热200，ctx8192/16384/32768/65536中位数加速2.394/2.625/3.143/3.401×，四点所有配对样本均>2×。
- 原始31-case、seed0/42全部通过；5轮×20次、预热20，几何平均3.13167×；第一点约0.963×，其余点中位数均>1。短点计时噪声大，不能声称所有shape都有稳定收益。
- 当前候选尚未提交OJ。最后真实OJ96.67对应发布前ef30版本，用户本轮提供六点20/34/40/65/116/215μs，总490μs。不要把SSH结果写成已超过97分。
- SSH服务器生产`/data/tilelang-metax/race_tests/mla/MLA.py`仍为4b32基线；本轮候选、ef30基线与生成代码位于独立stage，不假定服务器生产已同步。

## 分工与硬约束

1. Codex负责架构设计、任务拆解、HeteMCP调度、环境检查、静态审阅、验收、日志、独立备份和原样复制。DSH编写所有生产改动和独立profiling/validation代码；Codex不手改DSH算子产物。
2. 纯TileLang；生产禁止torch、numpy、triton及其他attention替代。保留flashattn、_get_kernel、run_kernel公开签名、输入布局、原地输出、kv_heads=1、FP32 accumulation/LSE与mask。
3. 模块级赋值必须ast.literal_eval兼容，OJ loader不能接受算术表达式全局常量。
4. 每次研究固定完整SHA基线、单一机制；失败候选不盲目叠加。超时/取消/静态拒绝/GPU失败全部留痕，重试要有明确调整。
5. 发布前静态审阅、GPU编译、双seed核心paired A/B、完整31-case、独立备份、目标并发哈希检查、DSH artifact原样复制、发布后哈希核对。
6. 不修改参考、容差、测试JSON，不缓存参考输出，不按case ID取巧，不覆盖原始日志和备份。

## HeteMCP 必须是真实调用

调用链：Codex MCP client → node dist/index.js → Orchestrator → TaskExecutor → AgentHarness → DSH。

已有调度器：`D:/workre/HeteMCP/.heteromcp/supervise-mla.mjs`，`submit SPEC KEY`及`read TASK_ID`。本轮全部通过它调用；不得用直接模型请求冒充MCP。

必须显式向stdio子进程传递：

```text
HETEMCP_DATABASE_URL=postgresql://heteromcp:<runtime secret>@127.0.0.1:55432/heteromcp
HETEMCP_DSH_DESKTOP=D:/dsh/dddd
HETEMCP_CREDENTIALS_PATH=<USER_HOME>/.dsh/.credentials.yaml
HETEMCP_ALLOW_UNMETERED=1
```

数据库容器`heteromcp-postgres`，PostgreSQL17，数据库/用户heteromcp，卷`heteromcp_postgres_data`。复用，禁止重建或删除；不要运行测试Compose替代它。凭据仅运行时传入，不写prompt/报告/源码。

TaskSpecV1使用workerProfile `deepseek-file-generation`、acceptance `["generated-file-scope"]`。生产输出单文件`MLA.txt`，预算输入40000–44000、输出16000–24000tokens，timeout600000ms。objective必须附完整哈希基线源码、单项改动、禁止项和外部验证边界。仅有文件范围验收成功不等于GPU/OJ成功。

任务快照`.heteromcp/runs/<id>.json`；artifact `.heteromcp/artifacts/<task>/<attempt>/`；`.heteromcp`被Git忽略，数据库也不保存全部大文件，要保留原目录。

## GPU 环境与验收

- MetaX C500、MACA3.5.3.20、TileLang0.1.9 MACA；50%sGPU分片，其他分片可能有负载，不干预其进程。
- `/data/tilelang-metax`；Python `/data/mla-validation-env-20261026-a57c1c819a/bin/python`。
- stage `/data/tilelang-metax/race_tests/mla/codex-20261026-a7b2de1e-a57c1c819a`。
- SSH runner `.heteromcp/remote-batch.py`固定host fingerprint；`.heteromcp/run-remote-secure.py`仅通过无回显getpass向runner stdin输入凭据，不落盘。
- 使用已有 `.heteromcp/mla-score96-20261026/fair-benchmark.py`，双方预分配output，NaN填充，检查原地有限输出；reference为未改的`test_tilelang_mla.before.py`（SHA `f4b651387eeef921abdda892e13857ba60d6cfda81a7011267153aabb72f75db`）。
- seeds0/42，rtol2e-4、atol1e-4、max_mismatched_ratio0.01；7轮交替A/B各100次、预热200，保存全部样本。
- 完整回归可同一fair脚本`--ids 1,...,31 --trials 5 --repeats 20 --warmup 20`。旧`MLA-validation.py`要求baseline.ref_program，纯算子不提供此接口；不要直接把纯baseline传给旧脚本。旧脚本此类FATAL是harness契约问题，不是算子数值失败。
- 不把旧历史comparator当作当前基线。GPU测试串行，生成代码检查器仅compile/export不计时。
- 生成代码文本及资源metadata不能证明硬件spill、occupancy或带宽；无有效counter，不虚构归因。
- 早期额外短/不规则边界case的问题未在本轮扩展验收，31/31只覆盖原始31case。

## 本轮否决/暂缓路线

- 四独立128维PV fragment：六点数值过，短点和最长点无稳定收益，不发布；有一条未消费no-split LSE计算需DSH清理才可重开。
- QK/PV分离：修复stage2 hid/did与stage3 Kernel binding的eager-builder scope冲突后核心通过，最长点约1.27×，远低于展开方案；不合入最终候选。
- QK/PV分离上的N64/K128 scoreproducer：已生成SHA518cba...，未GPU验收，停止扩展；不得当作通过版本。
- 原有Q cache、单tile shortcut、N16/64thread、无资源重构的N64、factor/reciprocal、盲增split均有既有负向证据，不重复。

## 下一步

优先将本轮已验收候选5f6c...提交真实OJ，记录提交源码哈希、逐点耗时/分数和总分；先检查OJ环境接受tl.UnrollLoop配置。未得到OJ反馈前不叠加分离架构或其他优化。

如果OJ仍无改善，先核对OJ是否实际使用新源码、配置是否生效、编译器版本与生成代码、计时方式，再决定下一实验。不能从SSH相对加速直接推断OJ分数。

详细实验、失败记录、任务及证据见`MLA_architecture_optimization_20261028.md`及`D:/workre/HeteMCP/docs/mla-architecture-20261028.md`；本轮证据根目录`.heteromcp/mla-architecture-20261028/`。
