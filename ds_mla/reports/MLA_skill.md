> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

---
name: heteromcp-mla-handoff
description: MLA优化跨窗口交接；Codex负责架构与验收，真实HeteMCP委派DSH实现。
updated: 2026-10-30
---

# 新窗口从这里继续

目标：修复优化版在OJ的编译错误，保留加速并突破97分。**回退正常版不算修复；SSH通过不等于OJ通过。** 主对话只保留决策和关键证据，细节写报告。

## 最少阅读

1. 本文件。
2. `D:/workre/tzb/log_report/MLA_9683_optimization_20261028.md`：最新基线与实验。历史 `MLA_OJ483_investigation_20261028.md` 与 `MLA_source_explicit_pv_20261028.md` 为旧轮证据。旧 `MLA_compile_rootcause_investigation_20261028.md` 是编译排查历史，其中“仍ef30/未通过OJ”已过时。
3. `D:/workre/tzb/MLA.py`：真实当前代码，先核对SHA。
4. `D:/workre/HeteMCP/.heteromcp/supervise-mla.mjs`：真实MCP调度器。

按需读取，不要开局灌入全部历史：

- 架构/性能历史：`D:/workre/tzb/log_report/MLA_architecture_optimization_20261028.md`。
- 初始至96.67的留痕：`D:/workre/tzb/log_report/MLA_optimization_history_and_MCP_trace_20261027.md`。
- 原始角色约束（原位置已归档）：`D:/workre/HeteMCP/.heteromcp/archive/2026-10-28/skill/MLA_skill.md`。
- OJ五版AST差分：`D:/workre/HeteMCP/.heteromcp/mla-unroll-differential-20261028/variant-comparison.json`。
- 原始证据根目录：`D:/workre/HeteMCP/.heteromcp/mla-compile-rootcause-20261028/`。

旧 `MLA_optimization_skill_current_20261028.md` / `MLA_skill_OJ_update_20261028.md` 有“尚未OJ提交”等过时状态，不能作为最新事实。

## 当前事实（96.83分新基线）

- 当前生产/最佳基线 `D:/workre/tzb/MLA.py` SHA `5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c`，用户粘贴原始源码，OJ20/26/39/64/114/211μs，总474μs，96.83分。与def723只差 `_SPLIT_TARGET_SHORT=64`（旧128）及末尾换行。完整归档 `D:/workre/HeteMCP/.heteromcp/mla-9683-20261028/MLA.baseline-9683-5e3efcc1da.py`。
- 用户参数测试：shorttarget64最佳；后续split32/48/80/96/128/256、merge等推荐参数均无更好结果，不再重复常量扫描。当前shorttarget64、longtarget64、merge64，block_n32、headtile16。
- 历史split256(ac9cdc)SSH约11%快，OJ总503μs、最长230μs，已否决。def723历史OJ483μs/96.67；新基线不是这个旧版本，比较不得误传旧SHA。
- 历史独立机制QK/PE576融合、merge32、PV窄fragment、QK/PV分离（含N64K128）均无有效收益。旧显式展开有OJ编译错误，当前源码显式静态PV已OJ通过，不能恢复故障版本。
- OJ TileLang0.1.10、C500（全卡仅用户猜测）；SSH0.1.9tir、50%C500。SSH收益不代表OJ收益。未经实际OJ验证的候选不得替换新最佳基线。
- 融合kernel内QK128维分块实验已结束且OJ性能否决，最新报告 `D:/workre/tzb/log_report/MLA_9683_optimization_20261028.md`（工作区同名小写docs稿）。DSH task4676c881-fb07-43b4-95be-d3e7557cd97c，attempt1a9c4d74-20d0-4d6b-9a58-acd2e230e19b。候选SHA9699e7d1938a973bf8b52de3ccdc75b4384d2a91bfd20f179a70f8f5fb844136，已静态审阅、六点及31点双seed原容差全部过；SSH第4点124.262→105.859μs，1.173853×。中长抽样动态共享57344→53248、mtreg244→212、streg28→52；未采硬件计数。独立交付 `D:/workre/tzb/MLA.candidate-qkchunk128-20261028.py`，2026-10-30用户报告真实OJ96.17分、20/26/55/96/178/337μs共712μs，正式否决；SSH约17%收益没有迁移。生产仍96.83基线，不重跑同机制或仅扫chunk宽度。所有DSH/GPU任务终态。
- 新实验N64/D256融合、两CTA输出分块、静态PV8MFMA，DSH task8c53e0b5-e000-4a32-8cc6-066843cd2bd7 / attempt8fa3bd47-4929-4dff-87e5-2f26c00ff9d9，SHA32d2924a6aea97b551c32aff958906f2d8f7c119a3ef7ce445cda10b81049eb4：六点双seed正确，SSH第4点124→212μs，第3–6点均慢，性能否决，不跑31点、不交付OJ。勿重复相同重复QK/重载结构。最新报告 `D:/workre/tzb/log_report/MLA_N64_optimization_20261030.md`（工作区docs/mla-n64-optimization-20261030.md）。当前所有任务终态，生产仍96.83基线。本轮恢复Docker和原heteromcp-postgres（只启动未重建）。
- 本轮B4流式PV（不增加KV读取/不重算QK）候选SHA01a2c12bc515ac49dddffd2569b86d6a51b29c9331ac8a9877ee9460505558a9；真实DSH task774c955a-a973-49b0-abcf-c97313d315cf，attempt2cf52e13-8834-42eb-a04d-7c11a077e981。规范化approvedPV索引/分组及两处B64→4后整文件AST完全相同。核心6/6及31/31双seed原容差过；SSH整体几何平均1.014237×，第4点核心1.012835×、full31仅1.006185×，收益小且核心部分轮次未加速。旧编译器中长mtreg244→218、streg28/private0/shared57344不变。独立试验文件 `D:/workre/tzb/MLA.candidate-pvstream-B4-20261030.py`，OJ待测，MLA.py保持96.83基线。最新报告 `D:/workre/tzb/log_report/MLA_PVstream_20261030.md`。所有任务终态；若OJ无收益勿仅扫B8/B16重复。
- 2026-10-30新增两种warp-local softmax均否决：共享score重排桥接，原task3368775d-92de-4c4d-88c4-eb5752b7636e产物ea90cf有未使用旧rowfill导致layout推断失败，定向DSH修复task2c7860ce-ba0a-4b45-b422-c18e4196e154产物b5ea3c3040522db6ce8c28ac7f63354581334922a85828c789f8ed9c721aae24；实际AllReduce128→64，静态barrier6→9/shared57344→59456/mtreg244→248，六点双seed过但第3–6点约慢2%。后续原生64threads整512输出单warptaska533e2a8-5f2e-45d0-aa3b-77970a4dc7fd，attemptec952f05-9cde-4b89-8981-2fd7f319121e，SHA3a835d00d751b59a8b04ddcb1d3bb70557402126b6e12cbf5880a4e67d200ee0，实际AllReduce64、mtreg256/private496；六点双seed过，第4点123→319μs、最长435→1175μs，慢2.4–2.7倍，否决。不跑31点、不交付OJ、不覆盖基线。最新报告 `MLA_WarpSoftmax_20261030.md` / `MLA_OneWarp_20261030.md`（工作区docs同名小写）。所有任务终态。下轮不能重复sharedbridge或全512输出单warp，须保留128threads输出分摊/KV复用，再找QK执行或加载调度方向。
- 2026-10-30继续架构实验均否决：两阶段split流水真实DSH taskd14e15b6-0096-4f1b-b117-7175e720dda5 / attempta862c41f-c29f-4acf-b394-6e3309ff9123，SHA15af3e0786fa4fad784afe5923ae8bcc6d35c7715fcdf85625968949fe7a00da，6次compile/export成功但动态shared57344→94208，超设备/target65536上限，未执行GPU计时。勿再直接用完整KV双缓冲。报告MLA_Pipeline_20261030.md（工作区docs/mla-pipeline-20261030.md）。
- 同轮QK512源码静态32MFMA、保留原sharedswizzle/128threads/KV一次复用，DSH taskf49049e2-f478-488e-aa7a-e53b02fc0146 / attemptf072a4ed-8186-4154-9d8e-83cbe02691df，SHA4f046b2d70dda437c8e41a7e960b5f75a3be2f4f648d7384de097b88de76b887；6compile过，shared57344不变、mtreg244→256/private0→104，核心6双seed原容差过但第4点124.413→175.406μs、最长435.338→635.026μs，否决。定向B4PV资源修正task24f94f91-4e3b-4616-be3c-5603d48f877c / attempt810604cf-b1b6-403a-98f1-17ebadee3bdf，SHAd8568d95223924dea1b688ff0f786911c24cbbc3353b987c8cc3c760340fe2f2；新B4宏全AST等于此前01a2参考，仅qk_static路径用，不改短ctx/no_split/QK本身。6compile与核心6双seed全过，mtreg仍256、private104→20，最长435.136→593.206μs，仍慢、否决。核心JSON42cbfb07ec1763340eb6b2e4742cff90b3a791f5a8e49433d036c9143cf7322d。两版均不跑31、不交付OJ、不覆盖生产。报告MLA_QKStatic_20261030.md（工作区docs/mla-qkstatic-20261030.md）。勿重复QK32全部literal展开或仅堆叠B4；减少循环不等于降成本。所有真实DSH/GPU任务终态，生产仍5e3/96.83/474μs。下一架构尚未选定，必须兼顾shared65536预算/寄存器生存期/128threads分摊/KV复用，不凭旧编译器资源推断OJ。
- 后续QK两组寄存器lookahead（不是全展开）task59c7c795-416f-465e-87e6-6df846db1f3b / attempt694de82e-9a15-4fc0-b837-3205305f6c48，SHA422e8313cb4753d94564299768177bf9b9c2f04734f25494d1e31b4d59e976f8，16serialpairs原序32MFMA、A8/B8，原sharedswizzle/PV64。6compile资源与基线mtreg244/private0/shared57344相同，核心6双seed数值过，但第4点124.388→145.044μs、最长436.314→514.967μs，否决，不跑31、不交OJ。报告MLA_QKLookahead_20261030.md。不要重复同样QK预取。
- 新独立候选延迟softmax分母sum归约：真实DSH task793a4e11-a7da-49e1-9bfa-21fce6722476 / attemptd9fdffb0-179e-438b-b50c-671a35bc5c1f，SHA3b20aa59e4d7afc95e45bad700ddccf46381a9195f6582b98bd0187bfaf0cd34。基于生产5e3，ctx>=8192/head16对齐/pv_specialize/split>1；每scorelane FP32 D=alpha*D+P，split末尾才归约L=sum D，FP32加法顺序变。max/QKPE/PV64/KVonce/其他paths/merge不变。实际生成sumAllReduce128移出keyloop；长ctx shared57344→56832、mtreg244→246/streg28/private0。6compile、核心6和完整31双seed原容差全部过；核心第4点124.137→122.732μs（1.01145×）、最长435.924→427.540μs（1.01961×），收益小，不能称突破。独立交付MLA.candidate-deferredsum-20261030.py，OJ编译/性能/分数待测，生产仍5e3/96.83/474μs。报告MLA_DeferredSum_20261030.md。coreJSON96aa09582c49d5f232b6ab2226ce53d7af1c09432efbff85777c1eb1cb034f393，full31JSON00cff47cf1ba32dfb8bb1db306df7b931aa0a8e53c50da5c549c6498d457d4b32；所有DSH/GPU任务终态。不把旧compiler SSH收益预测为OJ收益。
- 97分尚未实现。原31点之外部分泛化shape旧基线即失败，不改容差和测试掩盖。

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
- 服务器生产MLA.py仍4b32历史基线，比较必须显式传stage里的`MLA.baseline-9683-5e3efcc1da.py`，不能误用服务器生产文件。
- harness本地：`D:/workre/HeteMCP/.heteromcp/mla-score96-20261026/fair-benchmark.py`；stage同名。原参考`test_tilelang_mla.before.py`；cases `test_cases_mla_batch_ctx.json`。
- 核心`--ids 1,2,3,4,5,6`，7轮×100次、预热200，seed0/42；rtol2e-4、atol1e-4、mismatch_ratio0.01，双方预分配输出。完整31点`--ids 1,...,31 --trials 5 --repeats 20 --warmup 20`。GPU作业串行。
- 旧`MLA-validation.py`要求baseline.ref_program，不能直接配纯算子baseline。生成代码/IR文本不等于硬件spill/吞吐证据。
- IR观测用DSH `inspect-ir-builtin.dsh.py`：开启JIT内置`tl.enable_dump_ir/tl.dump_ir_path`；**外层PassInstrument会被JIT自己的PassContext屏蔽，不再采用**。编译/导出不执行kernel，但仍不同于真实OJ编译。

开局操作：核对生产SHA→读最新报告→选一个有证据支持的修复设计→真实MCP委派DSH→审阅验证；不要恢复历史失败版本作为“修复”。
