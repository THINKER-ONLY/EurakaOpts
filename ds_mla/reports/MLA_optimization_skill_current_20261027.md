> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

---
name: heteromcp-mla-optimization-current
description: 从当前 ef30c4... MLA 基线继续通过 PostgreSQL-backed HeteMCP 委派 DSH 优化，并执行 MetaX GPU 验收、备份和原样发布。
updated: 2026-10-27
---

# MLA 优化续接 Skill（当前版）

本文件是项目续接规程和状态快照。历史、任务总账和方法论见同目录 `MLA_optimization_history_and_MCP_trace_20261027.md`。继续工作前必须核对实际文件、数据库、服务器和 Git 状态，不把本文哈希当作无需验证的写权限。

## 1. 当前状态

| 项目 | 值 |
|---|---|
| HeteMCP cwd | `D:/workre/HeteMCP` |
| 当前生产算子 | `D:/workre/tzb/MLA.py` |
| 当前 SHA-256 | `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6` |
| 发布前稳定基线 | `D:/workre/tzb/MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py` |
| 稳定基线 SHA-256 | `4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba` |
| 最初原件 | `D:/workre/tzb/MLA.original-20261026.py` |
| 最初 SHA-256 | `8ddf480f343b4bca2d33b2f09a70646dd3a0580035540d2496290e58262ec9f2` |
| 最后确认 OJ | 96.67；当前 `ef30...` 尚未提交 OJ |
| 当前 DSH task | `6c8c371c-5347-4873-b25d-21e8cccf6f6a` |
| 当前 DSH attempt | `23f60442-c6c4-45c7-9714-2c80d21ef2fc` |
| 当前 artifact SHA-256 | `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6` |

当前实现从 `4b32...` 仅增加 conditional-rescale：在多 tile attention 路径里，`scores_scale[i] == 1.0` 时跳过 `acc_o[i,j] *= scores_scale[i]`。相对 `4b32...`，六核心 shape 几何平均 `1.04950x`，31/31 原始 case 双 seed 通过。

## 2. 角色和不可破坏约束

- Codex 负责需求拆解、HeteMCP 调度、环境/哈希检查、静态审阅、GPU/OJ 验收、报告、备份和原样发布。
- DSH 编写所有生产算子改动和用户要求委派的 profiling / validation 代码。Codex 不手改 DSH 算子 artifact。
- 生产模块必须是纯 TileLang，不导入或使用 torch、numpy、triton，也不替换为其他 attention 实现。
- 保留 `flashattn`、`_get_kernel`、`run_kernel` 签名、张量布局、原地 output、`kv_head_num=1`、mask、FP32 accumulation / LSE。
- 所有模块级赋值必须能被 `ast.literal_eval` 接受；不要重新引入算术表达式形式的全局常量。
- 每个实验从经过 SHA-256 固定的基线独立生成；失败候选不叠加到下一轮。
- 每轮只测试一个机制。若同时改变调度、数学和布局，无法可靠归因，不允许直接发布。
- 发布前创建独立且不覆盖的备份；发布前再次确认目标没有被用户修改；发布后核对目标与 DSH artifact 字节完全一致。
- 不删除或覆盖 `.heteromcp` 旧 task、artifact、trace、JSON 和日志。
- 不把 `HeteMCP succeeded`、AST 通过或 GPU compile 分别误写成 OJ 通过。

## 3. 已有负向证据，默认不要重复

| 方向 | 已知结论 |
|---|---|
| `block_n=64` | 需要约 92-96 KiB shared memory，超过 64 KiB limit，实际 launch 失败 |
| Q / Q_pe register fragment cache | 六点几何平均 0.8859x，拒绝 |
| single-tile softmax shortcut / direct partial write | 六点几何平均 0.9992x，中性，拒绝 |
| N16 / 64-thread / higher split | 六点几何平均 0.9409x，拒绝 |
| split64 -> split128 on longest case | 单轮约 0.68% 改善，证据不足；其他 case 多数变慢 |
| `a*scale-b*scale -> (a-b)*scale` + explicit reciprocal | 六点 0.9890x 至 0.9998x，中性偏慢，拒绝 |
| 重复整核 MCPTI counter 探测 | 未得到有效硬件 counters，低信息量；只有 activity timeline 可靠 |

只有出现新的硬件、编译器、生成代码或 profile 证据时，才重开上述方向。

## 4. HeteMCP 调用要求

调用链必须是真实的：

```text
Codex MCP client
  -> node dist/index.js
  -> Orchestrator
  -> TaskExecutor
  -> AgentHarness
  -> DeepSeek Harness
```

服务进程必须显式接收：

```text
HETEMCP_DATABASE_URL=postgresql://heteromcp:<runtime password>@127.0.0.1:55432/heteromcp
HETEMCP_DSH_DESKTOP=D:/dsh/dddd
HETEMCP_CREDENTIALS_PATH=<USER_HOME>/.dsh/.credentials.yaml
HETEMCP_ALLOW_UNMETERED=1
```

不要在日志、prompt 或报告中输出数据库密码、SSH 密码或凭据内容。数据库密码只可在获准访问 Docker 后从容器配置读取并在进程内使用。

现有持久化实例：

- 容器：`heteromcp-postgres`；
- PostgreSQL 17；
- 绑定：`127.0.0.1:55432`；
- database / user：`heteromcp`；
- volume：`heteromcp_postgres_data`；
- restart policy：`unless-stopped`。

不要用仓库的 tmpfs Compose fixture 替换它，不删除或重建 volume。

## 5. DSH 生产候选任务模板

任务 objective 必须包含：基线完整 SHA-256、完整源码、只允许的单项改动、禁止改动项、现有 profile 证据、已否决方向和外部验收边界。

```json
{
  "version": 1,
  "objective": "DSH 编写完整 MLA.txt；附经过哈希固定的基线和单项实验要求。",
  "allowedPaths": ["MLA.txt"],
  "excludedPaths": [],
  "acceptance": ["generated-file-scope"],
  "workerProfile": "deepseek-file-generation",
  "budget": {
    "maxInputTokens": 40000,
    "maxOutputTokens": 16000,
    "maxCostMicros": 3000000
  },
  "timeoutMs": 600000,
  "context": {
    "summary": "单变量实验及当前基线哈希",
    "decisions": [],
    "evidence": [],
    "openQuestions": []
  }
}
```

当前 executor 的生产源码 transport 使用 `MLA.txt`；成功后 Codex 只能在完整验收通过时把 artifact 原样复制为 `.py`。不要让 DSH 写 sidecar、执行 shell、访问网络或直接发布。

使用同一 idempotency key 重发表示同一提交；有意的新尝试使用新 key。任务超时后必须记录失败，缩小范围或改变策略后才能重试，禁止无调整循环。

## 6. artifact 和数据库留痕

| 证据 | 路径 |
|---|---|
| task 快照 | `.heteromcp/runs/<task-id>.json` |
| source / report | `.heteromcp/artifacts/<task-id>/<attempt-id>/` |
| DSH session | `.heteromcp/dsh-home/` |
| SSH job / log / report | `.heteromcp/mla-ssh-20261026/` |
| 当前深层实验 | `.heteromcp/mla-deep-20261027/` |
| 分阶段 profiling | `.heteromcp/mla-profile-20261027/` |

PostgreSQL 主要保存 task spec、状态、attempt 和结果引用，不保存全部大文件。`.heteromcp/` 被 Git 忽略，所以数据库、artifact 目录和文档三者都要保留。

## 7. 候选审阅门槛

拿到 DSH artifact 后按顺序执行：

1. 核对 HeteMCP task、attempt、artifact URI 和 SHA-256。
2. AST 解析和内存 compile；不要因为 `py_compile` 无权写 `__pycache__` 就误报源码错误。
3. 检查生产导入只含 TileLang。
4. 比较公开签名、模块常量和调用契约。
5. diff 必须符合“单变量实验”；额外顺手优化应退回或否决。
6. 所有模块级赋值执行 `ast.literal_eval` 兼容检查。
7. 检查没有 torch/numpy/triton、参考输出缓存、case ID 特判或容差修改。
8. 上传前验证本地 artifact hash；服务器端上传后再次核对。

## 8. GPU 验收门槛

### 核心六 case paired A/B

- shapes：batch=1、heads=16、kv_heads=1、dim=512、pe_dim=64、context 2048/4096/8192/16384/32768/65536；
- correctness seeds：0、42；
- reference：未修改的原始 `ref_program`；
- tolerance：`rtol=2e-4`、`atol=1e-4`、最大 mismatch ratio 0.01；
- output 预填 NaN，并检查原地写入和全部有限；
- baseline / candidate 都预分配 output；
- warmup 200；
- 7 轮交替 A/B、每轮 100 次；
- 同步 GPU events；
- 保留全部 samples 和 paired trial speedups。

### 完整回归

核心 A/B 有稳定价值后，运行原始 31-case、seed 0/42。要求：

- 31/31 candidate correctness pass；
- 零 fatal / case / validation error；
- candidate hash 与 DSH artifact 一致；
- 不把历史 comparator 的绝对性能当作当前基线 paired A/B；
- 不把早期额外边界 case 的结论扩大到当前未测试范围。

### OJ

SSH 验收通过后仍需真实 OJ。记录每点耗时、显示分、总分、提交源码 SHA-256。OJ 结果与 SSH 方向不一致时先调查环境差异，不伪造推断。

## 9. 发布清单

发布只能在下列条件全部满足时进行：

- 单项 diff 审阅通过；
- GPU compile 通过；
- 双 seed 核心 correctness 通过；
- paired A/B 显示稳定且有意义的收益；
- 31/31 完整回归通过；
- 发布目标 hash 仍等于实验基线；
- 新备份路径不存在；
- 备份后 hash 等于旧基线；
- 原样复制 DSH artifact；
- 发布后 hash 等于 artifact；
- 更新报告，明确 OJ 是否实际运行。

不要覆盖旧备份，不要覆盖服务器原始测试、case JSON 或历史结果。

## 10. 当前最合理的下一步

优先提交当前 `ef30...` 到真实 OJ。它已经完成本地/SSH 接受门槛，继续在没有 OJ 反馈时叠加新优化会降低因果可解释性。

若继续研究：

- 先根据 OJ 六点变化判断 conditional-rescale 是否在 OJ 后端同样有效；
- 长 context 可检查 conditional control flow 的生成代码和寄存器活跃区间；
- 只有获得新的内部阶段证据后，才研究 KV load、QK/PE GEMM、softmax、PV GEMM 的进一步拆分；
- 不以 256 registers/thread 或 `private_total=252` 单独断言 spill；
- 不再重复缺乏 counter 输出的整核 mcTracer 探测。
