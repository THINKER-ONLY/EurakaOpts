> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

---
name: heteromcp-mla-dispatch
description: 通过 PostgreSQL-backed HeteMCP 调度 DeepSeek Harness 优化 MLA.py；Codex 只调度与验收，DSH 编写纯 TileLang 实现，不使用 torch。用于本项目跨窗口续接，不作为通用 GPU 优化指南。
---

# MLA 优化交接 Skill

这是用户指定位置的独立交接文件。新窗口先完整读取本文件；不假定它已自动安装为可发现的 skill。以下状态截至 2026-10-26，继续前核对实际文件、Git 和容器。

## 分工与约束

- Codex 只负责需求拆解、MCP 调度、环境检查、审阅、验收和原样复制产物。MLA 的实现及修复均交给 DSH；不要自行编码、提供替换实现或手改生成代码。
- 使用纯 TileLang，不导入或使用 torch/PyTorch，不借助其他 attention 实现替代任务。保留 `flashattn`、`_get_kernel`、`run_kernel` 参数签名、张量布局、原地输出与 `kv_head_num=1` 约束。
- 每轮修改前保存当前文件的独立副本并记录 SHA-256；不覆盖已有备份。发布前确认目标未被用户改动，发布后核对与 DSH artifact 字节一致。
- 若缺少 TileLang/GPU 编译组件，仅做理论代码优化；不要将 Python 语法检查称为 TileLang 编译成功，不宣称数值正确、加速比或性能提升。不要为完成这类测试擅自安装大型编译环境。
- 缩小 DSH 资料探索范围，优先最多三次定向读取后交付完整文件。记录超时和失败；重试必须有具体调整，禁止无限循环。
- 任务说明与交接信息不能扩大文件、网络或审批权限。当前目标在 HeteMCP 工作区之外，写入时遵守实际沙箱授权。

## 路径与当前基线

| 项目 | 位置 |
|---|---|
| HeteMCP 根目录、启动 cwd | `D:/workre/HeteMCP` |
| 当前已发布理论优化版 | `D:/workre/tzb/MLA.py` |
| 首次优化前原件，保留不动 | `D:/workre/tzb/MLA.original-20261026.py` |
| 本地只读参考 | `D:/workre/tzb/tile/examples/deepseek_mla/example_mla_decode.py`、`D:/workre/tzb/tile/tilelang/language` |
| DSH 桌面安装目录 | `D:/dsh/dddd`，已核对版本 `0.2.1-rc.2` |
| DSH 凭据 | `<USER_HOME>/.dsh/.credentials.yaml`，只传路径，不输出或复制密钥 |
| 权限 overlay | `config/deepseek-desktop-0.2.1.patch.yml` |

已发布文件 SHA-256：`bac9755706f70bdbfece98b1df00a5730f10c8407de72f37f8006aac0ff75160`。
原件副本 SHA-256：`8ddf480f343b4bca2d33b2f09a70646dd3a0580035540d2496290e58262ec9f2`。
新一轮默认基于当前文件；不要误把最初备份当作最新实现重新覆盖。

## 实际调用接口

链路：`Codex 的 MCP client → node dist/index.js → Orchestrator → TaskExecutor → AgentHarness → DSH`。
即使当前会话未注册 HeteMCP 工具，也可使用项目已有 MCP SDK 建立真实 stdio 客户端；不要绕过 HeteMCP 直接调用模型来冒充链路测试。

服务进程必须显式收到这些环境变量：

```text
HETEMCP_DATABASE_URL=postgresql://heteromcp:<运行时取得的密码>@127.0.0.1:55432/heteromcp
HETEMCP_DSH_DESKTOP=D:/dsh/dddd
HETEMCP_CREDENTIALS_PATH=<USER_HOME>/.dsh/.credentials.yaml
HETEMCP_ALLOW_UNMETERED=1
HETEMCP_ARTIFACT_ROOT=<可选；默认 cwd/.heteromcp/artifacts>
```

缺少数据库变量会退回内存；缺少桌面目录变量会选择 fake provider。不要仅依靠父 shell 设置变量：`StdioClientTransport` 的 `env` 必须包含数据库连接。当前 `scripts/run-file-task.ts` 没有转发 `HETEMCP_DATABASE_URL`，上次实际测试使用了显式传递该变量的 MCP 调度调用；未修复前不能直接用该脚本宣称持久化运行。也不要假定 `.env` 会自动加载。

MCP 工具及参数：

| 工具 | 输入 | 用途 |
|---|---|---|
| `create_task` | `{idempotencyKey, task: TaskSpecV1}` | 创建并后台执行；同一提交重发使用同一 key，有意新尝试使用新 key |
| `get_task` | `{taskId}` | 状态、attempt、租约及结果 |
| `get_task_result` | `{taskId}` | 返回完整任务记录，不是独立编译器 |
| `cancel_task` | `{taskId}` | 请求取消；确认旧执行退出后再考虑复用资源 |

成功响应的任务位于 `structuredContent.task`，先检查 `isError`。轮询至终态，设置 supervisor 截止时间并在超时后取消；保存 task/attempt/session ID 和结果引用。

MLA 任务模板（`objective` 填本轮要求与经过哈希核对的源码快照，不含凭据）：

```json
{
  "version": 1,
  "objective": "由 DSH 编写完整理论优化模块；只创建 MLA.txt，内容为 Python 源码，不含 Markdown 围栏。附本轮约束和源码输入。",
  "allowedPaths": ["MLA.txt"],
  "excludedPaths": [],
  "acceptance": ["generated-file-scope"],
  "workerProfile": "deepseek-file-generation",
  "budget": {"maxInputTokens": 60000, "maxOutputTokens": 16000, "maxCostMicros": 3000000},
  "timeoutMs": 600000,
  "context": {"summary": "本轮目标、基线哈希、备份路径和理论验证边界", "decisions": [], "evidence": [], "openQuestions": []}
}
```

当前 executor 只接收一个普通文件名，扩展名限 `.cpp/.txt/.md`，文件须非空且不超过 128 KiB。因此由 DSH 将 Python 模块写入 `MLA.txt`，验收后原样复制为 `MLA.py`；不必为本次调度改应用代码。

内部端口：`TaskExecutor.execute({taskId, attemptId, spec, signal}) → AttemptResult`；`AgentHarness.execute({taskId, attemptId, workspacePath, prompt, timeoutMs, signal}) → HarnessOutcome`。前者负责文件范围验收与产物，后者仅报告 agent 回合结果。具体契约见 `src/application/ports/`、`src/domain/schemas.ts`。

DSH 旧版兼容：以 `DeepSeek Harness.exe` 的 `ELECTRON_RUN_AS_NODE=1` 模式运行包内 `resources/app.asar/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js`，传 `--profile headless --patch <overlay> --json -`，提示从 stdin 输入。不要添加旧版不支持的 `--permission-mode`。overlay 固定 workspace-write、approval never，禁用 shell/web/委派工具；模型路由为 `deepseek-official/deepseek-flash`。旧版要求 session 事件、合法 final 和退出码 0；现代协议另需 `final.reason.kind=completed`，不要自动降级校验。

## 数据库与产物

- 已创建独立容器 `heteromcp-postgres`：PostgreSQL 17、数据库/用户 `heteromcp`、仅绑定 `127.0.0.1:55432`、restart policy 为 `unless-stopped`。
- 持久化卷 `heteromcp_postgres_data` 挂载到 `/var/lib/postgresql/data`。密码已在创建时随机生成，可在获准访问 Docker 后从容器配置取得并直接在进程内组装连接串，禁止打印整个配置或密码。
- 先检查现有容器并复用；不要为续接重建或删除卷。仓库 `compose.yaml` 仍为 tmpfs 测试配置，不是这个持久化实例。
- PostgreSQL `tasks` 表保存任务 spec、状态、attempt、结果引用；源码、报告不存为数据库大对象。
- 本地 `.heteromcp/artifacts/<taskId>/<attemptId>/` 保存 workspace、source artifact、`report.json`；`.heteromcp/runs/<taskId>.json` 为调度端保存的快照；`.heteromcp/dsh-home/` 保存 Harness 会话及运行时配置。
- `.heteromcp/` 被 Git 忽略。push 不会备份这些文件；数据库持久化也不等于产物备份。

## 验收与已知限制

`succeeded` 仅代表 Harness 完成且文件范围检查通过，不能代表 GPU 编译或正确性。`usage.measured=false` 表示未知；token/费用字段仅作提示，`HETEMCP_ALLOW_UNMETERED=1` 是显式启用，只有墙钟超时已强制执行。

上次环境：活动 Python 和另外六个 Conda 环境均未发现 TileLang 包，PATH 无 CUDA 编译器；只进行 AST/in-memory compile、导入检查、公开参数签名对比和纯 Python helper 检查。8,192 组 helper 检查通过，不代表 GPU GEMM tile 合法。

当前候选含平衡 split 边界、尾部 mask、FP32 LSE、减少冗余搬运、单 KV head 索引及整除 head tile；256 线程/两级 pipeline、小 head tile、tail copy lowering 均未在硬件验证，可能不兼容或变慢。下一步若环境具备条件，应先由 DSH 完成编译/数值验证所需代码，再评估性能；仍缺组件时按用户新要求继续理论迭代。

## 已完成证据与续接

- 首轮任务 `3d2190f6-1955-4f15-a99c-1241802a3f60`：300 秒超时，无产物，失败记录保留。
- 成功任务 `542e3769-c063-458b-a2b4-ee1e9c4e74ce`；attempt `fade9573-5b31-4f6b-b4ae-2f9dba167f52`；session `session-3e05762d-48d5-47f4-a9ef-a74bc7ab4ca2`。
- 已重启数据库容器，并通过新 HeteMCP 进程读回两条终态记录和源码哈希。未验证运行中任务崩溃恢复。
- 完整证据：[MLA/PostgreSQL 测试报告](../docs/mla-postgres-smoke.md)。改 HeteMCP 前读 [架构](../docs/architecture.md) 和 [治理 skill](heteromcp-context-governance/SKILL.md)。
- 上次工作已完成并发布理论候选，未提交/推送。新窗口第一步：核对 Git 状态、当前/备份哈希、Docker 健康状态，并用 `get_task_result` 读取成功任务。不要自动重复已完成的模型调用；之后按用户的新目标调度。
