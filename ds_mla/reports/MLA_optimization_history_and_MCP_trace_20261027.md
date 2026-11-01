> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA 算子优化全量历史、MCP 留痕与方法总结

整理时间：2026-10-27，Asia/Shanghai。

本文汇总 `D:/workre/tzb/MLA.py` 从最初版本到当前版本的文件演进、PostgreSQL-backed HeteMCP / DeepSeek Harness（DSH）任务留痕、GPU 与 OJ 证据、成功与失败实验，以及可复用的优化方法。详细原始 JSON、trace 和 artifact 仍以文中列出的本机路径为准。

> 安全说明：本文不包含 SSH 密码、数据库密码或 DSH 凭据内容。凭据只在获准执行时经运行时输入或路径引用，不写入任务 prompt、源码或报告。

## 1. 当前结论

- 当前生产文件：`D:/workre/tzb/MLA.py`。
- 当前 SHA-256：`ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`。
- 当前实现由 DSH 生成，Codex 仅负责拆解、HeteMCP 调度、静态审阅、SSH/GPU 验收、备份和原样发布。
- 当前核心优化是在在线 softmax 中，当已计算的 `scores_scale` 精确等于 `1.0` 时，跳过 512 维 `acc_o` 重缩放。
- 相对发布前 `4b32...` 基线，六个核心 shape 的几何平均加速为 `1.04950x`；context 8192 至 65536 的加速为 `1.0656x` 至 `1.0821x`。
- 当前候选完成 31/31 原始 case、种子 0/42 的 GPU 回归，零 validation failure。
- 当前 `ef30...` 尚未提交 OJ。最后一个已确认的 OJ 分数仍为 96.67；不能用 SSH 相对耗时直接推算新 OJ 分数。
- 发布前版本已备份为 `D:/workre/tzb/MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py`，SHA-256 为 `4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba`。

## 2. 证据等级与作者边界

整个项目严格区分以下证据：

1. `HeteMCP succeeded`：只证明 DSH 回合完成且生成文件范围检查通过，不代表 TileLang 编译、数值正确或性能提升。
2. 静态检查：AST 解析、内存编译、导入/签名/全局字面量检查，只证明 Python 源码结构满足要求。
3. GPU 编译与数值检查：证明指定服务器环境下能编译、运行，并在指定 shape、seed、容差内通过。
4. SSH A/B：证明同一服务器同一轮测量中的相对性能，不等同于 OJ 绝对耗时。
5. OJ：只有用户实际提交并返回的分数才记为已确认 OJ 分数。

生产算子约束始终是：纯 TileLang；不在生产文件中使用 torch、numpy 或 triton；保留 `flashattn`、`_get_kernel`、`run_kernel` 的公开签名、张量布局、原地输出、`kv_head_num=1`、mask、FP32 accumulation 和 LSE。torch 只允许出现在独立验证或 profiling harness 中。

## 3. 版本演进总表

| 阶段 | 文件 / SHA-256 | 核心变化 | GPU / OJ 状态 |
|---|---|---|---|
| 最初版本 | `MLA.original-20261026.py` / `8ddf480f343b4bca2d33b2f09a70646dd3a0580035540d2496290e58262ec9f2` | 优化前原件 | 原始备份，保持不动 |
| 首轮理论候选 | `MLA.before-ssh-20261026-a57c1c819a.py` / `bac9755706f70bdbfece98b1df00a5730f10c8407de72f37f8006aac0ff75160` | balanced split、tail mask、FP32 LSE、减少冗余搬运、单 KV head 修正、整除 head tile | 当时本地无 TileLang/GPU，仅静态验证 |
| 初次 SSH 候选 | `MLA.before-validated-20261026-67f9c320b3.py` / `d7770ae66acebe087809fa894a3add58839906f29aed0cb7184f1f993049abcb` | 尝试将理论版落到真实 MetaX 环境 | 256 线程 / MMA layout inference 编译失败，未发布为最终版 |
| GPU 修复版 | `MLA.before-oj-20261026-3012d1240c.py` / `2e16438207a28bf711bc8007c85fcc76b5afb07b0afb7934d031d580fa3d9fbc` | 主路径调整为合法的 128 线程、零级 pipeline | 原始 31/31 GPU 通过；相对最初基线几何平均 2.32345x；OJ loader 后续因全局表达式拒绝 |
| OJ 字面量兼容 / 71.5 分 | `MLA.score71p5-20261026-a43bf064bd.py` / `ff56144e534081f3174685803c593acae5e816c38746e57071c62c39a6e00c48` | `_PARTIAL_ELEM_BUDGET` 从表达式改为等值整数字面量 `8388608`，算法不变 | 用户确认 OJ 71.5 |
| shape-based split / 96.17 分 | `MLA.score96p17-20261026-50119b5426.py` / `3ee71019b6251ed409cc88a6522628ea3ec46572f13dbe8c16d57e3399c39faa` | 用真实扫描结果重写 split 选择，扩大 batch/shape 并行覆盖 | 31/31 GPU 通过；用户确认 OJ 96.17 |
| hybrid parallel merge / 96.67 分 | `MLA.score96p67-20261027-189eac4be0.py` / `db89ab8fe60aae61a3bdbba5cff5da8c4d596271f20109b49ebccdd62e6541e3` | split<32 保留串行 merge，32-128 使用并行 tiled merge | 31/31 GPU 通过；用户确认 OJ 96.67 |
| merge64 | `MLA.pre-conditional-rescale-20261027-4b32b1ccfb.py` / `4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba` | 并行 merge `threads=128 -> 64`，其余 AST 不变 | 31/31 GPU 通过；六点几何平均 1.01946x；用户确认 OJ 仍为 96.67 |
| 当前 conditional-rescale | `MLA.py` / `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6` | `scores_scale == 1.0` 时跳过 512 维 `acc_o` 缩放；单 tile 路径保持原循环 | 31/31 GPU 通过；相对 `4b32...` 六点几何平均 1.04950x；OJ 未提交 |

这些阶段的加速比来自不同对照和不同测量轮次，不能直接连乘作为端到端或 OJ 加速。

## 4. 分阶段优化日志

### 4.1 HeteMCP 持久化链路与首轮理论优化

最初先验证真实链路：`Codex MCP client -> node dist/index.js -> Orchestrator -> TaskExecutor -> AgentHarness -> DSH`。PostgreSQL 使用独立容器 `heteromcp-postgres`，绑定 `127.0.0.1:55432`，数据保存在 named volume `heteromcp_postgres_data`。容器重启后通过新 HeteMCP 进程可读回终态任务，证明了任务记录持久化。

首个 MLA 任务在 300 秒超时；之后缩窄探索范围、延长到 600 秒，只做一次有调整的重试，成功得到 `bac975...`。当时本地缺少 TileLang 和 GPU 编译组件，因此只做 AST、签名、导入和 helper 检查，没有宣称 GPU 正确或加速。

理论候选引入：

- balanced split 边界；
- split / sequence tail mask；
- FP32 split LSE 和稳定合并；
- 减少 split 路径冗余 shared-to-fragment 传输；
- `kv_head_num=1` 索引修正；
- 可整除的 head tile 选择。

### 4.2 SSH 环境构建、真实编译失败与修复

用户授权后，在 MetaX C500 / MACA 环境构建并复用现有 TileLang 源码。初次依赖下载超时，保留失败记录；随后约束 `ml-dtypes>=0.5.1,<0.6` 和 `numpy<2`，复用缓存完成构建。

初次 DSH GPU 候选 `d777...` 在真实编译时触发 MMA layout inference 除零。问题不是 Python 语法，而是 MetaX 64-lane warp、16x16 MMA 与 256 线程布局不合法。DSH 根据真实错误生成 `2e164...`，将两个主路径调整为已验证合法的 128 线程、零级 pipeline。

`2e164...` 在原始 31 个 case、seed 0/42 全部通过，相对最初服务器基线 31 组几何平均加速 2.32345x，最高 15.16860x。额外 5 个边界 case 中只有 `heads16, ctx8193` 通过，另外 4 个短序列或不规则 head case 未满足原严格 FP16 容差；这些限制后续没有被隐去或扩展声明。

### 4.3 OJ 全局字面量兼容修复

OJ 在算子执行前对模块级赋值使用 `ast.literal_eval`，拒绝 `_PARTIAL_ELEM_BUDGET = 8 * 1024 * 1024`。DSH 只把它改为等值整数 `8388608`，得到 `ff561...`；规范化该常量后，新旧模块 AST 相同。用户随后确认 OJ 得分 71.5。

经验：OJ loader 兼容性必须单独验收。零耗时或零加速可能是加载失败，不能误判为算子性能结果。

### 4.4 71.5 到 96.17：用实测重做 split 调度

DSH 先编写独立 tuning harness，显式扫描 split / block 配置，再只修改 shape-based split 选择和常量，不改 attention 函数体。最终策略包括：

- context < 2048 不分片；
- 短 context 或已有多 CTA 时目标 128 CTA；
- 较长 context 且基础网格只有一个 CTA 时目标 64 CTA；
- split 最大 128，每片至少 32 key；
- FP16 partial-output 预算保持 16 MiB；
- 移除旧版 batch > 8 禁止 split 的限制。

相对 71.5 分版，31 组几何平均加速 4.27432x，batch=1 六点几何平均 13.78272x，31/31 双种子通过。用户确认新版本 OJ 96.17。

已明确否决 `block_n=64`：实际需要 92160 / 96256 B shared memory，而设备单 block 上限为 65536 B，launch 失败。后续不应重复该实验。

### 4.5 96.17 到 96.67：优化 split merge

profile 现象显示短 context 的 merge 固定开销明显。DSH 将输出维度按 64 元素分块，计算 split 的 FP32 LSE 权重并并行归约 partial output。初版任务超时，随后做定向 artifact recovery，得到并行原型；完整 GPU 数据显示少 split 时有约 1% 至 1.6% 回退，因此最终使用 hybrid dispatch：

- split < 32：保留原串行 merge；
- 32-128：并行 merge；
- >128：回退原路径。

相对 96.17 分版，六个 OJ shape 几何平均 1.0981x，31 组几何平均 1.0236x，31/31 通过。用户确认 OJ 96.67。

### 4.6 96.67 后的三轮负向实验

三个候选都由 DSH 从同一 `db89...` 基线独立生成，失败候选没有叠加到下一轮：

| 实验 | 六点几何平均 | 结论 |
|---|---:|---|
| Q / Q_pe 寄存器复用 | 0.8859x | 全部变慢；寄存器和布局成本可能抵消 shared load，但未用硬件计数证明机制 |
| 单 tile softmax + 直接写 partial | 0.9992x | 基本中性，不值得增加路径复杂度 |
| 短序列 N16 / 64线程 / 更多 split | 0.9409x | 前两点明显变慢，拒绝 |

原版本 split 扫描也表明：case 1-5 的现有 64/128 split 基本优于替代值；case 6 从 64 增至 128 只有约 0.68% 单轮改善，不足以证明稳定收益。

### 4.7 深层 profiling 与 merge64

使用服务器现有 `/opt/maca/bin/mcTracer` 获得 Runtime API / GPU activity 时间线；没有安装新 profiler 或修改驱动。每个核心 shape 在双种子验证、预热后抓取末尾 200 对 MLA kernel。

| Context | Attention / μs | Merge / μs | Attention 活跃时间占比 |
|---:|---:|---:|---:|
| 2048 | 14.592 | 6.144 | 70.85% |
| 4096 | 29.952 | 6.912 | 81.12% |
| 8192 | 99.072 | 6.144 | 93.81% |
| 16384 | 182.272 | 6.144 | 96.54% |
| 32768 | 348.544 | 6.144 | 98.24% |
| 65536 | 680.960 | 6.144 | 99.05% |

资源线索：case 1/2 主计算约 124/122 registers per thread；case 3-6 报告 256 registers per thread、`private_total=252`。由于没有 spill counter，不能把它解释成已量化的 spill 流量。

MCPTI availability 与 runtime profiler API 返回成功，但没有生成有效硬件 counter 数据。结论是“本环境未取得可用计数”，不是“服务器明确禁止 profiling”。重复整核 counter 探测的边际价值已经很低。

基于短 context merge 占比，DSH 只把 parallel merge 从 128 线程改为一个 MACA warp 的 64 线程，得到 `4b32...`。相对 `db89...`：

- 六点几何平均 1.01946x；
- 31 组几何平均 1.00686x；
- 31/31 双种子通过；
- 用户再次 OJ 后分数仍为 96.67。

### 4.8 代数强度削减：否决

DSH 从 `4b32...` 生成一个隔离候选，只做：

- `exp2(a*scale - b*scale)` 改为 `exp2((a-b)*scale)`；
- normalization 从逐元素除法改为每行显式 reciprocal 后乘法。

六点加速比依次为 `0.9890x, 0.9906x, 0.9973x, 0.9985x, 0.9982x, 0.9998x`。双 seed 数值通过，但整体中性偏慢，因此拒绝，未发布。

这说明“减少源码乘法/除法”不等于真实后端更快；编译器 lowering、舍入次序和依赖链必须通过 A/B 判定。

### 4.9 softmax 状态 profiling 与当前 conditional-rescale

在不直接写生产候选前，先让 DSH 编写值频率 profiler。首个复杂任务 420 秒超时；第二次缩小为单文件聚合统计，成功生成并在服务器运行。

对 seed 0/42、六个核心 shape 的结果：

- 非首 tile head-state 总数：114688；
- running max 更新：16819；
- running max 不变：97869；
- baseline 与 factored 两种写法都精确等于 1.0：97869；
- exact-one 比例：85.335%；
- 非有限状态：0；
- context 8192 / 16384 / 32768 / 65536 的可跳过比例约为 64% / 75% / 84% / 90%。

torch matmul 的累加次序可能不同于 production MMA，因此该 profile 只证明机会频率，不证明分支加速或数值等价。随后 DSH 生成 `ef30...`：在 split 和 no-split 两条 attention 路径中，仅当 `scores_scale != 1.0` 时执行 `acc_o *= scores_scale`；one-tile 编译期路径保留原循环；不带入已否决的 factoring / reciprocal。

相对 `4b32...` 的公平 paired A/B：双 seed、双方预分配 output、200 次 warmup、7 轮交替顺序、每轮 100 次、同步 GPU events。

| Case | Context | 基线 ms | 当前候选 ms | 加速 |
|---:|---:|---:|---:|---:|
| 1 | 2048 | 0.025037 | 0.025073 | 0.9986x |
| 2 | 4096 | 0.041664 | 0.041631 | 1.0008x |
| 3 | 8192 | 0.109637 | 0.102892 | 1.0656x |
| 4 | 16384 | 0.192988 | 0.179574 | 1.0747x |
| 5 | 32768 | 0.359268 | 0.332964 | 1.0790x |
| 6 | 65536 | 0.691689 | 0.639183 | 1.0821x |

case 3-6 的每个 paired trial 都大于 1。随后 31/31 原始 case、seed 0/42 全部通过，候选 hash 与 DSH artifact 一致后才发布。

## 5. HeteMCP / DSH 任务总账

下面按发生顺序列出 MLA 相关任务。完整 spec、终态、artifact URI 与 report 位于 `.heteromcp/runs/<task-id>.json` 和 `.heteromcp/artifacts/<task-id>/<attempt-id>/`。

| 用途 | Task ID | Attempt ID | 终态 | 源产物 / 决策 |
|---|---|---|---|---|
| 首轮理论优化 | `3d2190f6-1955-4f15-a99c-1241802a3f60` | `72823a14-8abd-400b-8cee-34842ea3f67c` | failed / TASK_TIMEOUT | 无源码 artifact；保留失败记录 |
| 缩窄后的理论优化重试 | `542e3769-c063-458b-a2b4-ee1e9c4e74ce` | `fade9573-5b31-4f6b-b4ae-2f9dba167f52` | succeeded | `bac975...`，理论候选 |
| 首个 SSH/GPU 候选 | `a7b2de1e-1e8e-4397-8c34-a8717a0eaaf9` | `9c31be3e-86fd-41dc-a03d-7ed8e9b59d57` | succeeded | `d7770...`，真实 GPU 编译失败 |
| 独立 GPU 验收脚本 | `4257637c-63da-4336-ab3d-9f20ab0c7071` | `86cf2f3e-69cd-465a-bdd2-c995b64097ac` | succeeded | `297e23...`，测试 harness，不是生产算子 |
| MMA layout 修复算子 | `d5d02bcc-b021-4fac-a427-ccb2a1ac1b1d` | `d990891f-8755-4d26-be99-38d125d580ae` | succeeded | `2e164...`，31/31 GPU 通过 |
| OJ 字面量兼容 | `a6355add-eabc-4767-9ffc-10b98cd9c8d4` | `7ac797a6-250e-4cad-ba45-2d2fb277e587` | succeeded | `ff561...`，用户后续确认 71.5 |
| split tuning harness | `fb089b7d-86eb-4c43-a102-ebb07d990af7` | `4b8bc011-8804-4c25-98cd-27111192dc3b` | succeeded | `9fb05c...`，调参脚本 |
| shape-based split 算子 | `bc8f150f-fa66-4633-b75b-15533dd1f8ff` | `acf05220-428f-4438-98a6-4beb798a4f60` | succeeded | `3ee710...`，用户后续确认 96.17 |
| parallel merge 首轮 | `41f7da67-67c5-4abc-b8c7-03e098475ef4` | `0b6d8623-901a-4c24-aaff-29c319c89ede` | failed / TASK_TIMEOUT | 无正式 artifact；草稿用于有界恢复 |
| merge artifact recovery | `4116e596-499e-479b-9c87-21882ed2f03f` | `8240c8df-53c8-4a1e-8847-6d86b36c7dfe` | succeeded | `640ad...`，并行 merge 原型 |
| hybrid merge 最终版 | `a3bae057-e6e9-4663-b030-4825b6d015d2` | `b418f4c4-0b98-4dae-b279-553d195d6d99` | succeeded | `db89...`，用户后续确认 96.67 |
| Q/Q_pe fragment cache | `92c471e9-be30-40ed-bc78-2ff00795df2c` | `957b1a15-58bd-4d66-8edc-4ea1d83833fb` | succeeded | `b7cabb...`，GPU 变慢，拒绝 |
| single-tile shortcut | `15bf9c43-0adb-4f4e-beb4-f873beef21a1` | `a4a6b418-f9b1-4e98-962b-0682438fe70d` | succeeded | `27ee7a...`，中性，拒绝 |
| N16 / 64-thread occupancy | `2eaa37ac-8116-4869-8b1d-e9868ad7ef6d` | `bed92614-b0a2-4814-a155-aefa8c71e642` | succeeded | `e1ad9b...`，GPU 变慢，拒绝 |
| 首版 profiling driver | `38581bee-02fe-46d0-a7a8-6b48de2a9c72` | `e1211ee0-e9f5-435c-ab84-e6d7445d3b6c` | failed / TASK_TIMEOUT | 缩小任务后重试 |
| profiling recovery | `9d4db2e8-d745-47ac-acfc-13e2a44cc33d` | `3099cc76-b89c-4398-a1f2-0636dff7d69d` | cancelled | 中断后不复用取消状态 |
| 精简 profiling driver | `fdc73952-3e49-4898-ab75-2c04deb4a735` | `127d9415-dfde-4b68-b3e0-08986f0fc60a` | succeeded | `f07f9e...`，服务器时间线驱动 |
| trace analyzer | `2e66b997-11af-4384-84c1-b9fdc053cf60` | `fda187e9-2030-45c9-8f44-11bbb12042e1` | failed / TASK_TIMEOUT | 冻结已生成源码，后续用兼容层 |
| profiler 兼容层 | `4723a881-f5b5-47f7-b4a6-6926d87d2300` | `970e3452-0a7a-42dc-8e80-f89945736969` | succeeded | `4b57f5...`，修复 vector store 与 MCPTI 符号 |
| merge64 候选 | `fefaedc0-7175-49e0-8dd6-09e6152a7e62` | `da5266d2-bd70-428b-bc28-72bbc454111f` | succeeded | `4b32...`，发布，OJ 仍 96.67 |
| algebra 候选 | `87d2bf28-1afc-4694-a695-311e029dd025` | `ee1075f6-215e-4745-96c5-6fd285c5b611` | succeeded | `cef156...`，GPU 中性偏慢，拒绝 |
| 首版 softmax 状态 profiler | `2909f61b-41b6-48ed-b117-1ac9acb5f857` | `90d3d7dc-0e68-4fe5-881c-77745d999f5d` | failed / TASK_TIMEOUT | 420 秒超时；不原样重试 |
| 精简状态 profiler | `937c795b-e0f3-4a9d-8141-e22bda224554` | `c5c85421-b7b6-4111-9fc5-da29ed58bb41` | succeeded | `b42955...`，得到 85.335% exact-one |
| conditional-rescale 最终候选 | `6c8c371c-5347-4873-b25d-21e8cccf6f6a` | `23f60442-c6c4-45c7-9714-2c80d21ef2fc` | succeeded | `ef30...`，GPU 全验收并发布 |

在首个 MLA task 之前另有两个 bubble-sort connectivity smoke task：`aa404ef8-e958-4495-bb81-528d73cb3697`（PROCESS_FAILED）和 `4be8fde4-a5ac-4333-ab97-a745eec0091d`（succeeded）。它们只验证 HeteMCP / DSH 通路，不属于 MLA 算子优化结果。

## 6. 可复用优化方法

### 6.1 从 shape 并行度入手，而不是先做微观代数

本项目最大跃迁来自 split 调度，而不是单条指令替换。对 decode MLA，先计算 `batch * head_blocks` 能否填满设备；不足时用 split-KV 增加 CTA，并受以下约束：每片至少一个 `block_n`、partial buffer 预算、merge 成本和合法 shared-memory 上限。

### 6.2 把 attention 与 merge 分阶段测量

短 context 中 merge 可占近 30% 活跃时间，长 context 中 attention 可占 99%。同一种优化不可能对两端同样有效：

- 短 context 优先降低固定 merge / launch / synchronization 成本；
- 长 context 优先优化 KV 循环内每 tile 重复发生的工作。

### 6.3 用频率 profile 验证数据相关机会

conditional-rescale 的关键不是“乘法换加法”，而是在线 softmax 不变量：running max 未改变时，rescale 精确为 1。先量化机会频率，再写候选，避免为低频分支增加控制流。长 context 的机会频率从约 64% 增至 90%，与最终加速趋势一致。

### 6.4 每轮只改变一个机制

候选从经过哈希固定的同一基线生成；factor/reciprocal、conditional-rescale、merge threads 等实验互不叠加。这样 A/B 的因果归属清晰，失败候选不会污染后续版本。

### 6.5 公平 A/B 标准

- 双 seed（0、42）先做正确性；
- reference、容差和 case JSON 不改；
- output 预填 NaN，验证原地写入和有限值；
- 双方预分配 output；
- 编译、随机分配和 reference 计算不计时；
- 足量 warmup；
- 交替 A/B 与 B/A，保留全部样本；
- 同步 GPU events；
- 使用中位数，并检查 paired trial 分布；
- 核心 A/B 通过后才跑 31-case 完整回归。

### 6.6 先做资源合法性，再谈性能

`block_n=64` 因 shared memory 超限直接不可行；256 线程候选因 MMA layout 不合法而编译失败。静态 AST 成功不能替代 TileLang 编译和实际 launch。

### 6.7 不把源码算术简化当成加速

factor multiplication、显式 reciprocal 在源码层看似减少运算，但实际 A/B 中性偏慢。需要考虑 compiler CSE、后端 reciprocal lowering、依赖链、寄存器活跃区间和数值次序。

### 6.8 保存负向证据

负向结果能防止重复浪费：Q cache、single-tile shortcut、N16/64-thread、N64、algebra factoring、盲目增 split 都已有证据。后续若重试，必须有新的编译器、硬件或 profile 证据，而不是换一种描述重复同一假设。

## 7. 当前 skill 的有效状态

原始交接文件位于 `D:/workre/HeteMCP/skill/MLA_skill.md`。它准确记录了职责、HeteMCP 调用链、数据库和发布规则，但其中“当前基线”和若干环境结论停留在 2026-10-26 初期，不能直接把其中的 `bac975...` 当作当前文件。

当前续接必须以以下状态覆盖旧基线：

- 当前生产：`D:/workre/tzb/MLA.py`，SHA-256 `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`；
- 前一稳定基线：`4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba`；
- 最后已确认 OJ：96.67，对应 `db89...` 和用户确认未涨分的 `4b32...`；
- 当前 `ef30...` 的 OJ 分数未知；
- GPU / TileLang SSH 环境已经可用，不再是“只能理论验证”；
- `block_n=64`、Q cache、single-tile、N16/64-thread、algebra factoring/reciprocal 已被否决；
- 继续前应先实际提交当前版本 OJ，或针对 OJ 结果确定下一假设。

更新后的可直接续接 skill 另见同目录 `MLA_optimization_skill_current_20261027.md`。

## 8. 已知限制与未完成事项

- 当前 `ef30...` 尚无 OJ 分数。
- SSH MetaX 环境与 OJ 环境的绝对耗时差异很大，只能用同轮相对 A/B 判断机制。
- 早期 5 个额外边界 case 中有 4 个未满足原严格容差；本项目正式验收范围一直是原始 31-case。后续若扩展通用性，需要单独处理这些 shape。
- MCPTI 硬件 counter 未产出可用指标；目前没有可信的 bandwidth、occupancy、instruction throughput 或 spill 字节数。
- case 3-6 的 256 registers/thread 和 `private_total=252` 只是寄存器压力线索，不是 spill 结论。
- 当前 31-case 完整报告中的性能对照使用历史 accepted comparator，主要用于全形状正确性；判断 `ef30...` 相对 `4b32...` 的收益应使用六核心 paired A/B 文件。
- `.heteromcp/` 被 Git 忽略；PostgreSQL 保存 task spec/终态，不保存所有大体积 trace 和源码 blob。Git push 与数据库持久化都不能替代 artifact 备份。
- 本轮没有 Git commit 或 push。

## 9. 证据路径索引

### 阶段报告

- `D:/workre/HeteMCP/docs/mla-postgres-smoke.md`
- `D:/workre/HeteMCP/docs/mla-ssh-acceptance-20261026.md`
- `D:/workre/HeteMCP/docs/mla-oj-literal-fix-20261026.md`
- `D:/workre/HeteMCP/docs/mla-score71-optimization-20261026.md`
- `D:/workre/HeteMCP/docs/mla-score96-merge-optimization-20261026.md`
- `D:/workre/HeteMCP/docs/mla-score9667-experiments-20261027.md`
- `D:/workre/HeteMCP/docs/mla-profile-and-merge64-20261027.md`
- `D:/workre/HeteMCP/docs/mla-conditional-rescale-20261027.md`

### 当前轮关键证据

- value profile：`D:/workre/HeteMCP/.heteromcp/mla-deep-20261027/softmax-state-profile.json`
- current-baseline paired A/B：`D:/workre/HeteMCP/.heteromcp/mla-deep-20261027/conditional-rescale-fair.json`
- 31-case full regression：`D:/workre/HeteMCP/.heteromcp/mla-deep-20261027/conditional-rescale-full.json`
- DSH 当前 artifact：`D:/workre/HeteMCP/.heteromcp/artifacts/6c8c371c-5347-4873-b25d-21e8cccf6f6a/23f60442-c6c4-45c7-9714-2c80d21ef2fc/workspace/MLA.txt`

### HeteMCP 留痕结构

- task snapshot：`D:/workre/HeteMCP/.heteromcp/runs/<task-id>.json`
- artifact：`D:/workre/HeteMCP/.heteromcp/artifacts/<task-id>/<attempt-id>/`
- harness session：`D:/workre/HeteMCP/.heteromcp/dsh-home/`
- SSH job/report/log：`D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/`
- profiling 数据：`D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/`

## 10. 推荐下一步

1. 先将当前 `ef30...` 提交真实 OJ，记录六点耗时、单点分数和总分。
2. 若 OJ 提升，固定新 OJ 版本快照和哈希，再决定是否继续。
3. 若 OJ 不提升，先比较 OJ 六点变化与 SSH A/B 是否同方向，不立即回滚或叠加新优化。
4. 下一轮长 context 研究应优先观察生成代码中的 conditional lowering、寄存器活跃区间和 KV 循环内部结构；没有新证据时不重复已否决实验。
5. 任一新候选继续遵循：独立 DSH task、单变量改动、双 seed、paired A/B、31-case、备份、哈希并发保护、artifact 原样发布。
