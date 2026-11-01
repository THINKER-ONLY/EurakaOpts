> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA 分段 profiling 与 64 线程合并优化

2026-10-27。已完成服务器六点分段 profiling、DSH 优化、31 组 GPU 回归及两端保存。新候选位于 [MLA.py](D:/workre/tzb/MLA.py)。当前只有旧版本的实际 OJ 得分 **96.67**；新候选尚未提交 OJ。

## 主要发现

短序列应关注合并阶段，长序列应关注主计算阶段。在这台 SSH 服务器当前编译环境里，第 1 点的合并约占 GPU kernel 活跃时间的 **29.15%**，第 6 点的主计算约占 **99.05%**。

本轮针对短序列合并开销做了一个独立改动：并行合并 kernel 从 128 线程改为 64 线程。原主计算、无 split 路径、串行合并后备路径、split 调度、数值公式和公开接口全部保持一致。AST 对比确认，恢复这一个线程数常量后，整个模块与 96.67 分源码一致。

未开启 profiler 的 A/B 测量中，新候选前两点耗时下降约 **4.69%、3.72%**，六点加速比几何均值 **1.01946×**。31 组原始配置、种子 0 和 42 全部通过，31 组加速比几何均值 **1.00686×**；最慢一点的实测回退约 **0.21%**。

## 工具选择与硬件指标状态

服务器是 MetaX C500 的 50% sGPU 分片，MACA 3.5.3.20，使用已有 TileLang/MACA 构建和 PyTorch 环境。选择现有原生 `/opt/maca/bin/mcTracer` 获取 Runtime API 与 GPU activity 时间线，另用未追踪的 GPU events 和主机时钟交叉检查。没有安装新工具链，也没有修改驱动、sysctl、容器权限或编译限制。

硬件计数器没有产出可用数据，具体证据如下：

- 正确 MCPTI Profiler API 的 Initialize、DeviceSupported、GetCounterAvailability 和 DeInitialize 均返回 0；可用性镜像大小为 2472 字节，但设备支持字段保持 `UNKNOWN`。这不能证明计数器已可采集。
- 另一个独立进程按 SDK 的 profilerScope 配置尝试 Runtime `mcProfilerInitialize/Start/Stop`，均返回 0，执行 5 次真实 MLA 调用后仍没有生成计数文件。
- 因此本报告没有 SM 利用率、带宽利用率、指令吞吐或实际 occupancy 计数。也不能将此结果表述为“服务器明确禁止硬件 profiling”。
- 原生 activity tracing 已成功，包含真实 GPU 时长、CPU launch 关联、grid/block 和资源元数据。这是本轮归因和优化的依据。

早期探测器曾使用错误的三个 MCPTI 符号名。该结果已排除，不能视为权限证据；DSH 编写的兼容层修正后才得到上述结果。

工具功能参考：[沐曦官方性能分析工具文档目录](https://developer.metax-tech.com/doc?category_ids=61)。服务器实际版本与 ABI 以已安装 SDK 的头文件、示例和工具帮助为准。

## 分段结果

每个形状先经过双种子数值检查和预热，再执行 200 次连续调用。分析器只选择末尾 200 对 MLA kernel，排除参考实现、初始化、预热以及 `Launch` flow 箭头。两阶段必须交替、时长为正，且位于同一 GPU queue/thread；六点检查均通过，没有观测到选中事件之间的重叠。

以下为原 96.67 分源码的采集结果。时长是 200 个样本的中位数；合并占比由时长总和计算，以两阶段的活跃时间之和为分母，未包含间隔。

| 点 | KV context | 主计算 / μs | 合并 / μs | 合并占活跃时间 | 未追踪整次调用 / μs |
|---|---:|---:|---:|---:|---:|
| 1 | 2048 | 14.592 | 6.144 | 29.15% | 26.406 |
| 2 | 4096 | 29.952 | 6.912 | 18.88% | 42.959 |
| 3 | 8192 | 99.072 | 6.144 | 6.19% | 110.692 |
| 4 | 16384 | 182.272 | 6.144 | 3.46% | 194.173 |
| 5 | 32768 | 348.544 | 6.144 | 1.76% | 360.376 |
| 6 | 65536 | 680.960 | 6.144 | 0.95% | 693.041 |

未追踪整次调用列为直接调用已编译 kernel 的 GPU-event 批次中位数，每批 100 次，共 7 批。整次调用包含两个 GPU kernel、其间隔及原包装器行为，event span 也可能包含主机提交间隔，不能直接与两个 kernel 的中位数相加比较。

采集态两 kernel 间隔及相邻调用之间间隔的中位数均为 2.816 μs。这是追踪状态下的调度/排队观测，不等于可直接消除的算子开销。第 1 点 trace 批次跨度约 31.34 μs/次，而未追踪 event 测量约 26.41 μs/次，说明 profiler 对短任务有明显扰动；性能收益统一用独立、未开启 profiler 的 A/B 测量判断。

CPU 提交耗时与 GPU 执行也分开记录。直接调用的 CPU 提交中位数约 11–12 μs/次，经过 `run_kernel` 包装约 14–15 μs/次；提交、排队和 GPU 执行可以重叠，不能相加作为总延迟。CPU launch 通过实际 `co_id` 与 GPU 事件关联。

### 资源线索

| 配置范围 | 主计算寄存器/线程 | 主计算动态共享内存 | private_total 原始字段 | private_per_thread 原始字段 |
|---|---:|---:|---:|---:|
| 第 1 点 | 124 | 55296 B | 0 | 0 |
| 第 2 点 | 122 | 55296 B | 0 | 0 |
| 第 3～6 点 | 256 | 57344 B | 252 | 0 |

合并阶段动态共享内存均为 512 B；第 2 点每线程 80 个寄存器，其余五点为 42。主计算与合并的 block 均为 128 线程，合并 grid 为 16×8×1。

长序列的寄存器数达到 256，并报告非零 `private_total`，是下一轮检查寄存器压力和临时值生命周期的线索。但 `private_per_thread` 同时为 0，本轮没有读取 spill 计数或访存计数，不能据此量化 spill 字节/流量，也不能断言内存带宽瓶颈。

本轮测得的是两个完整 kernel 的分段耗时，尚未细分主计算内部的 QK GEMM、softmax、PV GEMM 与 KV 加载耗时。对这些内部环节的热点判断仍待更细的计数或插桩证据。SSH 环境的编译结果和耗时也不能直接等同 OJ 环境。

## 优化与验收

实现由 DSH 经 PostgreSQL-backed HeteMCP 生成。并行合并阶段没有 MMA，独立尝试一个 MACA warp、64 线程的归约映射；保留 tile64 和原归约公式。该改动是否减少同步成本是待验证机制，本轮可以确认的是总体延迟小幅下降，未采集新候选的逐指令计数。

七轮交替 A/B、每轮各 100 次、输出预分配、编译和参考计算不计时，结果如下：

| 点 | 96.67 基线 / μs | 64 线程候选 / μs | 加速比 |
|---|---:|---:|---:|
| 1 | 27.172 | 25.897 | 1.04923× |
| 2 | 43.581 | 41.958 | 1.03868× |
| 3 | 111.283 | 109.629 | 1.01508× |
| 4 | 194.650 | 193.029 | 1.00840× |
| 5 | 360.640 | 359.252 | 1.00386× |
| 6 | 692.844 | 691.162 | 1.00243× |

前两点七个配对批次的加速比全部大于 1。第 5 点有一个批次略慢，全部原始样本均保留，没有删除离群值。

随后复用原 DSH 验收脚本运行原始 31 组配置，种子 0、42，与未修改的原始参考实现比较；输出用 NaN 初始化并检查全部有限。使用原有 `rtol=2e-4`、`atol=1e-4`、最大不匹配比例 1%。31/31 通过，基线也全部通过，且均有有效计时。该轮每点 5 批、每批 20 次，前两点分别得到 1.05037×、1.04398×，与独立六点测量方向一致。

静态检查确认仅并行合并的线程数发生变化，模块全局赋值通过 literal 检查，导入仍只有 TileLang；没有更改测试、容差或 runtime/compiler 限制。额外极短 context 或非标准 head 配置的既有问题没有在本轮修复或扩展验收。

## 保存与复现证据

- 新候选：[D:/workre/tzb/MLA.py](D:/workre/tzb/MLA.py)；SHA-256：`4b32b1ccfb3c6cb084cbb01f49ad6732912aa378db6d1b6e1eece968da9f1cba`。
- 96.67 分版本备份：[MLA.score96p67-20261027-189eac4be0.py](D:/workre/tzb/MLA.score96p67-20261027-189eac4be0.py)；SHA-256：`db89ab8fe60aae61a3bdbba5cff5da8c4d596271f20109b49ebccdd62e6541e3`。
- 服务器新候选：`/data/tilelang-metax/race_tests/mla/MLA.py`。
- 服务器备份：`/data/tilelang-metax/race_tests/mla/MLA.score96p67-20261027-189eac4be0.py`。
- 发布前确认两端仍是原版本，创建独立备份后才替换；发布后核对两端与 DSH artifact 字节一致，原始测试及配置哈希保持一致。
- 新候选尚未提交 OJ，不预测分数或显示分是否变化。

本地证据根目录为 `D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/`：

- [六点分段分析 JSON](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/all-cases-analysis.json)，含每点 200 对原始样本、阶段份额、间隔、CPU 关联和资源直方图。
- `native-v2-c1.json` 至 `native-v2-c6.json` 为完整原始 trace，`v2-c*.device.cu` 和 `v2-c*.host.cc` 为对应编译源码；所有文件带检索和哈希记录。
- `v2-timing-c*.json`、`v2-trace-c*.json` 为 DSH 驱动报告。
- [正确符号的 MCPTI 探测](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/native-probe-corrected.json)与 [SDK 原生计数器尝试](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/v2-counter-c1.json)。
- [六点 A/B](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/merge64-fair.json)、[31 组验收汇总](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/merge64-full.supervisor.json)、[发布记录](D:/workre/HeteMCP/.heteromcp/mla-profile-20261027/publication-189eac4be0.json)。
- `all-cases-analysis-manifest.json`、`profile-driver.dsh-snapshot.py`、`analyze-trace.dsh-snapshot.py` 和下列兼容层 artifact 可用于重新分析；使用新的输出文件名，避免覆盖证据。

调度记录：

| 用途 | task | attempt / 状态 |
|---|---|---|
| 首版 profiling 驱动 | `38581bee-02fe-46d0-a7a8-6b48de2a9c72` | `e1211ee0-e9f5-435c-ab84-e6d7445d3b6c`，600s 超时，保留 |
| 中断的恢复任务 | `9d4db2e8-d745-47ac-acfc-13e2a44cc33d` | cancelled；恢复后未复用已取消状态 |
| 精简驱动 | `fdc73952-3e49-4898-ab75-2c04deb4a735` | `127d9415-dfde-4b68-b3e0-08986f0fc60a`，succeeded |
| 时间线分析器 | `2e66b997-11af-4384-84c1-b9fdc053cf60` | `fda187e9-2030-45c9-8f44-11bbb12042e1`，400s 超时，使用其冻结的 DSH 源码加下列兼容层 |
| 精确兼容修复器 | `4723a881-f5b5-47f7-b4a6-6926d87d2300` | `970e3452-0a7a-42dc-8e80-f89945736969`，succeeded |
| 64 线程合并候选 | `fefaedc0-7175-49e0-8dd6-09e6152a7e62` | `da5266d2-bd70-428b-bc28-72bbc454111f`，succeeded |

分析兼容层修正了向量化 C++ 指针写出识别，并保留原始阶段身份验证；探测兼容层修正三个 MCPTI 符号名。两者均由 DSH 编写，原实现文件未手改。完整实现、分析修复和优化代码均有 DSH 来源；Codex 负责调度、审阅、运行、归因、报告与原样保存。

`succeeded` 只代表 Harness 文件范围检查通过；GPU 和分析结果由独立运行验证。用量 `measured=false`，费用/token 未准确计量。`.heteromcp/` 被 Git 忽略，Git 提交或推送不会自动备份这些 trace/产物；本轮没有提交或推送 Git。
