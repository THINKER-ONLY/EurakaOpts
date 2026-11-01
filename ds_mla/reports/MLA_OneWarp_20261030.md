> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# 原生单warp attention实验（2026-10-30）

用户96.83/474μs基线SHA5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c固定。上一sharedscore重排softmax方案实测AllReduce64，但新增3个静态barrier和2KiBshared，第3–6点约慢2%，已否决，见mla-warpsoftmax-20261030.md。

本轮不复用shared重排，回到基线：仅原PV专用/heads16对齐/ctx>=8192/split>1时attentionCTA128→64threads。QK FullCol、softmax/reduce表达式原样，让全部score天然在单warp。不改N32、KVcopy一次、QK不重算、无额外kernel/global。PV单warp计算完整512维输出，acc_o每线程128FP32；32条literal MFMA、B4流式加载/ki2串行、A4每ki一次，原同步不变。更多每线程累加器与更少线程带来的吞吐/寄存器压力为明确代价，不预判收益。

真实HeteMCP→DSH taska533e2a8-5f2e-45d0-aa3b-77970a4dc7fd / attemptec952f05-9cde-4b89-8981-2fd7f319121e。成功产物SHA3a835d00d751b59a8b04ddcb1d3bb70557402126b6e12cbf5880a4e67d200ee0。静态hosthelper/main_no_split/原PVmacro/merge/moduleAST不变。逐组核对32条PV加载组B4和MFMA的Bindex0/A0/outputindex0..31，线程布局覆盖全16×512无重叠。源码原样，Codex未改DSH实现。先实际compile/export核对warpGEMM合法性及资源，随后核心6点双seed。

## 实际编译资源

双方ctx2048/8192/65536共6次compile/export成功。单warp M16N32 FullCol合法，不需要编译器补丁。生成QK max/sum归约确认为AllReduce64/stride16；动态共享57344→56832，中长attention .max_block_size64、mtreg244→256、streg28→38、private_memory_size0→496。private为编译资源元数据，不等于实际spill动态次数；但明确没有实现降低每线程寄存器压力。merge均78/22/private0不变。

核心6点原参考seed0/42有限、原容差全通过；7轮×100、预热200、预分配输出。

|点|基线μs|候选μs|加速比|
|---:|---:|---:|---:|
|1|24.609|24.156|1.018758|
|2|40.369|40.530|0.996021|
|3|72.284|176.320|0.409960|
|4|122.975|319.119|0.385357|
|5|228.460|605.576|0.377260|
|6|434.650|1175.229|0.369842|

第3–6点耗时约基线2.4–2.7倍，否决，不跑31点、不交付OJ。单warp带来的更高每线程累加器/执行压力及private memory与退化并存，但未有动态硬件计数，不将全部退化精确归因于spill。初始两个短点沿原128threads路径，小波动不是收益。

核心JSON SHA b42b3d11155d06d7ec27002c81b9f6217894120f71dec0fc8f8f5cc58efae3c0f；profile JSON 898b655382a33392cbf5ce73867ac5194781e2da95fb31aff2e6dfbaab1ab705；ZIP b787532581ef4e93f62cd423f50e790018af52b04b3953f64230260f04e3a061。SSH core作业a45b15bd，profile作业38624657，全部exec exit0，artifact hashes已核对。文件夹 `.heteromcp/mla-onewarp-20261030/` 保存完整证据。

## 架构决策与当前状态

本轮两种warp-local softmax均实现意图但不改善性能：共享桥接增加3个静态barrier/2KiB/shared，约慢2%；原生单warp没有桥接但acc_o每线程128FP32，private字段增加且慢2.4–2.7倍。说明降低归约通信不能独立于输出分片/数据复用/寄存器压力来评估。两个实现都不交付用户测OJ，不用失败候选替换最佳版。

下一轮若继续，应保留128threads输出分摊和KV一次加载，优先考虑无额外全局流量的QK/PE执行结构或数据加载调度；不重复sharedbridge或完整输出单warp，也不因旧编译器收益再次推出OJ可迁移。当前所有DSH与GPU任务终态，生产MLA.py仍96.83基线SHA5e3efcc1da。本轮未得到更好候选。
