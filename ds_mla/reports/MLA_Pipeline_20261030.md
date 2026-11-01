> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# Split attention 软件流水实验（2026-10-30）

目标：独立于已否决的warp-local路线，保留128threads、N32、原静态PV/B64、KV一次加载复用，尝试隐藏下一块全局加载等待。固定基线SHA5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c，OJ96.83，六点20/26/39/64/114/211μs，总474μs。

真实HeteMCP→DSH task d14e15b6-0096-4f1b-b117-7175e720dda5 / attempt a862c41f-c29f-4acf-b394-6e3309ff9123，正式范围验收成功。候选SHA15af3e0786fa4fad784afe5923ae8bcc6d35c7715fcdf85625968949fe7a00da。Codex未手改DSH源码。

允许差异：factory新增split_pipeline_stages，只在pv_specialize、heads16对齐、pe_dim64、num_split>1、ctx>=8192、每split至少4块时为2，其余0。main_split的T.Pipelined使用该深度；main_no_split、merge、原算法和模块常量保持。AST去除指定赋值、还原指定loop参数后与基线完全一致。DSH新增注释中的“unaligned splits”措辞不准确：实际条件是头数对齐，split尾部保留原mask。

假设与代价：流水编译必须真的产生下一块加载与计算交叠，不能仅凭参数声称已实现；双缓冲可能增加shared/寄存器和同步或降低并发。未启用显式展开/流水order配置，不修编译器，不改变精度或容差。

编译/资源及核心六点双seed配对A/B待完成；SSH TileLang0.1.9、50%C500，不能代替OJ0.1.10验证。当前生产MLA.py保持已OJ确认的96.83基线。

## 编译资源否决

实际6次compile/export均通过，但8192/65536新attention的动态共享57344→94208，超过已有设备probe与编译target记录的65536字节每块上限。不能把导出成功当可launch。候选生成了双缓冲轮换和prologue/epilogue，未见显式async指令；不据此声称硬件异步加载有效。性能任务不运行，直接资源否决，不跑31点，不交付OJ。编译JSON SHA c9c09a2b226e3854767b2917c6e92f52b774b3122d8473c1a123ac22ac141b62，ZIP856344079085d2613551ed0b0f1245bb494fc344dff4be1f9aca0bdef4f7f067，SSH作业aed7865c全部step0。元数据长ctx mtreg244→224、8192仍244，private0，但无法抵消shared容量问题。所有任务终态。下一条独立设计：保留原shared布局的源码显式QK32MFMA，避免增加双缓冲。
