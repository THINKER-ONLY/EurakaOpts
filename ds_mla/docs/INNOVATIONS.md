# 可用于汇报的技术创新

本文的“创新”指本项目中经设计、实现或验证的优化思路，不宣称学术首创。
所有实现来自历史真实 HeteMCP→DSH 任务或用户提供的最佳源码；本轮只归档和解释。

## 1. 用 shape 预算控制 split 并行度

MLA decode 的query短、KV长，单个CTA串行扫描可能不足以填满设备。
将KV区间拆给多个CTA，再按split LSE稳定合并；split过多又会增加中间输出和merge成本。
`_select_num_split` 由 `base_grid=batch*ceil(heads/16)` 推导split，限制至少32个key、最多128个split，
并用8388608个FP16 partial元素（16MiB）控制scratch预算。当前short/long target均64，不能照旧源码注释写成128。

区间边界为 `floor(N*s/S)` 到 `floor(N*(s+1)/S)`，保证精确覆盖；tile尾部独立mask。
这同时处理并行度与正确性。当前实现中部分注释保留历史128 target，实际常量和执行逻辑为64。

代码：[split选择](../code/MLA.py#L51)、[精确分片](../code/MLA.py#L533)。
证据：[历史报告](../reports/MLA_optimization_history_and_MCP_trace_20261027.md)。
贡献表述：**shape 驱动的并行度与临时内存预算协同选择，是71.5→96.17阶段的主要变化。**

## 2. Hybrid parallel merge 与64线程归约

split输出保存FP16 partial和FP32 LSE。合并使用最大LSE减法后求指数权重，避免直接指数溢出。
split32—128采用tiled parallel merge；较小split保留串行路径，避免并行固定开销。
输出按64列切块，merge线程再由128减为64。

代码：[并行merge](../code/MLA.py#L708)。
证据：[merge报告](../reports/mla-score96-merge-optimization-20261026.md)、[merge64](../reports/mla-profile-and-merge64-20261027.md)。
贡献表述：**hybrid merge 对应96.17→96.67；merge64仅确认本地约1.95%收益，OJ仍96.67。**

## 3. 跳过在线 softmax 的 identity rescale

在线softmax维护running max，旧输出乘 `alpha=exp2(m_prev*scale-m_new*scale)`。
若 `alpha` 精确等于1，512维输出向量的缩放可直接跳过；分母更新和max逻辑保持原式。
不是用近似阈值省略乘法，也没有改变单tile路径。

值分布profile中，排除首tile后约85.335%的观察状态满足exact-one。
该profile借助torch计算，仅描述值频率，其matmul顺序可能与MMA不同，不能把85%称为运行时间下降。
配对A/B确认六点几何平均1.04950×，中长context约1.066—1.082×。

代码：[条件缩放](../code/MLA.py#L594)。
证据：[值频率](../evidence/mla-deep-20261027/softmax-state-profile.json)、[配对A/B](../evidence/mla-deep-20261027/conditional-rescale-fair.json)。
贡献表述：**利用在线状态的精确恒等条件，消除高维无效工作。**

## 4. 源码静态 PV：编译兼容性与资源下降

固定dim512/M16/N32的专用路径：显式FullCol布局，kernel作用域只分配一次A4/B64 local，
静态写出加载和16个原生MFMA调用，ki2保持serial。其他shape保留原GEMM路径。
这样不依赖IR阶段的显式循环体复制，避免沿用已知OJ故障方案。

在SSH编译器的中长context样本中，mtreg256→244、streg38→28、private元数据252→0，
attention FUNC代码19456→16640字节；31点配对几何平均1.504439×。
用户反馈OJ通过，但总耗时只由旧490μs变为483μs，仍96.67分；单次OJ微差不能确定稳定收益。
**private字段下降不等同于测得动态spill减少；没有有效硬件counter。**

代码：[PV宏](../code/MLA.py#L143)、[完整静态PV原件](../code/MLA.optimized-source-pv-20261028-def723b6f9.py)。
证据：[完整报告](../reports/MLA_source_explicit_pv_20261028.md)、[ELF元数据](../evidence/mla-safe-pv-20261028/elf-metadata.json)。
贡献表述：**用源码层静态索引表达稳定的数据布局，取得真实OJ编译通过与本地资源改善。**

## 5. PV stream B4：缩短临时 fragment 活跃区间（待OJ）

保留QK、KV一次加载和输出分摊，仅把PV的B64改为B4分组加载/消费，不增加KV全局读取。
中长mtreg244→218，shared57344/streg28/private0保持；完整31点双seed通过。
核心六点几何平均1.014237×，部分轮次无收益；不能宣称显著提分。

代码：[候选](../code/MLA.candidate-pvstream-B4-20261030.py)、[完整diff](../slides/diffs/pv_stream_b4.diff)。
证据：[报告](../reports/MLA_PVstream_20261030.md)。
贡献表述：**候选通过缩短临时存活区间降低寄存器资源，但性能收益小，OJ尚待验证。**

## 6. 延迟分母归约：从每block一次改为每split一次（待OJ）

原分母递推为 `L_t = alpha_t*L_(t-1) + sum_j(P_tj)`。
改为每个score位置维护FP32 `D_tj = alpha_t*D_(t-1)j + P_tj`，split末尾再 `L=sum_j(D_j)`。
数学上利用sum与统一缩放的线性性；浮点加法次序变化，必须实测原容差。
只对ctx>=8192、head16对齐、PV专用且split>1生效，max归约仍逐block执行。

生成代码确认sum AllReduce128移到keyloop外；shared57344→56832，mtreg244→246，private0。
完整31点双seed原容差通过，几何平均1.014005×；核心最长context435.924→427.540μs（1.019610×）。
这比只看寄存器数量更接近减少同步的结构思路，但当前收益仍小。

代码：[候选](../code/MLA.candidate-deferredsum-20261030.py#L608)、[diff](../slides/diffs/deferred_sum.diff)。
证据：[报告](../reports/MLA_DeferredSum_20261030.md)、[31点原始数据](../evidence/mla-deferredsum-20261030/3b20aa59e4-full31.json)。
贡献表述：**在原容差验证下，将softmax分母的跨线程sum从KV循环中移出；尚无OJ收益结论。**

## 7. HeteMCP 的协作与证据机制

Codex负责架构假设、约束、任务调度、静态审阅、SSH/GPU验收与发布；DSH负责生产及诊断代码。
真实链路为MCP SDK stdio client → HeteMCP → Orchestrator → TaskExecutor → AgentHarness → DSH。
PostgreSQL保存任务状态，artifact保存源码；两者不能互相替代。

归档提取了任务/attempt/provider/终态/checks/artifact SHA，见 [51条任务摘要](../evidence/hetemcp_tasks.json)。
`succeeded`只说明文件范围验收通过；usage中`measured=false`的零不能写成“零token/零成本”。
**这是可追溯的分工和验收流程，不能据此宣称已测得优于单agent或减少多少成本。**
