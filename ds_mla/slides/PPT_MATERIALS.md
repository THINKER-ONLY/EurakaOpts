# PPT 内容、讲稿与素材清单

建议12页，面向GPU算子优化与多模型协作汇报。下文为可直接搬入PPT的文案和讲稿，非已制作的pptx。
配套PNG/SVG位于 [figures](figures/)，完整代码在 [code](../code/)，短摘录在 [snippets](snippets/)。
报告覆盖10月26—30日历史实验；整理日期2026-11-01。

## 第1页：结果与边界

标题：**HeteMCP驱动的MetaX MLA Decode优化**

- 最佳用户反馈：OJ 96.83分，核心六点总474μs。
- 纯TileLang，保持接口、原参考与原容差。
- Codex设计/验收，HeteMCP持久化调度，DSH写代码。
- PV stream / deferred sum是待OJ候选，尚未达到97分。

讲稿：先交代可确认的结果，再区分最佳版本和研究候选；成绩不是本次归档重新测得。
引用：[最新状态](../README.md)、[OJ观察值](../data/oj_observations.json)。

## 第2页：算子结构与难点

- Q与KV点积叠加位置编码，在线softmax后得到512维输出。
- KV head=1；核心heads16，context2K—64K，query短而KV长。
- 需要兼顾CTA并行、共享内存、寄存器生存期和跨线程归约。
- 原31点集合限定了正确性声明范围。

图示：`QK+PE → online max/exp → denominator + PV → split LSE merge`。
讲稿：split、PV和softmax归约分别对应并行度、局部数据布局和同步成本。
引用：[创新点](../docs/INNOVATIONS.md)。

## 第3页：Codex / HeteMCP / DSH工作流

配图：[workflow.svg](figures/workflow.svg)。
主文案：**单机制假设 → 固定基线SHA → DSH生成 → 差分审查 → GPU数值/配对A-B → OJ → 接受或否决**。
旁注：任务succeeded只表示文件生成和范围检查；零usage且measured=false不是零成本。
讲稿：职责分离让设计和实现可以独立审阅，任务状态与源码SHA使失败实验也可追溯。
引用：[51条任务摘要](../evidence/hetemcp_tasks.json)、[方法](../docs/METHODOLOGY.md)。

## 第4页：从71.5到96.83的演进

配图：[oj_scores.png](figures/oj_scores.png)、[可编辑SVG](figures/oj_scores.svg)。
里程碑：71.5 → 96.17 → 96.67 → 96.83。
机制依次为loader字面量兼容、shape split、hybrid merge、最佳short target64。
讲稿：不同阶段还叠加了本地有效机制，但不能把不同对照的加速比连乘。
引用：[历史表](../docs/HISTORY.md)。

## 第5页：创新一——有预算的split与merge

- 根据batch/head网格选split，并限制最小key数和16MiB partial预算。
- 整数边界覆盖KV，尾部mask保证分片正确。
- split32—128使用并行tiled merge，较小split保留串行。
- 当前merge64线程、输出64列tile。

代码选用：[split_policy.txt](snippets/split_policy.txt)、[parallel_merge.txt](snippets/parallel_merge.txt)。
讲稿：并行度不是越大越好，需要将attention与merge一起评估。

## 第6页：创新二——消除identity rescale

公式：`alpha=exp2(m_prev*scale-m_new*scale)`；`alpha==1`时跳过512维乘法。
85.335%来自排除首tile后的值频率profile；本地六点几何平均实际改善约4.95%。
代码：[conditional_rescale.txt](snippets/conditional_rescale.txt)。
讲稿：省掉的乘法比例不等于耗时比例，profile和A/B分别回答“能省什么”与“到底快多少”。
引用：[创新点第3节](../docs/INNOVATIONS.md)。

## 第7页：创新三——源码静态PV与编译兼容

- 显式FullCol布局，A4/B64 local，16个静态MFMA，保留serial ki2。
- SSH中长mtreg256→244、private252→0，原31点通过。
- 真实OJ编译通过：96.67 / 483μs；不能搬用SSH约1.50×作为OJ收益。
- 资源元数据变化尚无硬件counter归因。

配图：[static_pv_resources.png](figures/static_pv_resources.png)；代码：[布局与分配](snippets/pv_static_mfma.txt)、[静态MFMA](snippets/pv_mfma_calls.txt)。
讲稿：全局/局部IR显式展开都曾OJ失败，本路线改为源码静态索引，编译兼容是独立验收维度。
引用：[源码PV报告](../reports/MLA_source_explicit_pv_20261028.md)。

## 第8页：反例——SSH更快，OJ更慢

配图：[qk_portability.png](figures/qk_portability.png)、[可编辑SVG](figures/qk_portability.svg)。
QK128分块：SSH第四点124.262→105.859μs（1.174×）；OJ第四点64→96μs。
OJ合计474→712μs，+50.21%，96.83→96.17，最终否决。
讲稿：QK分块后完整KV重载是明确额外成本。资源下降不能证明在另一编译器/设备配置下也能抵消成本。
两环境绝对柱高不能直接比较；图分为两个面板，各自只作同环境基线/候选比较。
引用：[原始OJ反馈](../evidence/mla-9683-20261028/oj712-observation-20261030.json)。

## 第9页：失败实验的工程价值

| 实验 | 观察 | 结论 |
|---|---|---|
| 单warp完整512输出 | private496，约慢2.4—2.7倍 | 减少归约参与线程可能放大寄存器压力 |
| 两阶段完整KV双缓冲 | dynamic shared94208 > 65536预算 | compile/export成功不表示可以launch |
| QK全静态展开 | private104，最长435→635μs | 消除循环可能延长活跃区间并增大代码 |
| QK lookahead | 资源字段相同，最长436→515μs | 资源字段不足以解释运行时间 |

讲稿：失败样本限定了下一轮搜索空间；不重复仅换常量的同机制。
引用：[实验表](../docs/EXPERIMENTS.md)。

## 第10页：待OJ候选——PV stream B4

- 不增加KV读取、不重算QK，B fragment分组即时消费。
- mtreg244→218，shared/private不变；核心6点与完整31点双seed通过。
- 六点几何平均1.014237×；部分轮次无收益。
- 独立交付，保持96.83最佳基线。

代码：[4元素fragment](snippets/pv_stream_local.txt)、[加载后立即消费](snippets/pv_stream_consume.txt)、[完整diff](diffs/pv_stream_b4.diff)。
讲稿：寄存器下降很明显，但收益仅约1%，要避免把资源数字当作提分结论。

## 第11页：待OJ候选——延迟分母归约

原式：`L_t=alpha_t*L_prev+sum_j(P_tj)`。
新式：`D_tj=alpha_t*D_prev_j+P_tj`，split末尾 `L=sum_j(D_j)`。

- sum AllReduce128实际移出keyloop，max归约仍保留。
- FP32加法次序变化，原31点双seed原容差全部通过。
- 最长context435.924→427.540μs，约1.96%；OJ待测。

配图：[candidate_speedups.png](figures/candidate_speedups.png)。
代码：[循环递推](snippets/deferred_sum_loop.txt)、[最终归约](snippets/deferred_sum_final.txt)。
讲稿：这是独立机制，未叠加B4或QK实验。短context未启用新路径，图只展示8K以上。

## 第12页：结论与下一步

- 已确认最佳：96.83 / 474μs；两个候选等待真实OJ验证。
- 有效方法：shape预算、消除恒等操作、源码静态布局、减少循环内归约。
- 核心经验：以实际生成代码、原容差和同轮测量连接设计与结论。
- 下一步：提交指定SHA候选、记录六点OJ数据；有证据才迭代，不覆盖最佳文件。

引用：[最新状态](../README.md)、[数据与证据](../docs/EXPERIMENTS.md)。

## 汇报措辞核对

可说“用户反馈OJ96.83”“指定31点双seed原容差通过”“SSH中该轮配对加速”。
候选标注“OJ待测”；失败标注“否决”。不写“达到97分”“所有shape正确”“成本为零”或“已证明spill/occupancy改善”。
这些材料是工程优化复盘，未做论文新颖性检索或单agent对照实验。
