> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA：OJ 编译失败后的 PV 局部展开候选（2026-10-28）

## 已确认的问题

上一候选5f6c03f884...在SSH环境通过了31-case双seed，且中长序列明显加速，但用户真实提交OJ后失败。OJ报错为`Warmup execution failed`，调用链停在`OptimizeForTarget -> SplitHostDevice`，并非已观测到输出数值不符。用户补充payload中`pass=false`、执行耗时0；第1点参考耗时0.419ms，与历史配置一致。

错误栈使用`tvm::tirx`，SSH源码使用`tvm::tir`；可确认编译器实现路径不同，不能仅凭符号确定准确版本。服务器默认`/opt/conda`无已安装TileLang，只有已构建的`/data/tilelang-metax`源码（commit `ee6db4376484f2f7270183c01fd0d90f794965cb`），因此目前无法在原服务器复现相同OJ编译器。用户两份错误都在C++栈中途被截断，缺少最终Check failed条件，不宣称已经定位到某个确切断言。

上一报告中的SSH性能及正确性仍是真实测量，但不构成OJ接受证据。5f6c...应标记为**OJ编译失败版本**，不继续提交原样源码。

## 修复假设与独立候选

研究从已通过OJ96.67的ef30c4edde...重新开始，移除5f6c...新增的全局`tl.UnrollLoop`配置，将展开范围限制在两处PV GEMM内部：

```python
with T.attr(0, "pragma_auto_unroll_max_step", 256):
    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
```

原jit decorator只保留FAST_MATH。QK、在线softmax公式、外层KV循环、merge、split调度、布局、函数签名和host包装均不改变。目的为保留PV片段索引静态化的收益，减少全局自动展开对其余IR的影响。**这是针对兼容性疑点的有界候选，不等于真实OJ已修复。**

源码全部由DSH经真实PostgreSQL-backed HeteMCP生成，Codex仅提供设计、审阅、测试和原样复制：

- Task：`4560128a-dab1-412f-b6c2-1f7507fab857`。
- Attempt：`619fc739-aca2-41a0-9ba2-e547ba8494bd`。
- Artifact SHA-256：`7f5f1fdfa05a363f199d92a1feaab6c6b7d9e09ae4e2783d75b2b2393a1b7813`。
- Artifact：`D:/workre/HeteMCP/.heteromcp/artifacts/4560128a-dab1-412f-b6c2-1f7507fab857/619fc739-aca2-41a0-9ba2-e547ba8494bd/workspace/MLA.txt`。
- 对照基线：`ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`。
- 与ef30的diff只有两处局部attr scope和说明性注释；没有把分离QK/PV等候选混入。

## 独立核心六点验收

双方输出预分配、NaN初始化、原始reference、seed0/42、原rtol2e-4/atol1e-4/最多1%mismatch；各预热200次，7轮交替A/B，每轮100次，保存所有样本。六点双方数值均通过。

| 点 | context | ef30基线μs | 局部PV候选μs | 同轮加速比 |
|---|---:|---:|---:|---:|
|1|2048|25.549|25.510|1.00151×|
|2|4096|41.111|40.430|1.01684×|
|3|8192|101.801|46.131|2.20677×|
|4|16384|234.708|89.400|2.62537×|
|5|32768|434.573|143.475|3.02891×|
|6|65536|928.361|277.350|3.34725×|

中长四点所有配对样本均>1.8×；短点接近基线，不夸大为稳定改善。不同轮次GPU负载不同，不用前一轮绝对耗时和本轮直接作比。这里没有真实OJ新分数。

## 完整回归与生成代码

继续复用既有fair-benchmark显式传入原始参考，在原始31-case上执行seed0/42、每方预热20、5轮×20次。代码检查复用已经由DSH编写并修复API的source inspector，对context2048/8192/65536分别导出基线和候选的host/device源码，不在检查器中执行kernel或计时。

本候选独立验收结果为 **31/31双seed通过**，所有原始case双方输出均通过原容差且全部有限。完整回归几何平均加速 **2.96370×**，31点中位数加速均>1；最大batch32、ctx65536为4.01884×。完整回归前六点加速1.024、1.032、2.196、2.384、2.564、2.684×，计时批次比独立核心测量短、环境存在波动，不挑选某一轮作为OJ预测。

三shape的baseline/candidate host源码均字节相同；device源码出现的MMA调用文本由3变34（前一全局展开为37），for声明数分别23→19、24→20、24→20。PV动态片段索引循环已被展开，QK/PE循环保持基线结构，外层KV循环保持原样。静态调用出现次数变化来自循环展开，不是实际运算次数或硬件吞吐计数；没有有效spill/occupancy计数。

验收结果见本轮证据目录的`acceptance.json`，完整31点见`local-pv-full31.json`，代码检查见`source-inspection/report.json`。这些是本候选独立证据，未复用前一版的31/31结论。OJ同版编译仍未能本地复现，不将本地通过写成OJ修复已确认。

## 保存、回滚与后续

本轮验收后生产目标为`D:/workre/tzb/MLA.py`，发布时再次核对目标仍为5f6c...，先把失败版本独立保存为`MLA.OJ-failed-global-unroll-20261028-5f6c03f884.py`，再原样复制7f5f...。原96.67版本备份`MLA.score96p67-pre-unroll-20261028-ef30c4edde.py`继续保留。

实际发布以 `.heteromcp/mla-oj-unroll-repair-20261028/publication.json` 和目标SHA为准。报告及续接补充同步到`D:/workre/tzb/log_report/`，历史报告不篡改。

下一步真实OJ重新提交7f5f...。如果仍在SplitHostDevice失败，继续收集末尾断言/同版编译器信息，暂不重复调数值公式或容差；需要立即恢复已知可提交状态时，使用ef30备份。

本轮所有spec、错误原文、环境检查、artifact副本、GPU结果和代码导出保存在`D:/workre/HeteMCP/.heteromcp/mla-oj-unroll-repair-20261028/`。SSH日志仍位于`.heteromcp/mla-ssh-20261026/`。SSH密码及数据库凭据不写入报告、任务prompt或源码。
