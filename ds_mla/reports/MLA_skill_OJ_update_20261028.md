> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA 续接补充：OJ 失败与 PV 局部展开（2026-10-28）

本文件覆盖同目录早先`MLA_optimization_skill_current_20261028.md`的候选/OJ状态，角色分工、MCP链路、纯TileLang约束、原容差、备份/并发哈希/原样复制流程继续有效。

- 5f6c03f884eb92dab1dad8eea47411facce93f62704bcbd52f4e99af35b11a70：用户真实OJ提交已失败，在Warmup编译SplitHostDevice阶段，非已观测到数值mismatch。旧skill“尚未提交”过时，不再原样推荐该候选。
- OJ使用tirx路径，SSH使用tir路径，服务器默认Conda无同版已安装TileLang。原错误末尾截断，具体断言尚未知；不要把“全局展开导致”写成已证实的根因。
- 新候选7f5f1fdfa05a363f199d92a1feaab6c6b7d9e09ae4e2783d75b2b2393a1b7813从ef30重新生成，只在两处PV GEMM外增加`T.attr(0, "pragma_auto_unroll_max_step", 256)`；不包含全局tl.UnrollLoop配置。数学公式、split/merge、host包装不变。
- DSH task4560128a-dab1-412f-b6c2-1f7507fab857，attempt619fc739-aca2-41a0-9ba2-e547ba8494bd；源码MLA.txt保存在对应.heteromcp/artifacts目录，禁止手改。
- 新候选六点双seed通过，长四点SSH中位数2.207/2.625/3.029/3.347×；前两点近基线。独立31/31双seed全部通过，完整回归几何平均2.96370×；三shape的host source一致，PV索引局部展开生效，QK/PE及外层循环保持基线结构。证据以本轮acceptance.json为准，不转用5f6版本证据。
- 目标D:/workre/tzb/MLA.py实际是否发布，以目标hash和本轮publication.json为准。失败版本备份名MLA.OJ-failed-global-unroll-20261028-5f6c03f884.py；已知OJ96.67的ef30备份MLA.score96p67-pre-unroll-20261028-ef30c4edde.py继续保留。
- 新候选真实OJ尚待提交。即便SSH全部通过，也不能宣布已修复OJ或已突破97分。
- 下一步优先真实OJ；再失败时收集完整末尾断言和同版编译器信息，不调容差、不写环境侦测掩盖失败，不叠加新的数学/分块机制。

详细报告`MLA_OJ_local_PV_repair_20261028.md`；本轮工作区证据目录`D:/workre/HeteMCP/.heteromcp/mla-oj-unroll-repair-20261028/`，代码检查器是已有DSH版本，GPU测试串行，所有失败保留。
