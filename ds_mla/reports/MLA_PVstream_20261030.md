> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# B4流式PV实验（2026-10-30）

基线5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c，用户OJ96.83、20/26/39/64/114/211μs共474。生产文件保持该版。

先审阅generatedsource的同步：PV前barrier保护刚写入S的跨线程读，尾barrier保护下轮KV共享覆盖，未发现能直接删掉的冗余同步，故不实施删同步。独立实验改为PV B临时live range：64half降4half，四次load后立即相应MFMA，共16个源码literal组，ki2仍串行；A4每ki只加载一次，barrier和FP32acc每分量顺序不变，不改变KV读取量/共享布局，不重算QK，无新增kernel或globalbuffer。

真实HeteMCP→DSH task774c955a-a973-49b0-abcf-c97313d315cf，attempt2cf52e13-8834-42eb-a04d-7c11a077e981，正式成功产物SHA01a2c12bc515ac49dddffd2569b86d6a51b29c9331ac8a9877ee9460505558a9。Codex未修改DSH产物。审阅逐组对照原64个加载rhs与16条MFMA参数：只改Bbuffer索引和Bvector0，输出accvectorj/A0原样。规范化这些批准变更后整文件AST与基线完全相同。

核心六点双seed原参考/原容差测试进行中。只有测量资源后才能声称降低物理寄存器；compiler可能重排，不能预先假设B4等于实际mtreg下降。SSH仅筛选，不保证OJ收益。

## 核心结果

六点双seedfinite、原容差通过。7轮×100，双方预热200，原DSH fairbench。

|点|基线μs|候选μs|中位数加速|七轮配对最小|
|---:|---:|---:|---:|---:|
|1|24.571|24.243|1.013516|0.976874|
|2|41.539|40.568|1.023916|0.971613|
|3|72.136|71.406|1.010218|0.978395|
|4|124.641|123.062|1.012835|0.992706|
|5|228.434|225.485|1.013079|1.006255|
|6|434.668|428.751|1.013799|1.010738|

收益很小且前四点并非每轮加速，不能声称稳定提分。第5/6点配对全部快，继续做完整31点与资源检查，避免仅用中位数判断。核心JSON SHA c880711df9313936815852798fc374de60af865f1cca35aceba5ae90b52840cff；作业mla-pvstream-01a2c12bc5-core-20261030-95fec1d9.json。

## 完整31点

31/31双seed原容差均通过，5轮×20次、双方20预热。几何平均1.014237×，最小1.006185×，最大1.045678×；第4点1.006185×。收益幅度小，整体一致性仍不证明OJ收益。完整JSON SHA `1abee615fed209de98d09961fe33e845d7200c5b5723890666718c2739f728268`，作业 `mla-pvstream-01a2c12bc5-full31-20261030-d25ea1ff.json`（step0执行，step2为资源导出）。

## 编译资源与交付

复用原DSH exporter和ELFmetadata解析器，无改动。context2048/8192/65536，双方共6次compile/source/IR/host/mcbin导出全部成功，ELF6文件12kernel解码无错误。中长attention mtreg244→218，streg28不变、private0不变；2048仍mtreg122/streg28/private0。merge均78/22/0不变。动态共享内存2048为55296、8192/65536为57344，双方不变。说明B存活范围改变影响了旧编译器的寄存器分配，但不是动态occupancy/吞吐硬件计数，更不证明OJ0.1.10会同样变化。

profile JSON SHA fae49caaa65ea061dd36e93314197dd7a71218b9b6f7fe4b3e11f8347876626c0，ZIP SHA37c22224fff6cfd698a2ae8bc0e39caf50ccf764f7eb1207863ea405c2fb55f6e。资源/证据路径 `.heteromcp/mla-pvstream-20261030/elf-metadata.json`、`profile-extracted/`。

本轮只交付独立试验文件 `D:/workre/tzb/MLA.candidate-pvstream-B4-20261030.py`，SHA01a2c12bc515ac49dddffd2569b86d6a51b29c9331ac8a9877ee9460505558a9。MLA.py仍用户最佳96.83基线。OJ待测，可比较六点耗时，但不保证微小SSH收益跨评分档或在OJ有效。若无收益不再扫B8/B16等相同机制。所有DSH/GPU任务终态，原31点之外不扩大正确性结论。
