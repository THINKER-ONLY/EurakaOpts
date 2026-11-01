> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# Warp内softmax布局实验（2026-10-30）

基线为用户已OJ验证96.83、474μs，SHA5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c。用户称B4PV提升不明显，未提供新OJ耗时/分数，不能把它标记为已知OJ失败或新最佳。

架构依据：旧编译器生成的QK FullCol每行分散在128threads两个64lane warp；softmax max/sum的AllReduce128跨warp共享归约。保持QK/PV/N32、KV一次copy，另将FP32scores[16,32]经2KiBshared重排到soft_s layout：thread=(i//8)*64+(j//4)*8+i%8，localindex=j%4；每warp8rows、每row8lanes，使归约可能只需warp通信。scale/denom/LSE经三份FP32shared16桥接到原acc_o布局，防止不同fragmentlayout错误使用同名row索引。

明确代价：增加FP32score shared读写与跨布局数据交换/barrier。只有实际生成source证明归约变为warp-local且完整调用实测受益才接受；不能把“warp内”当作无需同步/更快的先验结论。只对heads16对齐、ctx>=8192、原PV专用split路径启用，fallback原函数。原mask/FP32acc/LSE/FP16partial、merge/dispatch均保持。

任务：真实HeteMCP→DSH `3368775d-92de-4c4d-88c4-eb5752b7636e`，attempt `adf0708b-34d3-4dd6-b902-8f06bd06cefb`，生成中。Codex只架构、审阅、验证编排和原字节交付，生产/诊断代码归DSH。

## 首版静态审阅与编译失败

产物SHAea90cf69f1c9a6d72e710194c724be9476413acb9e4c239e16d42b01fdfc1ea2。静态module/helpers/PVmacro/main_no_split/mergeAST相同，main_split内新增scorelayout及共享bridge。实际profile exporter双方context2048/8192/65536：baseline3次与candidate2048过，candidate8192/65536失败layout_inference.cc443，原logsum或scores_max不能推断layout。原因定位为新分支不使用旧rowfragments，但旧T.fill仍无条件执行，导致缺layoutconstraints。尚未执行数值或计时，不把compile失败当正确性通过。外层runner status成功，但execstep1 exit1，明确为失败。

定向DSH修复任务2c7860ce-ba0a-4b45-b422-c18e4196e154，attemptdfe0ce5f-e5cb-4463-b00a-89fbeac1eae7；只调整main_split初始化if/else，warp新路径不初始化旧rowfragments。其余算法/布局不变。原失败JSON/ZIP保留，不绕过编译或修改编译器。

## 修复版验证及否决

DSH修复产物SHAb5ea3c3040522db6ce8c28ac7f63354581334922a85828c789f8ed9c721aae24，仅核对指定初始化if/else替换，其余AST与首版一致。实际双方6次compile/export全部成功。生成中长attention的AllReduce由128/stride16变为64/stride8，架构意图生效；但静态barrier声明6→9（不是动态硬件barrier计数），动态共享57344→59456，mtreg244→248、streg28/private0不变；merge不变。shared交换增加同步与寄存器，不能把warp内归约直接视作性能提升。

核心六点、seed0/42原容差finite全部通过；7轮×100，预热200，预分配output。

|点|基线μs|候选μs|加速比|
|---:|---:|---:|---:|
|1|24.233|24.366|0.994537|
|2|41.185|40.899|1.007010|
|3|72.218|73.408|0.983784|
|4|124.298|126.705|0.981008|
|5|228.518|233.349|0.979298|
|6|434.757|445.069|0.976831|

第3–6点慢1.6%–2.4%，否决此sharedbridge实现，不跑31点、不交付OJ。不能精确将退化单独归因于barrier，未采硬件计数。首版失败与修复成功都保留。核心JSON SHA1558df34068bc415424882ba8966a7886b1807b94a44ede327b7d82f90669760，profileJSON ca22d9cb979b159697d8beeadaf7c675897af2fcea7186b7f4f5c9fd8d0fdcbd，ZIP7d3ec81924c4f08c294d6b74998f552f05ebf9906346ac0f2c829ddaec7a4b8a。SSHcore作业65321d86，profile作业33d03395均逐step检查exit0。

## 后续独立设计

不堆叠sharedbridge，回到原96.83基线。另委派原生64thread单warp attention CTA，N32/KV一次加载不变，FullColQK单warp推断layout，softmax原有语义；PV512输出单warp分配128FP32/thread并使用32条源码literalMFMA/B4流式加载。意图移除跨warp归约而不引入score交换；代价为更多每线程寄存器及可能spill/局部吞吐下降。真实DSH taska533e2a8-5f2e-45d0-aa3b-77970a4dc7fd / attemptec952f05-9cde-4b89-8981-2fd7f319121e，生成中。未预先认定warp1的M16N32FullCol编译合法。
