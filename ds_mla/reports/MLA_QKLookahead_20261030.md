> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# 两组寄存器缓冲 QK lookahead（2026-10-30）

固定生产基线5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c，OJ96.83、六点20/26/39/64/114/211μs总474μs。上轮QK32全部literal展开出现private104字节/B4修正仍20，性能失败。此轮只保留两组4half Q/KV操作数，预取下一步sharedload，在16次serial pair循环内按原顺序做32次MFMA；原shared布局、KV一次读取、128threads、原PV/B64与PE/softmax/merge保持。

真实HeteMCP→DSH task59c7c795-416f-465e-87e6-6df846db1f3b / attempt694de82e-9a15-4fc0-b837-3205305f6c48，正式范围验收成功。产物SHA422e8313cb4753d94564299768177bf9b9c2f04734f25494d1e31b4d59e976f8。母版是已正确但慢的4f046b2d70源码静态QK版本，只有newQKmacro和qk_a/qk_b4→8两处alloc不同；macro和alloc规范化还原后整文件AST等于母版。没有堆叠B4PV或完整KV双缓冲。与生产5e3比较，不把母版误称最佳。

数据流：prologue读取k0至bank0；每pair先读取k=2pair+1到bank1，MFMA bank0，若非末pair则读取k=2pair+2到bank0，MFMA bank1。16次pair合计32次16维MFMA。安全AST解释器逐128threads完整执行，仅允许指定加载/serial16/uniform if/MFMA，验证32次操作数顺序、每thread256half逻辑加载及全部地址范围正确。其他函数/布局/数值规则不变，仍需实际编译与GPU数值验证。

约束：不启用显式展开或编译器patch，新增寄存器缓冲8+8half，期望限制操作数存活；这不是已证明的速度/occupancy改进。SSH0.1.9/50%C500，OJ0.1.10，不预测OJ分数。编译资源导出进行中。

## 编译资源

双方2048/8192/65536共6次compile/export成功，SSH作业8a6eedb6逐exec0。新路径中长shared57344不变、mtreg244/streg28/private0，与生产基线同值，限制寄存器存活目标在资源元数据上实现（不是硬件计数/occupancy证明）。原全部展开4f的private104在本版不再出现。profileJSON SHA3069b84d690bb0ff1c570594b146bf41fccf954f6d08a4ac5055f81416305f54，ZIP97d56a0b9950c25e9db9060fea979402ecb6cb548f39565b387c572bc5b3539e。核心六点配对A/B进行中。

## 核心性能否决

6/6双seed原容差finite过，但第4点124.388→145.044μs，最长436.314→514.967μs，启用新路径慢约16%–18%，否决，不跑31、不交付OJ。资源字段相同不代表速度相同，也不能仅凭本结果唯一归因访存指令或调度。核心JSON SHA50aac9dea2a3d766627d85ae8a21e8eb8efe06c134eee2c4e72a1d5521818b670，SSH作业4a2e6885逐exec0。此路线所有任务终态；后续独立尝试把在线分母sum跨线程归约延至split末尾，优先减少动态同步而非再改QK。
