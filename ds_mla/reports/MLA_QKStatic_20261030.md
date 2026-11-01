> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# 源码显式 QK512 MFMA（2026-10-30）

固定用户96.83基线SHA5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c。独立机制，不堆叠已否决的warp-local、QK128分块或双缓冲流水。

本轮前一实验两阶段流水可compile/export，但动态共享94208超过设备65536上限，在资源检查后否决，未执行kernel。此处改为消除原QK512 GEMM的运行时ki32循环/地址依赖：源码显式32组MFMA，原顺序累加，kernel-scope qk_a4/qk_b4复用。原PE64 T.gemm、PV/B64、128threads、N32、KV一次copy、mask/FP32/LSE/FP16partial和merge不变，仅ctx>=8192、heads16对齐、dim512/pe64的split专用。

关键数据流：每warp负责16keys，两warp32keys，每lane负责row=lane%16，4维片段k=r*16+(lane//16)*4+p；Q_shared[row,k]和KV_shared[warp*16+row,k]进入原同序B,A MFMA。acc_s线程映射(j//16)*64+((j%16)//4)*16+i，local=j%4。显式保留原生Q_shared和KV_shared交错布局，物理offset=(j//64)*(M*64)+i*64+((i//4+j//32)%2)*32+((i//2+j//16)%2)*16+((i+j//8)%2)*8+j%8，其中M16/32。这是原编译结果的映射，不是新增数据重排/额外copy。

真实HeteMCP→DSH task f49049e2-f478-488e-aa7a-e53b02fc0146 / attempt f072a4ed-8186-4154-9d8e-83cbe02691df。DSH实现，Codex架构/审阅/编排，不手改产物。正式产物、编译、数值、性能结果待完成。SSA/ELF资源并非硬件计数，SSH0.1.9/50%C500不代表OJ0.1.10实测。不得在OJ成功前覆盖基线。

## 正式产物及静态审阅

候选SHA4f046b2d70dda437c8e41a7e960b5f75a3be2f4f648d7384de097b88de76b887，69789字节。DSH正式终态成功。新增宏32组MFMA/256次标量加载，所有128threads×32步索引逐项审阅正确；去除指定宏/条件/初始化、将原QK分支还原后整文件AST与基线一致。手工复核factory条件、acc_s映射、Q/KV两份T.Layout均与设计一致。布局公式已与原lowered offset逐线程/步骤匹配，并验证16×512与32×512物理映射各自双射。编译作业启动，未据静态审阅声称数值/性能通过。

## 编译及资源

双方2048/8192/65536共6次compile/export均成功，profileJSON SHA898f07874762e34f3d7cd52202f11a1036df5896c5b96c283756f0879e82aa75，ZIP642b544d409519f89a7017f7464cd03a7d8c1dc625e982276c2f64564de570fa，SSH作业ee432bec逐step0。中长attention动态shared57344不变；QK运行时ki32已消除，仅PE ki4保留。生成源码静态MFMA调用点18→49（并非动态执行数量变多），sync声明10不变。ELF metadata mtreg244→256，streg28→38，private0→104字节；这不是动态spill次数。merge未变。资源压力增大，核心计时验证进行中。

## 原版核心否决

核心6/6 seed0/42原容差finite全部通过，但第3–6点慢36%–46%。第4点124.413→175.406μs，最长435.338→635.026μs，否决原版，不跑31点。核心JSON SHA1fb8c332f787fdaa29cbff0f31b008e9a4488711721f21aa2df0f64c5ffe49cd5，SSH作业860953fb逐step0。资源压力增加与退化相关，但未用硬件计数证明因果。

下一次针对性修正：仅newQKpath替换为已验证B4流式PV，削减PV操作数存活占用，其他path/原macro不动；先检查private104是否消失再决定运行。不是预先认定组合有效。

## 定向B4资源修正版

真实HeteMCP→DSH task24f94f91-4e3b-4616-be3c-5603d48f877c / attempt810604cf-b1b6-403a-98f1-17ebadee3bdf，正式产物SHAd8568d95223924dea1b688ff0f786911c24cbbc3353b987c8cc3c760340fe2f2，83903字节。静态AST与4f046母版仅新增pv_mfma_streamB4（与已验证01a2c12参考macro除函数名外全AST一致）、main_split的pv_b在qk_static为4否则64、新路径调用新macro。原QK32/布局/PEmath/原PVmacro/main_no_split/merge/hosthelpers均逐AST不变。第一二点不启用B4，保证资源修正只作用新QK路径。该版对生产5e3基线作比较；4f原版结果用于判断压力修正效果，不混称为生产基线。编译资源导出进行中。

修正版6次compile/export成功，共享57344不变，长ctx mtreg256/streg38仍不变，private104→20，未完全消除。profileJSON SHA f16bc37035cb4d3c59a5654a4dee3a2caa63f9acaab5157b881210daff95bc03，ZIP e64fc467c758a336b659c255641c355cd4f3910a74aeb876c12ebb3ea61601d2，SSH作业67128d4f所有step0。因资源压力部分改善，做最终核心计时验证，不凭private字段推断会提分。

## B4修正版最终核心否决

六点双seed原容差/finite全部通过，7轮×100次、200预热，预分配输出、交替配对；完整调用计时如下（SSH0.1.9/50%C500，非OJ）：

|点|基线μs|B4修正版μs|加速比|
|---:|---:|---:|---:|
|1|24.635|24.771|0.994523|
|2|41.088|40.771|1.007786|
|3|71.785|92.713|0.774271|
|4|124.096|164.984|0.752168|
|5|228.242|308.291|0.740347|
|6|435.136|593.206|0.733533|

核心JSON SHA42cbfb07ec1763340eb6b2e4742cff90b3a791f5a8e49433d036c9143cf7322d，SSH作业a3aef0ce逐step0。相对原版4f有缓解，但启用新路径的第3–6点仍明显慢于生产5e3，不跑31点、不交付OJ。两个QK版本都否决。private104→20不足以恢复性能，不能用此证据唯一归因spill，未采硬件计数。所有DSH/GPU任务终态；生产MLA.py仍5e3、OJ96.83/474μs。

后续约束：不要复跑同样的QK32全部literal展开或仅堆叠B4，更不要把宏源码行数/运行时循环减少等同更快；需控制每线程操作数存活和shared总预算，同时保持KV一次读取。下一架构尚未选定。
