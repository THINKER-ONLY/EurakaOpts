> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA 显式展开编译错误：定向排查（2026-10-28）

本轮目标是修复优化版本的编译错误并保留性能收益。恢复 ef30 只能提供正常对照，不能作为修复成果。本文覆盖旧 skill 中“5f6/7f5 尚未 OJ 提交”等过时状态。

## 已确认的版本差分

- ef30：OJ96.67；无额外展开配置。
- 5f6：只增加全局 `tl.UnrollLoop(auto_max_extent=16, explicit_unroll=True)`，OJ 在 `SplitHostDevice` 编译失败。
- 7f5：只给两处 PV GEMM 添加局部 `pragma_auto_unroll_max_step=256`，默认仍显式展开，同阶段失败。
- 8465：改为后端 pragma 提示，SSH 正确但几何平均0.979×，性能不合格；未 OJ 验证，不能作为修复版本。

五版的完整 AST 差分已留存在 `.heteromcp/mla-unroll-differential-20261028/variant-comparison.json`。未发现数值公式、mask、split/merge、host 包装等混入修改。两次 OJ 错误均发生于执行前；不能把平台统一显示的 wrong answer 直接解释为数值精度问题。

## 新证据：上游存在同一阶段的明确缺陷

上游 [TileLang PR2290](https://github.com/tile-ai/tilelang/pull/2290)，修复提交 [ac2e1d7918ce25744959cecfed34e5ee24bd8b08](https://github.com/tile-ai/tilelang/commit/ac2e1d7918ce25744959cecfed34e5ee24bd8b08)，2026-05-28 合入。PR 原文：

> LoopUnroller::Unroll copies the loop body via Substitute, which preserves buffer names. After SeqStmt::Flatten, duplicate DeclBuffer statements violate SSA and crash VarUseDefAnalyzer in SplitHostDevice.

后续说明强调，实际问题是多个副本定义了**同一个 Buffer 对象**，不只是文本名称重复。修复对每个复制的循环体执行 `UnrolledBodyDefFreshener`，重新创建缓冲区定义并重映射访问，同时保留 DeclBuffer 的底层 data 别名关系。原有 `ConvertSSA` 调用仍保留；因此“有 ConvertSSA 就不可能有此错误”不成立。

回归测试在两次展开中复制 `T.decl_buffer(..., data=A.data)`，检查 Buffer 对象不同、底层 data 对象相同、各 store 使用各自 Buffer 定义。

该缺陷与本次**触发方式（显式复制）、IR 类型（tirx）、报错阶段（SplitHostDevice）**高度吻合，是目前最强线索。尚未取得 OJ 精确版本、最终断言或原环境复现，不能将匹配写成已证明根因。

只读下载证据：`.heteromcp/mla-compile-rootcause-20261028/{upstream-fix-2290.json,pr2290.txt,pr2290-comments.txt,unroll-fixed.txt,split-current.txt}`。没有安装/修改任何编译器或驱动。

## 为什么之前 SSH 成功不足以排除此缺陷

SSH 用 tir，OJ 栈用 tirx。已检查的本机 tir `FlattenBuffer` 会去掉 DeclBuffer，并且其 UnrollLoop 没有上述 freshener。即使旧环境展开能编译，也不能证明 tirx 环境中的 Buffer 定义合法。需检查实际 pass 前后 IR，而非继续重复相同数值测试。

## 正在执行的定向诊断

通过真实 HeteMCP→DSH 生成只读 PassInstrument 工具，采集 ef30/7f5 的 UnrollLoop 前后与 SplitHostDevice 前 IR，记录各层循环内的 Buffer 定义、编译器来源及完整 pass 名。该工具不执行 GPU kernel，不修改 IR 或编译器。先据此决定可安全限制的展开层级；不能仅把范围从全局缩小到 PV 就声称避开同一问题。

任务 `1b4ef551-a626-4d95-a15e-2079ce306c47`，attempt `9d6494c6-3b4d-4b53-9373-b6968b830374`。任务成功仅表示脚本产物生成，实际运行结果另记。

## 续接：真实 IR 证据与假设修正

首个外层 PassInstrument 工具静态审阅不采用：JIT 内层会另建自己的 PassContext，外层监听器不保证收到 lowering 回调。直接 lowering 工具任务 `5d8dea8c-7a11-499d-8841-e05e0fac6d9e` 超时，无产物。调整为内置 DumpIR 后，DSH任务 `97034642-d62a-4177-b8e3-1925fe37e5ef` 成功，产物 SHA `58f24ba1d4b4e7f6573688e6d3e89642b4fb04aee14c33ba24657fe6dbe38f3f`，SSH实际成功导出 ef30/7f5 各84份 pass IR。压缩包 SHA `0ac3b860ef0412ffa01427a97755c05ea94a9842481a1a4124b659848171e5410`，完整本地证据 `ir-builtin-evidence.zip` 及 `ir-builtin-extracted/`。

PV在展开前的实际IR（`baseline/041_tl.LoopUnswitching.py`）结构：ki2内有A_local缓冲区对象，j16/local_id4内也有B_local缓冲区对象，最后独立j16调用`tvm_mfma`。这些打印出的 `T.Buffer` 是旧tir实际访问的缓冲区对象表达，不等于证明OJ里已出现重复DeclBuffer；需要结合tirx的定义语义，不能仅用文本名字判SSA错误。

限制深度候选 `7d21c0093a28ebdb3bc89ef56b10791819e32a64f403652c1b0ee2490d44bd20`（DSH任务 `e3cc5d76-cf82-4e56-a4ea-148fca2b6e87`）仅在7f5 decorator加 `auto_max_depth=1,explicit_unroll=True`。实际生成代码保留ki2，内部j/local_id静态展开，MMA静态调用16个；host字节保持与基线相同。六点原参考、seed0/42全部通过，7轮×100次，speedup0.994/1.014/2.157/2.349/2.541/2.673×；完整报告已回收。首次SSH回传中断不影响已回收的JSON完整性，原失败日志保留。

**修正：这个depth1候选仍展开了含B_local访问对象的内层加载循环，不能认为只保留ki就排除PR2290风险，不作为已修复提交。** 保留为性能/层级对照，不覆盖生产MLA.py。

进一步源码修复实验：把PV概率/值显式装入fragment，在局部展开attr外完成copy，attr内仅fragment/fragment GEMM。已读实际MACA gemm_rrr实现，不创建A_local/B_local和ldmatrix，只迭代ki并调用MMA。配合depth0保留ki、仅展开MMA索引循环，避免把加载缓冲区对象复制机制带入该范围。代价可能是额外寄存器和copy，必须实测。DSH任务 `0bf949c3-9ca2-43c1-ab2d-f37df6dc8081`，attempt `a3e0c5ea-2782-4dd8-9d0a-283752c37fe3`。待产物审阅/编译，不宣称OJ已通过。

## 本轮最终验证状态

- depth1：31/31原始case、双seed全部通过，几何平均2.52922×。六点加速仍明确，但其内层加载展开可能仍触发OJ对象定义问题，未提交OJ。
- PV fragment/fragment候选SHA `d358c19a76de019615882dcdc6f2e55b28de242c57353a32e3947b90110d0693`：生产差分符合设计，host helpers AST不变；SSH六点、seed0/42全部通过。7轮×100次，speedup0.96577/0.98139/0.93755/0.93017/0.93138/0.92901×，几何平均0.94566×。**性能否决，不跑完整31点，不替换生产，不请求用户提交这个慢版。**
- RR真实展开前IR：attr内为 `for ki,j in T.grid(2,16): T.tvm_mfma(...)`，直接使用V_pv/P_pv的data和偏移，没有原来的A_local/B_local分配及加载循环。加载已位于attr外。IR导出与kernel source均已回收，压缩包SHA `8f19a373cae93c63ba815469cc00b127f533b5c2f570ee16e51c685516fd867c`。源码结构避开了已识别的复制区域，但旧tir观察不能证明OJtirx全部定义行为。
- 所有任务终态、错误和GPU作业均保留；无活动DSH/GPU任务。生产 `D:/workre/tzb/MLA.py` 仍为ef30，本轮未做新的回退或覆盖。

**尚未完成：优化版在真实OJ编译通过，以及突破97分。**目前最强归因是PR2290相同缺陷路径，而非已知具体OJ断言。需要取得OJ TileLang准确版本/评测镜像或在同版环境中验证Buffer定义，才能把源码修复做成可证结论。已向用户询问版本/镜像入口，不再重复索要平台已截断的堆栈。继续路线优先：同版复现→检查展开副本Buffer对象身份→设计保留加载索引静态化且定义唯一的源码；不继续把纯SSH通过当成OJ编译兼容证据。

## 用户补充 OJ 版本后的定向核查

用户确认OJ：TileLang0.1.10，镜像描述 `marco torch 2.8.0+metax`（不是已确认可拉取的完整registry地址）。上游v0.1.10 tag commit `69bc43e2cfbc38976b0c5e1c4b2299b66e8970ee`，已只读下载三份对应源码：

- `v0110-src_transform_unroll_loop.cc`：tirx；246行Substitute后直接push_back；无UnrolledBodyDefFreshener；仍有ConvertSSA。
- `v0110-src_transform_flatten_buffer.cc`：174行保留并更新DeclBuffer，与旧tir剥离DeclBuffer不同。
- `v0110-tilelang_engine_phase.py`：257行UnrollLoop、279行SplitHostDevice，与OJ280行调用非常接近。

这是具体版本源码支持，不再只是当前main与旧tir的泛化比较。但无法证明MetaX定制包与公开tag逐字相同；完整平台断言也仍缺失。

本机Docker只有数据库及无关runner镜像，未见marco/MetaX。SSH环境没有docker命令，也未发现已有0.1.10环境。容器查询日志 `mla-v0110-container-inventory-20261028-47c8de7f-0.log` 清楚返回docker不存在，不能把runner外层status成功误读为找到镜像。没有新装大型编译工具链。

源码结构路线：dim512且headtile16时改为16个独立PV32全fragment，GEMM warpcols由16降到1，原加载j索引天然常量，**不添加任何显式展开配置**。四个PV128历史失败不能直接排除此路线，因为PV128仍有warpcols4及动态寄存器索引。代价是重复A加载和更多GEMM片段，需实际编译/数值/性能检查。其他shape保留正常分支。DSH任务 `4e77cc52-6690-4055-9a42-840741f51e6e`，attempt `216a8cc2-6e50-4f15-828e-9404e90009e5`。

PV32实际产物SHA `bcede140496b2fe5723825b64a3d45dbef2bbf34c5f2c658a48258379ad49600`，48392字节；静态审阅hosthelper AST和签名完全不变，新增32个PV32 GEMM（split/no-split各16），无新增UnrollLoop配置/pragma。原参考、seed0/42六点全部通过，7轮×100次，六点speedup0.971667/0.997188/0.983008/0.978509/0.981547/0.982108×，几何平均0.982308×。**未保留加速，性能否决，不跑完整31点，不发布，不要求用户提交OJ。**对应`pv32-fair.json` SHA `febd3c78ac54f823dd256761180ab2ca31d98215ce01f71f3e7b465001cf12c81`。该结构即使避免新增显式展开，也不能被描述为满足性能目标的修复。

版本信息已使缺陷归因从“泛化猜测”推进到“公开0.1.10源码与已修复缺陷匹配”；但MetaX包实际源码及终端断言尚缺，暂不声称根因已100%确认或优化编译已修复。仅知道torch2.8.0+metax不足以构造相同TileLang/TVM编译器；也不能直接用上游新版本覆盖服务器假装复现。

PV32的IR/生成代码导出也已完成，证据包SHA `8e2f29e4e550acd38ff7fbd21b71b9a8af3c262033adb1d22df2e00b2e09148cc`。生成代码42134字节、18个静态MMA调用文本；16个PV各自仍有ki2及local_id4动态加载循环。**原先认为PV32会使所有加载索引静态化过于乐观：j循环虽消除，local_id仍在。** 这解释了该架构没有完整实现原优化的结构目标；无硬件计数，不能仅由文本归因具体耗时。不得继续简单堆窄tile或重复本实验。

本轮全部DSH/GPU任务结束，生产MLA.py保持ef30，未发布无收益/未确认OJ兼容的候选。下一步应以0.1.10特定缺陷为约束设计源码：需要同时使PV内层加载索引静态且Buffer定义唯一；或获得MetaX0.1.10实际镜像后用其编译器验证。torch镜像描述不是可拉取镜像地址，不能据此声称已复现同环境。
