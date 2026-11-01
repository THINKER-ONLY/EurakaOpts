> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA OJ：正常版本与两次失败版本差分排查（2026-10-28）

## 结论先行

两次失败共同涉及**TileLang IR阶段的显式循环展开**。第一次全局展开、第二次PV局部展开；第二次只改范围，没有改变默认`explicit_unroll=True`，所以并没有排除共同机制。上一轮把“去掉全局配置”作为兼容性修复的依据不够充分。

目前确认的是编译器在`SplitHostDevice`失败，尚无数值错误证据。最终断言被平台截断，不能声称具体根因已确定。SSH的tir实现与OJ栈中的tirx不同，只能证明实现路径有差异，不足以识别精确版本或断言。

## 与正常源码的完整AST对照

| 版本 | 哈希前缀 | OJ状态 | 相对ef30的实质改动 |
|---|---|---|---|
| 最早96.67 |db89ab8fe6|通过96.67|历史基线|
| merge64 |4b32b1ccfb|通过96.67|merge线程128→64（相对db89）|
| conditional-rescale |ef30c4edde|用户确认96.67|多tile identity rescale跳过（相对4b32）|
| 全局展开 |5f6c03f884|SplitHostDevice编译失败|只增加JIT `auto_max_extent16,explicit_unrollTrue`|
| PV局部展开 |7f5f1fdfa0|同阶段编译失败|两处PV增加maxstep256 scope，explicit仍默认True|

对五份真实文件分别执行AST解析、module literal_eval检查、公开函数签名及host函数AST比较。两份失败文件的module globals、公开签名、host函数AST与ef30全部相同。全局版恢复原decorator后整个模块AST等于ef30；局部版解开两处PV attr scope后整个模块AST等于ef30。未发现隐藏的split/merge、shape、mask、容差或数学公式修改。

完整机器对照为 `.heteromcp/mla-unroll-differential-20261028/variant-comparison.json`，两份逐行diff为同目录`oj-failed-global.diff`、`oj-failed-local.diff`。比较的是本机完整源码；未独立取回OJ转换后的缓存源码。

## 错误可能性排序

1. **优先：显式展开后的IR与OJ编译器后续pass不兼容。** 两版改变范围仍在相同阶段失败，共同机制保留。进一步用显式展开/后端提示做单变量对照。
2. **可能但未证明：展开与use-def/SSA、重复局部绑定等交互。** 本地UnrollLoop实现明确会在IR复制后调用ConvertSSA，因此不能直接断言“忘记SSA”或“必然重复定义”；OJ同版源码和终端断言缺失，需证据。
3. **仍需排除：OJ上传源码/转换链与本地文件不一致。** 栈中的行号和cache名说明两次产生不同源码，但没有还原其完整AST或SHA。后续需提交指定诊断文件，避免复制混用。
4. **当前证据不支持：数值精度、mask、分片公式、资源launch超限。** 错误发生在编译host/device拆分，尚未kernel执行；这些逻辑也与正常版本相同。不能靠调整容差解决编译错误。

## 最小差分诊断候选

以7f5f...为输入，由DSH仅增加两处内层scope：

```python
with T.attr(0, "pragma_auto_unroll_max_step", 256):
    with T.attr(0, "pragma_unroll_explicit", 0):
        T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
```

本地实际实现中，`explicit=False`把合格循环标记为kUnrolled而不复制IR循环体；MACA codegen再生成`#pragma unroll`供后端编译器处理。不是完全关闭展开，而是改变展开发生的阶段，符合本轮区分假设的目的。不得修改compiler源码、驱动、测试、容差或加环境检测/异常fallback掩盖错误。

- DSH task `fa977b4b-fe28-4210-8a81-2e26d3374996`，attempt `5b87cddb-0738-4940-bc5b-8fd510c740b1`。
- Artifact SHA-256 `84650654090517b75f577174ae68dfe5b700d0e1ae0832b93404c40788d898b8`。
- 唯一差分：两处PV的`pragma_unroll_explicit=0`scope，未混入其他机制。
- 原始产物`.heteromcp/artifacts/fa977b4b-fe28-4210-8a81-2e26d3374996/5b87cddb-0738-4940-bc5b-8fd510c740b1/workspace/MLA.txt`。

本候选用于诊断，不把SSH通过或代码检查当成OJ已修复。最终GPU及生成代码结果见同目录证据及后续记录。生产`MLA.py`不会在本轮未经实际OJ接受就继续换成新的“已修复版本”；交付独立候选便于单变量测试，保留ef30正常备份。

最小差分候选SSH六点seed0/42全部通过；加速比分别0.95807、1.01153、0.97235、0.95668、0.99174、0.98578×，几何平均0.97917×。它没有保留原2–3倍收益，因此**不接受为性能候选，不跑性能发布用31-case，不替换生产文件**。保留独立诊断源码供需要时测试OJ是否编译通过，而非要求用户再次提交一个已确认更快的版本。

生成代码导出确认：context2048/8192/65536的host source与ef30字节相同。device文本从13242/13525/13528B变为13318/13609/13612B，循环声明数保持23/24/24，MMA静态调用文本数保持3。PV ki2、j16、local_id4和MMA输出j16循环前出现`#pragma unroll`，循环体没有像显式展开版本一样复制为约50KiB文本。这证明差分实验按设计改变了展开发生方式；没有最终ISA或硬件计数，不能断言后端完全忽略提示。

为恢复可提交状态，本轮将当前失败的7f5f...独立备份，再把ef30已知OJ96.67的原字节恢复到`MLA.py`，保留全部失败证据。此举撤回当前不兼容优化，不宣称解决新优化的OJ编译问题。实际恢复记录见本轮`rollback.json`。

最新状态覆盖早先skill中的7f5f“尚未提交”表述：7f5f也已实际OJ失败；当前工作基线应以恢复后的ef30为准。5f6/7f5不得继续被标记为OJ可用。独立诊断文件名`MLA.diagnostic-backend-unroll-hint-20261028-8465065409.py`，不是新的生产或性能版本，亦未实际OJ验证。

## OJ验证如何解释

- 指定后端提示候选仍失败：说明IR显式复制不是充分解释；暂停展开路线，优先取得完整终端断言/同版编译器/转换后源码。
- 指定后端提示候选通过：支持显式IR展开触发不兼容的判断，再检查OJ耗时和分数；不能只因通过就声称完成性能目标。
- 正常ef30在当前同一OJ任务也失败：优先排查平台环境、任务参数或上传转换链变化，而不继续把原因锁定在展开。

旧失败7f5与5f6均保留，已知正常ef30文件为`D:/workre/tzb/MLA.score96p67-pre-unroll-20261028-ef30c4edde.py`。详细skill角色约束继续按原MLA skill，所有代码修改由HeteMCP→DSH生成；本轮不替换该正常备份。
