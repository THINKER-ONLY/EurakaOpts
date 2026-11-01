> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA：源码静态 PV 与实际二进制排查（2026-10-28）

目标是保留优化收益，同时避开 OJ TileLang 0.1.10 显式展开复制 Buffer 定义的疑似缺陷。**最新用户反馈：本候选已通过OJ编译与评测，六点20/35/39/64/114/211μs，总483μs，仍96.67分。** 文中性能加速倍数限定为SSH TileLang 0.1.9 MACA环境，不能再当作OJ预期收益。

## 固定版本与实现

- 基线：`D:/workre/tzb/MLA.py`，SHA256 `ef30c4edde76aab2e21d462e43acc59d3c18e3b08c00ab07f80d9aa353ec45a6`。
- 新候选：`D:/workre/HeteMCP/.heteromcp/mla-safe-pv-20261028/MLA.def723b6f9.dsh.py`，SHA256 `def723b6f97f9cb2d0aa377c762be7e082a9fee40a98b705d942a53c34db0790`。
- 单一机制：dim512、M16、block_n32 时，保留原 acc_o fragment，显式标注原 MACA FullCol 布局；在 kernel 分配作用域一次分配 A/B local，直接写出4个A加载、64个B加载、16个原生 `T.tvm_mfma` 调用。ki2保持serial，不引入编译器显式展开配置。
- QK/PE、softmax、FP32累积/LSE、条件rescale、mask、split/merge、host helpers与公共签名保持基线。其他形状保留原GEMM。生产源码仅导入tilelang及其language，模块赋值literal兼容。
- 此路线避免原方案通过编译器复制PV加载循环体来静态化索引；不同于此前已否决的RR fragment与16个PV32 GEMM。没有更改编译器、驱动、测试参考或容差，没有环境探测或异常fallback。

不能将源码设计直接等同于 OJ 已通过：MetaX 0.1.10实际包及终端断言仍未知。PR2290匹配仍是强证据，未变成同环境已证实根因。

## 真实 HeteMCP→DSH 留痕

所有实现、诊断代码均由 stdio MCP SDK→HeteMCP→Orchestrator→TaskExecutor→DSH 生成，复用 `heteromcp-postgres`，未直接调用模型。

| 任务 | Task / Attempt | 结果 |
|---|---|---|
| 首个生产实现 | `bf9b0586-36e2-4e6a-8e14-c2f357fba747` / `44aa8dd4-4975-4b7e-b388-2a92280636a4` | 600秒超时；终止后留下完整文件，仅作为实验产物固定SHA上机 |
| 同源码正式生成验收 | `cf1e3f80-dd19-41fc-be1f-03858661ca2f` / `13dc30dc-24ff-4e5d-99ed-6ef6d67d45a1` | succeeded；源SHA与实测候选逐字节相同，文件范围独立验收通过 |
| 首个通用诊断工具 | `056d9301-d2ab-47f5-9123-4e74a712915e` / `530c4b24-f0fb-4f45-9926-dd7914bbc0e2` | 600秒超时；124391字节、过度设计，未执行 |
| 精简mcbin诊断工具 | `202412f8-0281-4770-8e9e-46c7e9d323b4` / `f84a2758-6dbd-4618-847c-904fab3d18d7` | succeeded；SHA `1ef134ce42b3781f8a9a9e7828cd5e7b880f3b66d12e74d137ee8d3abbd3a304`；实际运行成功 |

调度首次Docker EPERM后通过实际工具提权重试。首次TaskSpec context将对象误填为字符串，修正schema后才创建任务；没有用fallback入口绕开真实链路。日志、失败产物、任务快照均保留 `.heteromcp/`。

## 核心六点

基线/候选同轮paired A/B；原参考、rtol2e-4、atol1e-4、mismatch_ratio0.01，seed0/42，双方预分配输出，预热200，7轮×100次。

| context | 基线μs | 候选μs | 中位数加速 | 配对最小加速 |
|---:|---:|---:|---:|---:|
|2048|25.334|25.797|0.9820×|0.9665×|
|4096|42.035|41.085|1.0231×|0.9920×|
|8192|102.559|72.502|1.4146×|1.4053×|
|16384|179.441|125.696|1.4276×|1.4157×|
|32768|332.782|229.512|1.4500×|1.4410×|
|65536|637.896|436.070|1.4628×|1.4585×|

六点双seed均通过。短两点基本持平，中长四点七轮均快于基线。SSH绝对耗时不能直接投射为OJ耗时或分数。

证据：`def723b6f9-core.json` SHA `57a775ca68e15d405b5a068e357da47f1b96dc604f4c9b00610a0990f953a0d4`；SSH作业`mla-safe-pv-def723b6f9-core-20261028-d87f5e8d.json`。

## 完整31点

原31点、双seed、原参考/原容差；5轮×20次、预热20。31/31全部通过且finite，几何平均1.504439×，最小中位数加速0.998148×；最大batch32/context65536为1.611575×。

证据：`def723b6f9-full31.json` SHA `ce454f0f928602313f93cbd0ce2e16954d9e06f23b779e962f2133b631d77679`；SSH作业`mla-safe-pv-def723b6f9-full31-20261028-60102275.json`。

## 实际二进制与资源排查

只读取得服务器真实 `maca_module.cc`、`mxcc.py`、runtime module源码，发现 `InspectSource("asm")` 可退回返回CUDA源码，不能将其算作ISA。实际接口为device module `write_to_file(path,"mcbin")`；`"maca"`格式明确导出的是源码。

精简DSH工具对context2048/8192/65536，两版本共6次compile、源码/IR/host导出、mcbin导出全部成功，二进制magic均`7f454c4602010180`，确认为ELF64。三个host pair均逐字节相同。候选lowered IR中acc_o实际分配64个FP32/thread，PV加载下标静态，ki2保留。

| context | 基线ELF字节 | 候选ELF字节 |
|---:|---:|---:|
|2048|145576|141128|
|8192|147736|143712|
|65536|147752|143712|

这些文件大小不是吞吐或spill计数。GNU objdump不认识MetaX ISA，readelf报告部分重定位不支持；普通工具未直接解析寄存器/私有内存字段，诊断JSON按null记录。ELF `.note` 内有MetaX type0x30 MessagePack，含`macahca.kernels`、`.mtreg_count`、`.streg_count`、`.private_memory_size`、`.share_memory_size`等；进一步解析由DSH实现，结果另补。

证据包`profile-evidence.zip` SHA `60b19033bc757c3afd803ddb506de1ed4620802d3dad7b38e3f2de3680f99ac41`；报告SHA `cbf1fc29685fcc543380f38d6cc6057cf458304a56394bcf58e6ff017e30a96d`；作业`mla-safe-pv-binary-profile-20261028-39348e6c.json`。未采得有效硬件性能计数，不声称已证明spill/occupancy/吞吐归因。

## 额外覆盖与范围限制

原31点均为dim512、16头、context>=2048，不能当作所有API形状回归。另建独立supplemental JSON，原测试集未修改。长度1的无split双seed通过。长度31在旧基线验证阶段即因30.04% mismatch超出原容差中断，候选未运行；这是已发现的基线泛化限制，不能说是候选正确性失败或宣称额外8点全部通过。剩余case分别运行，失败均留痕，不放宽容差。

补充8点结果：101(ctx1)、105(ctx2049)、108(batch2/heads32/ctx4097)双seed通过；102(ctx31)、103(ctx33)、104(ctx1024)、106(heads17/ctx65)、107(dim256/ctx64)均在旧基线的首seed数值校验先失败，候选数值校验未执行。这些失败不影响原31点通过事实，也不支持“所有shape正确”的结论。作业`mla-safe-pv-supplemental-rest-20261028-4636a157.json`的外层status虽为succeeded，但5个相关exec实际exit1，逐step已检查。

## ELF资源元数据最终结果

DSH解析任务`13241c53-94e0-4055-b4ec-c9a93ee82af7` / `518647cb-6715-49a8-b44f-d3a9d5b4dc53` succeeded。脚本SHA `3b77750b9a138e089784885965197f06ff7cbc2dc56fe8e96964dcec1ada30cb`，无第三方依赖。6个ELF、12个kernel全部解码，无错误、MessagePack描述符无trailing bytes。原始完整metadata及ELF SHA保存在`elf-metadata.json`；二进制符号表function大小单独核对，脚本按`.kd_symbol`查FUNC导致summary的function_byte_size为null，未据此误报，下面按`.name`匹配真实FUNC符号。

| attention context | .mtreg_count 基线→候选 | .streg_count | .private_memory_size | FUNC代码字节 |
|---:|---:|---:|---:|---:|
|2048|124→122|28→28|0→0|16896→15872|
|8192|256→244|38→28|252→0|19456→16640|
|65536|256→244|38→28|252→0|19456→16640|

三个shape的merge均为mtreg78、streg22、private0、代码3840字节。`.share_memory_size`均为0，这是静态共享字段，**不代表实际launch不使用共享内存**；65536的lowered IR明确`dyn_shared_memory_buf:57344`，host pair保持相同。

现在可证实：源码静态化后，中长attention的编译资源元数据中mtreg/streg和private memory下降，代码体积缩小，paired运行时间稳定改善。无法仅凭private252→0给出动态spill次数、占用率或指令吞吐。硬件性能计数仍未采得。

## 交付状态

候选通过正式DSH文件验收、原31点双seed与二进制核查。用户随后报告OJ六点20/35/39/64/114/211μs，总483μs，仍96.67分，证明该源码路线已能通过真实OJ编译/评测；未独立拉取提交源码SHA，当前本地生产SHA仍def723b6f9。相对旧490μs仅少7μs（1.43%），没有重复OJ计时，不能区分微小收益与波动，**不得再把SSH1.504×作为OJ可迁移收益**。PR2290仍为旧显式展开失败的强归因，成功候选不等于证明了旧崩溃的精确断言。

发布前后SHA在`.heteromcp/mla-safe-pv-20261028/publication.json`。最新OJ原始反馈在`.heteromcp/mla-fused-qk-20261028/oj-observation.json`。97分尚未实现；后续融合QK/PE、merge32及split扫描结果见 `mla-oj483-investigation-20261028.md`，本报告的def723是上一轮已OJ验证版本。
