# MLA 决赛 OJ 实验记录（2026-11-11）

基线与所有候选均为纯 TileLang、无 torch 计算、无跨调用结果复用。

## 基线

- 线上最佳：submission 163109，display **96.83**，rank 11（35 队）。
- 源码已从 OJ 拉回并逐字节保存：`/root/lzy/workbench/v028_oj163109_96p83_recovered/submission.py`
- 复核提交 167108 复现 96.83，逐 case kernel 时间完全一致
  （0.021 / 0.020 / 0.026 / 0.040 / 0.065 / 0.114 / 0.211 ms）。

## 关键结论

1. **OJ 判题环境是确定性的**：同一源码两次提交的逐 case 时间一致（1µs 量化）。
2. **本地 C500 是 50% 切片，不能迁移并行度结论**：本地 104 SM 上 split=104
   快 27%，但 OJ 上慢 35%。OJ 的有效并行度约为 64 个 CTA。
3. split=104 在 OJ 上的退化与 merge 无关：merge tile 32 仍是 96.33。
   因此是主 kernel 的 CTA 数超过设备并行度，而不是 merge 的寄存器问题。
4. merge tile 64→128 无变化（96.83），merge 不是瓶颈；固定开销主要是
   两次 kernel launch（本机 noop launch ≈ 6.7µs/次）。
5. `num_stages=1`（仅长上下文）在 OJ 上每个 case 慢 1–4%，否决。

## 实验表

| 提交 | 变体 | 结果 | 结论 |
| --- | --- | --- | --- |
| 167108 | 96.83 源码原样（对照） | 96.83 | 环境可复现 |
| 167125 | v029 stauto（多 tile 开 num_stages=1） | 96.83（逐 case 略慢） | 否决 |
| 167131 | v031 adaptive(split=104)+stages | 96.33 | 否决 |
| 167133 | v034 split=104（无 stages） | 96.33 | split 数是原因 |
| 167138 | v039 merge tile=128 | 96.83 | merge 非瓶颈 |
| 167139 | v044 split=104 + merge tile32 | 96.33 | merge 非原因 |
| 167140 | v046 基线 + merge tile32 | 96.83（8192 0.039） | 中性 |
| 167144 | v070 fast-path run_kernel | 96.83（mean 97.071；0.020/0.020/0.026/0.039/0.064/0.114/0.210） | 当前 MLA 最佳单体 |
| 167152 | v071 fast + pass configs | 96.83（16384 0.065、65536 0.213） | 略退，否决 |
| 167153 | v072 fast + 去条件缩放 | 96.83 | 中性 |

## 本地参考（仅相对比较，绝对性能不可用）

- 组件拆分（ctx 2048，split 64）：主 kernel 14.0µs、merge 4.7µs。
- 去掉 PV：-3.3µs；去掉 QK：-1.2µs。
- 长上下文（65536）：PV ≈ 250µs、QK ≈ 224µs（总 406µs）。
- `run_kernel` CPU 入队 12.9µs，直接调用 kernel 9.9µs；fast-path 降到 10.8µs。

## 工具链

- OJ 客户端：`/root/lzy/ojtool/oj.py`（查询、拉源码、SPJ 逐 case 报告、提交）。
- 自动提交+判题+记录：`/root/lzy/ojtool/loop.py`，结果追加到 `oj_ledger.jsonl`。
- 本地基准：`mla_decode/benchmark_c500.py`、`/root/lzy/ojtool/mla_probe.py`。
- 性能分析：`mcProfiler`（/opt/mcProfiler-ubuntu18.04）、torch.profiler、
  `mx-smi`、`mcTracer`、`mx-report`、`mx-diagease`。

## NSA 摸底（进行中）

- 提交 `167146`（contest 7，v025_cached_s1_block_start），待判。
- 本地 20 个代理 case：相对 v001 平均 2.85x，估计分数 ~70（仅参考，本地不可迁移）。
- 决策门槛：NSA 需进入前 14 名（≈89 分）才优于 MLA 当前第 11 名（20 技术分 → 12 决赛分）。

## 计分公式（已从 OJ 数据反推）

subtask 只统计 **6 个 case**（2048/4096/8192/16384/32768/65536），
display score = 6 个 **整数** display 分的均值（当前 95+96+97+97+98+98 = 581/6 = 96.8333）。
另外两个 case（重复的 2048）不计入。

| case | 当前 kernel | 当前 ratio | 当前 display | 升 1 分所需 |
| --- | ---: | ---: | ---: | --- |
| 2048 | 0.020 | 95.44 | 95 | ≤0.0175 ms（-12.5%） |
| 4096 | 0.026 | 96.81 | 96 | ≤0.0244 ms（-6.2%） |
| 8192 | 0.039 | 97.51 | 97 | ≤0.0311 ms（-20%） |
| 16384 | 0.064 | 97.91 | 97 | ≤0.0612 ms（-4.4%） |
| 32768 | 0.114 | 98.12 | 98 | 不可达 |
| 65536 | 0.210 | 98.26 | 98 | 不可达 |

最现实的目标：16384、4096、2048 各升 1 分 → 均值 97.33（约第 3 名）。

## 失败实验：2 CTA/SM（bn=16，单 warp）

动机：shared 从 70KB 压到 18.5KB，目标 2 CTA/SM、128 splits。

实现：threads=64；Q/Q_pe 放 fragment；直接写 partial（去掉 O_shared）；
自定义 acc_o 一维 warp 布局；PV 显式 MFMA 改为单 warp × 32 个 N tile；
bn=16、split target 128。单 warp 最小用例与 torch 参考一致（相对误差 3e-4）。

结果（本地 C500）：
- 2048 / split64：34.3µs（2-warp 基线 23.9µs）
- 16384 / split64：294µs（2-warp 基线 117µs，慢 2.5×）
- 16384 / split128：312µs

结论：单 warp 承担完整 QK/PV 串行链，per-CTA 变慢 2.5×；即使 2 CTA/SM
完全生效也只能追回 2×，净亏。共享内存受限下的 occupancy 路线在此 kernel
形状下不成立。保留作负结果记录：v100_bn16.py（展开 macro）、v101_bn16_dyn.py。

## 已否定：80 splits 假说（167175）

104 splits 在 OJ 上慢 1.35×，104/1.35 ≈ 77，暗示评测机有效并行度可能是 80 个
CTA 而非 64。若成立，64 splits 只用了 80% 的卡，80 splits 可免费拿约 20%。
v102_split80.py 已提交（2048 仍 64；4096+ 用 80），等待结果。

OJ 结果 167175：display **95.67**，逐 case 全面退化
（4096 0.039、8192 0.068、16384 0.109、32768 0.190、65536 0.362）。
结论：评测机有效并行度就是 **64 CTA**；64 splits 恰好一波，任何 >64 的 split
都会进入第二波。split 维度到此封死。

## 软件栅栏融合 merge（v112，已提交 167187）

- 用 atomic_add(return_prev) + while 自旋替代 sync_grid；64 CTA 全部驻留（已由
  split 实验确认评测机 64 SM），栅栏本身仅 0.7µs。
- 心内 merge：每 block 负责 (head, 128 列 tile)；LSE 用 fragment 归约并行算权重
  （串行版本实测慢 11µs，并行后 merge 共 3.7µs）。
- 本地：2048 21.5µs（基线 23.9）、4096 37.1、16384 115.2、65536 404.6；
  正确性 max_abs 与基线一致（1.2e-4 / 6e-5）。
- 预期 OJ：省掉第二次 launch，2048/4096/16384 各跨整数分阈值。

## 根因结论：纯访存延迟受限

逐组件去掉证明计算不是瓶颈（本地 ctx2048，基线 23.9µs）：
- 去掉 PV：20.6（-3.3）
- 去掉 QK：22.7（-1.2）
- 去掉 exp2：24.1（0）
- 去掉缩放：24.4（0）
- 去掉两次归约：23.7（0）
- 去掉归一化除法：23.9（0）

生成代码确认 Q/Q_pe/KV/K_pe 四次拷贝之间无同步、完全重叠。设备拷贝带宽
~939 GB/s，而 kernel 长上下文读带宽仅 ~305 GB/s（约 1/3 峰值）；短上下文
更低（延迟受限）。提升只能靠更高的访存并行度（双缓冲/occupancy），但：

- KV 双缓冲需 64KB，加 K_pe 4KB + S 1KB（即使 Q/O/Q_pe 全放寄存器）
  仍超 64KB 上限 5KB；
- 降 block_n 到 16 需单 warp，per-CTA 慢 2.5×（已实测否决）。

因此在该 kernel 结构下 96.83 已接近硬件上限。

## 已否定：wide-32（block_h=32, 256 线程）

目的：用 4 warps 提升访存 MLP（计算浪费一半，假设访存受限可隐藏）。
结果（本地 2048）：31.4µs vs 基线 23.9（慢 31%）。T.gemm PV 变慢 + 2× 计算
无法被隐藏。否决。

## 精确 A/B（同进程交替，CUDA event）

| ctx | 基线（main+merge） | v114（栅栏融合+预取） | 收益 |
| --- | ---: | ---: | ---: |
| 16384 | 116.84 µs | 114.41 µs | 2.08% |
| 4096 | 34.0 µs | 32.9 µs | 3.2% |
| 2048 | 18.7 µs | 17.9 µs | 4.3% |

不足以跨整数分阈值（16384 需 -4.4%、4096 需 -6.2%、2048 需 -12.5%）。

## 硬件上限（最终确认）

torch 设备属性（C500）：
- shared_memory_per_block = 65536，**optin 也是 65536**（无更大动态共享内存）
- shared_memory_per_multiprocessor = 65536 → 每个 SM 只能容纳 1 个 64KB 的 CTA
- regs_per_multiprocessor = 131072（寄存器充足，瓶颈只在 shared）
- warp_size = 64

结论：KV 双缓冲（64KB）加上 K_pe/S 必然超限，2 CTA/SM 需 ≤32KB 也不可能。
在该 kernel 结构下 96.83 已是硬件可达上限；v114（栅栏融合+预取）为技术最优，
OJ display 仍 96.83，内部 ratio 97.0745（略高于 97.0710）。

## Warp specialization 尝试（未成功，记录负结果）

动机：纯带宽基准确认 2 warps 拷贝 831 GB/s、4 warps 1895 GB/s。若让多出的
warp group 只搬数据、group0 只算（2-warp GEMM），理论上每 tile 的 load 从
2.8µs 降到 1.2µs，可省 ~20%。

已验证可用的部分：
- `T.ws(i)` + 256 线程结构可编译运行；单组（只 ws(0)）多 tile 结果正确。
- 无 WS 的“两 buffer 列切分”分解（QK 拆两个 K=256、PV 拆两个 N=256）正确。

失败点：
- 列切分 T.copy 在 loop_range>1 时生成错误地址（group1 源地址出现 -1792 常量），
  1 tile 正确、多 tile 错误。
- 两 group 交替搬运 + 循环（任意同步组合、Pipelined/serial/stage1）在 k>0 时结果错误；
  同样的模式在独立 probe 中正确，说明是 TileLang 0.1.9 race 分支的 ws lowering
  与该 kernel 组合下的缺陷。
- 拷贝放在 ws 外（256 线程）时布局推断按 256 线程生成，与 ws(0) 内 2-warp GEMM
  的 expected layout 不一致，1 tile 即错误。

结论：当前工具链版本下无法可靠实现 4-warp 搬运 + 2-warp 计算。相关文件：
v140-v150（workbench）。潜在收益约 20%，但被编译器缺陷阻断。

### WS 补充实验（v152-v154，全部失败于 loop_range>1）

- v152：256 线程 + ws 分组内逐元素写共享列半区；1 tile 正确，2 tiles 错误。
- v153：同 v152 但 128 线程、无 ws；**多 tile 正确**（但无速度收益，拷贝仅 2 warps）。
- v154：256 线程 + 逐元素拷贝放在 ws 外、仅 compute 在 ws(0)；1 tile 正确，多 tile 错误。
- 用 TileLang 0.1.10（OJ 同版本）复跑 v141 仍错误，排除版本差异。

结论：TileLang 0.1.9/0.1.10 的 ws lowering 在本 kernel 形状下，只要
“循环 + 跨 group 共享 buffer + 多迭代”同时出现就产生确定性错误；所有
可想到的同步/拷贝写法都无法绕过。4-warp 搬运（理论 ~20% 收益）在当前
工具链下不可实现。

### WS 根因（最终定位）

把循环完全静态展开（k 为 Python 常量、两份独立 body）后**仍然错误**，
排除“动态 k / 循环”因素。检查生成代码发现：

- 正确版本（128 线程）：output_partial store 索引为 `i*65536 + (tx>>6)*32768 + ...`（128 线程布局）。
- WS 版本（256 线程）：store 索引为 `(tx&15)*32768 + (tx>>6)*256 + ...`（按 256 线程/复制布局推断）。

即：compute 在 ws(0) 的 128 线程内执行，但 fragment 布局按整个 256 线程 block
推断（replicated），导致一半 fragment 元素从未被计算/写出。单 tile 时编译器
恰好特化正确，多 tile 时不成立。

显式 annotate_layout 会与 gemm 自带的 C 布局注解冲突（ValueError），无法从源码层覆盖。
该缺陷在 TileLang 0.1.9 与 0.1.10 均复现，而 OJ 使用其自带 0.1.10 编译，
因此“本地修编译器”无法嵌入提交。

结论：4-warp 搬运这条路在当前工具链下彻底封死，需要上游修复 ws 的 per-group
layout inference。

### 手写 MMA 内核（v170-v174）：可行但性能不达标

为了绕开 T.gemm 按 block 线程数划分 warp（256 线程必然 4 warp）以及 T.ws
布局缺陷，完全手写了整条计算链：
- QK/QK_pe：T.tvm_mfma，2 warp ×16 列，K=576 分 36 步；
- softmax：共享内存 8 路部分归约（PMAX/PS）+ 128 线程共享原子自旋屏障；
- PV：16 个 n 块静态展开，acc_o[64] 全常量索引（确认在寄存器）；
- epilogue：手写列映射存储；KV/Q 拷贝用全部 256 线程（T.copy）。

结果：**数值全部正确**（max_abs 1.2e-4），证明 4-warp 搬运 + 2-warp 计算
的结构端到端可行。但性能：2048 41.4µs、4096 57.1µs、16384 139.9µs，
比 T.gemm 版本（23.9 / 35.2 / 117）慢 1.2–1.8×。去掉屏障、PV、甚至把线程
数降到 128 都不改变——**手写 TIR 的 MMA 调度质量不及 TileLang 的 gemm 代码生成**。

结论：在 OJ 自带工具链下，4-warp 搬运的收益（约 20%）被人手代码的损失
（约 70%）吞掉。该方向需要上游修复 ws 的 per-group layout inference 才有意义。

## 手写内核 OJ 实测（决定性负结果）

提交 167353（v191，手写 no-max + 256 线程搬运）：
- 2048 0.024、4096 0.037、8192 0.063、16384 0.114、32768 0.215、65536 0.416 ms
- display **95.5**（基线 96.83）

本地（50% 切片）vs OJ（满卡）的缩放差异是关键：
- T.gemm 基线：112µs → 64µs（**0.57×**，满卡带宽/并行收益明显）
- 手写内核：106µs → 114µs（**1.07×**，完全不吃满卡收益）

说明手写 TIR 生成的访存/调度在真实 C500 上效率远低于 TileLang 的 gemm
codegen（本地的相对比较被切片环境误导）。**手写本地快也无意义，OJ 为准。**

结论：手写路线彻底放弃。96.83 仍是上限。

## NSA 决赛刷分（2026-11-11 起）

基线：v026 fastcall，84.71（109 case 平均）。榜首 92.07，前 20 有分（≥87.6），
前 14 才超过 MLA 当前 12 决赛分（≥~89）。

逐步提交的候选（每个都是一次完整 109-case 回归）：
1. nsa_t64（threads 固定 64）
2. nsa_fullrow（QK/PV 全部 FullRow）
3. nsa_vpref（V 的 load 提前到 K 之后，与 QK gemm 重叠；新增独立 V buffer）
4. nsa_fullcol（全部 FullCol）
5. nsa_gather48（S=4/8 也走 gather 单次大 gemm 路径）
6. uss_true（强制 shared-score 路径）

本地过滤结果：t128/t256 编译失败；uss_false 布局冲突；其余通过。
后续根据 OJ 逐 case 回归组合最优项。

### NSA 搜参结果（OJ）

| 版本 | display | 结论 |
| --- | ---: | --- |
| 基线 v026 | 84.71 | 最优基线 |
| t64 | 84.00 | 否决 |
| fullrow | 84.64 | 中性 |
| vpref（V 预取，有效性块内） | 83.93 | 否决（随机块含哨兵块，V 未初始化→NaN） |
| fullcol | 65.86 | 严重负优化 |
| gather48（S=4/8 gather） | 83.14 | 否决（随机块 gather 大量无效） |
| pol128（D=128 用 FullRow） | 84.36 | 略负 |
| vpref32（D=32 预取） | 66.21 | 同上 NaN 问题 |
| **nomax1（S=1 去掉 max 归约）** | **85.00** | **正向，采纳** |

关键学习：官方 NSA 测试用 `randperm(t//BS)[:S]` 随机选块，大量哨兵块；
任何“预取/未初始化共享”方案必须无条件加载或清零，否则 NaN 全崩。
S=1 去掉在线 max 归约（分数有界）是唯一有效优化（+0.29）。
