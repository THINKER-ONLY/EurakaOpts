# MLA 决赛下一步攻坚计划（2026-11-11 夜）

现状：OJ 97.17（583/6，第 7）。榜首 97.83（587）。升一档需要
97.33(584) → 97.5(585) → 97.67(586) → 97.83(587)。
判题机逐 case：0.018/0.023/0.035/0.058/0.103/0.191 ms。

## 判题机上的真实分解（由 OJ 逐 case 反推，最可靠）

- 固定开销 ≈ 12.5 µs（launch + prologue + 栅栏 + merge）
- 每 tile 边际 ≈ 5.5-5.75 µs（36 KB KV）
- 其中：拷贝 ≈ 2 µs（判题机带宽约为本地 2 倍），计算 ≈ 3.5 µs
  ⇒ **判题机是「计算 + 串行化」受限，不是纯带宽受限**。
  这也解释了为什么本地 16% 的拷贝优化（rotstage / maca_async）在判题机上只有 1.7%。

## 已验证不可行

- `T.maca_async_copy`：本地数值正确且快 20%，判题机 5 个 case 全 WrongAnswer
  （单/双 barrier、显式 linear layout 都试过）。判题镜像 0.1.10 不可用。
- `T.gemm` 作为 PV（替代 pv_mfma 宏）：本地慢 1.7 倍（8192: 100.6 vs 62）。
- 256 线程手写 MMA（v173/v190/w1/w2）：计算比编译器慢 2 倍以上。
- split 16/32/48/64/80/104/128：64 最优。
- 减少 partial / merge / barrier 的各种写法：±1µs 以内。

## 下一步（按期望值排序，均需在 OJ 验证）

1. **256 线程 + `T.ws(0)`（仅 compute 在 ws 内）+ 手写 pv_mfma**
   - 关键事实（已由 probe 验证）：`T.ws(0)` 会把 layout inference 的 thread_bounds
     收窄到 128，T.gemm 在 ws 内是合法的 1x2；ws 内的 `T.sync_threads()` 会被降低成
     `__sync_thread_partial<id,128>`——**必须避开与 pv_mfma 宏内部同步的 barrier id 冲突**
     （历史失败点）。做法：把 pv_mfma 开头的 `T.sync_threads()` 提到 ws 外做全块同步，
     宏内只用 partial barrier；或者把 S_shared 的写-读顺序改成 ws 外的全块同步。
   - 收益来源：判题机上拷贝只占 1/3，4 wave 拷贝收益有限；但 256 线程让 CTA 可寻址
     寄存器翻倍（128 线程只能用到 32K/131K），可缓解「拷贝与计算抢 256 寄存器」的
     相互干扰（本地实测该干扰 ~41µs @16384）。
2. **消掉 `T.reduce_sum` 及其 2 个 alias barrier**：它的 scratch 落在 byte 51200，
   与 K_pe_shared/S_shared 基址重叠，每 tile 强制两次全块同步。最小改动：把
   logsum 改成从 S_shared 在 ws 外用 T.Parallel 求行和（fp16 求和，误差 ~2e-3，
   checker 容差 2e-3，需实测）；更进一步用 ones 列（KV_shared 扩到 [32,544]、
   Q_shared [16,544] 后 32 列补零）让 PV 顺带算出每行和。
3. **pv_a 向量化**：`T.copy(S_shared[row, c:c+4], pv_a)` 把 4 条标量 LDS 合成 1 条
   uint2（宏内 32 处 → 8 处），小但安全。
4. 固定开销 12.5µs 若含 ~5µs launch，则 2048/4096 无法靠 kernel 内优化跨档；
   8192 需要每 tile 再降 1µs（-18%），主要靠 1 和 2。

## 工具与命令

- 提交：`cd /root/lzy/ojtool && XPUOJ_TOKEN=$(cat /root/.xpuoj_agent_token) python loop.py <file> --contest-id 6 --order 1 --label xxx`
- 0.1.10（判题机同版本）本地验证：
  `PYTHONPATH=/opt/tilelang-metax-v0.1.10:/opt/tilelang-metax-v0.1.10/3rdparty/tvm/python /data/envs/xh202608/bin/python mla_probe.py --mod <file> --ctx 8192 --split 64`
- 判题机配额 10 次/30 分钟；OJ 保留历史最高分，实验提交无风险。


# 2026-11-12 01:05 — 更新（重要）

## 结果：97.17 → 97.33（Accepted 167603/167621，584/6）
逐 case（判题机）：2048 **0.017**(96) / 4096 0.024(97) / 8192 0.036(97)
/ 16384 0.059(98) / 32768 0.104(98) / 65536 0.195(98)。
最佳文件：`workbench/v299_wsalways.py`（= `EurakaOpts/mla_decode/v033_hybrid_97p33/submission.py`）。

## 关键发现 1：MACA 的 partial barrier 是空实现
`src/tl_templates/maca/common.h:320` 的 `tl::__sync_thread_partial<id,count>` 函数体为空
（注释称"maca 中 warp 内全同步"）。因此 **`T.ws()` 内部的 `T.sync_threads()` 全部是 no-op**。
只要 ws 区域跨 2 个以上 warp（128 线程 = 2 warp），区域内就无法同步 → 256 线程 + ws 的
写法规格上不可用。这是 v291/v296 首次 launch 偶发 NaN 的根因（thread 0 自旋等其它 CTA 时，
其余 127 线程无人等待就读取 partial）。
可行写法：ws 区域内部不同步，把需要的同步放到 ws **外面**的整块 `__syncthreads()`。

## 关键发现 2：256 线程只在「单 tile」的 split 上赚
判题机实测（全 256 线程版 167594）：2048 0.017（好 1µs），4096 0.026 / 8192 0.041 /
16384 0.068 / 32768 0.124 / 65536 0.234（全部更差，最多 +22%）。
分界点是 split 内 tile 数：1 tile 时拷贝收益 > 宽 block barrier 代价。
因此 v299 用 `fast_256 = (loop_range_split == 1)` 只对 ctx=2048 开 256 线程。

## 关键发现 3：判题沙箱的源码限制
`tilelang_runner.py` 用 triton_sandbox：**禁止 import 白名单之外的模块（含 contextlib）**，
**禁止 class 定义**。提交代码只能 import tilelang（+已有标准库）。

## 判题机成本模型（线性拟合 6 个 case）
`kernel_ms ≈ 11.2µs 固定 + 5.77µs/tile`。barrier 约 0.25µs 个（128 线程），
去掉 reduce_sum 的 2 个 barrier ≈ 0.5µs/tile。
下一档门槛：8192 需 ≤0.031（-14%），4096 需 ≤0.016（-33%），16384 需 ≤0.030（-49%）。
榜一 97.83 = 587，需要再 +3 分 ⇒ 必须让 8192 再降低 ~14% 且 4096 降低 ~33%。

## 下一步候选（按可行性）
1. 8192 的 -14%：把 `T.copy(KV)` 的 r244 rot 寄存器暂存（已在 167578 验证 +1~2µs）
   与 4 barrier/tile 的当前版本合并；再配 ones-column 去掉 reduce_sum（本地 -0.5µs/tile）。
2. 固定开销 11.2µs：确认 launch 占比（2048 单 tile = 0.017ms ⇒ 固定+1tile）。
   若 launch ≈5µs，则 8192 永远差 ~2µs；此时只能继续压每 tile。
3. 不再尝试：256 线程（>1 tile 更差）、T.maca_async_copy（判题 WA）、T.gemm PV（慢 1.7x）。


## 2026-11-12 02:51 — 最终状态：**97.33（Accepted 167650）**
逐 case：0.017(96) / 0.023(97) / 0.036(97) / 0.058(98) / 0.105(98) / 0.197(98)，
mean_ratio 97.400（历史最好）。最佳文件 `workbench/v300_final.py`
= `EurakaOpts/mla_decode/v034_final_97p33/submission.py`。
（v300 = v299 去掉 merge 里两处 `__threadfence`：4096 0.024→0.023、16384 0.059→0.058，
把两个贴着档位边界（97.05%、98.07%）的 case 拉开余量。0.25µs 级的 barrier 单价说明
barrier 不是瓶颈，fence 才是 merge 里真正的固定开销。）

## 唯一还有量级空间的方向（未实施，约 1 小时工作量）
现状瓶颈不是 instruction 数而是 **每 SM 只有 2 warp、且拷贝与计算在同一条 warp 上串行**
（本地：拷贝 21µs + 计算 46µs，实际 108µs/8 tile，1.6 倍干扰）。
可行重构：
- `block_n=16` + **64 线程/CTA**：QK 是 M=16,N=16，在 1 warp 下是合法 1×1 划分；
- Q [16,512] 放**寄存器**（64 线程 × 64 half = 32 regs），shared 只用
  `2×KV(8KB) + 2×K_pe(1KB) + S(0.5KB) + 4KB 暂存 ≈ 22.5KB` ⇒ 每 SM 2 个 CTA；
- 于是可以开 `T.Pipelined(num_stages=2)` 双缓冲，把 KV 拷贝藏进计算，消掉那 1.6 倍干扰；
- 需要重写 pv 宏（warp 内 N=16 形状）与 Q 的寄存器化 A-fragment 加载。
验收方式：本地 16384 若能从 111µs 降到 <80µs（>25%），再提交判题机；
否则放弃（本地 5% 级收益在判题机上历来只剩 1~2%，不要再浪费提交）。

## 2026-11-12 03:20 — 占用率方向已实测关闭
`v400_occ.py`（bn=16 + 64 线程 + Q 寄存器化 + 32 个 N tile 的单 warp acc_o）已实现，
数值正确但 **8192 本地 190µs vs v300 62.6µs（慢 3 倍）**。
原因：bn=16 迫使 QK 只用 1 个 warp，而该 warp 还要独自承担 512 宽的 PV。
配合此前 256 线程(ws)、双缓冲两个方向的实测失败，「增加计算 warp」已无可行分支。
→ 最终版固定为 `v300_final.py`（OJ 97.33，submission 167650）。

# ===== 交接（2026-11-12 收尾）=====

## 现状
- 最佳提交：**submission 167830，97.5（585/6）**，实时榜 **第 2 名（并列）**，榜一 97.83（587）。
- 文件：`v540_merge256.py` = `EurakaOpts/mla_decode/v036_merge256_97p5/submission.py`
- 逐 case：0.015 / 0.021 / 0.031 / 0.051 / 0.091 / 0.170 ms（display 96/97/98/98/98/98）

## 已验证的关键事实（都带生成代码校验或差分斜率）
1. **PV 必须跑满 4 warp**：M=16,N=512 在 256 线程下是合法 1×4（每 warp 128 列）。
   这是本会话最大的收益（判题机 -15%）。QK 是 M=16,N=32，只能待在 T.ws(0) 的 2 warp 里。
2. **MACA 的 `tl::__sync_thread_partial` 是空函数**（common.h:320）→ `T.ws` 内的同步全是 no-op，
   所有栅栏必须放在 ws 外面。v297 曾因此首次 launch 偶发 NaN。
3. **拷贝占 39% 但对分数贡献为 0**：删掉 KV/K_pe 拷贝（v590_nocopy，已校验）
   2048 15.20 / 4096 18.74 / 8192 24.95 —— display 一个都不变。
4. epilogue 写 1.4µs、merge 全局读 1.0µs（v612/v611，均校验过 generated code），删掉也不涨档。
5. launch 6.7µs，与 CTA 数无关（1/16/32/64/128 全 ~6.8µs）。
6. shared 上限 65536 B（`shared_memory_per_block_optin`，驱动报告）；KV tile 32768 B ⇒
   完整双缓冲 = 65536 B，K_pe/S/merge 暂存的 3KB 无处安放（字节级死锁）。
7. fp16 的 MMA 只有 16x16x16（/opt/maca/include 枚举）⇒ 136 条/tile 是算术下限。

## 若约束变化，按此顺序接
- **shared > 66KB**：上 3~4 片 dim 切分轮转（Q/Q_pe 进寄存器，gemm_rs 已验证可行）。
- **异步拷贝可用**（判题镜像的 maca_async_copy 目前数值错误）：直接把 KV 拷贝藏进计算。
- **更大 fp16 MMA**：QK 的 72 条和 PV 的 64 条可减半。
（注意：1 和 2 只减少拷贝，而第 3 条已证明拷贝对分数无影响——真正需要赢的是
 fixed 6.7 launch + 2.9µs split-K 尾部 + 3.3µs/tile 计算。）

## 不要重复的死路
256 线程 + ws（>1 tile 更慢）／maca_async_copy（判题 WA）／T.gemm 作 PV（慢 1.7x）／
8-warp PV（+3~6%）／寄存器或 fragment 预取（+58~60%）／KV 线性布局（+49%）／
手写 QK 交错双链（中性）／T.serial（中性）／num_stages=1（更慢）／
减少 CTA 数（launch 不变）／fallback merge（数值错误）。

# ===== 交接 2026-11-13（比赛 11-14 15:59Z 截止）=====

## 当前最优
`v920_noprobar.py` = `EurakaOpts/mla_decode/v038_psum_noprobar_97p5/submission.py`
submission **168408**，判题机 0.014/0.019/0.029/0.048/0.084/0.157，
display **585（97.5）**，mean_ratio **97.863**。实时榜第 2（与 C1040 并列），榜一 97.83。

## 与榜一的精确反解（本机设备时间口径）
榜一 judge 12(2048)/16(4096) ⇒ 设备约 11.0 / 15.0。
解 F + 1·T = 11、F + 2·T = 15 ⇒ **榜一: 每 tile 4.0µs，固定 7.0µs**。
我们: **每 tile 4.83µs，固定 8.06µs**。
差 = 0.83/tile + 1.06 fixed。而每 tile 的 MMA 地板是 2.79、拷贝 1.37 ⇒ 我们的 4.83 里
只剩 0.67 可动；固定 8.06 = launch 2.87 + epilogue 1.62 + partial 0.63 + 自旋 0.51 + merge 0.70 + Q 0.38 + 杂 1.3。

## 唯一没解释清的一块
`T.copy(acc_o, O_shared)`（16×512 fp32 fragment → fp16 shared，每线程 8 次 8B 存）**+1.45~1.62µs**。
已排除：bank conflict（padding +2/+4 生效但无变化）、转换吞吐（`__float22half2_rn` 打包指令）、
布局散射（转置/thread-major/直连全局三种写法都更慢 +3~4µs）、fp32 shared（更慢）。
**下一手应该从这里入手**：它在最小内核（无 MMA、无 merge、64 CTA）里也复现 +1.45µs，
是一个干净的、可独立优化的靶子。若能压到 0.3µs，2048 就到 11.5 → judge 12 → **97（586）**；
配合每 tile 再省 0.5，4096 到 15 → judge 16 → **98（587）**。

## 已封闭（勿重走）
更宽 fp16 MMA（`MACA_16x16x32_F32F16F16F32::fma` = 两次 16x16x16，模拟）；fp16 累加器不存在；
fp8 全链路 3.4~4.6% 误差；shared SM 总量 65536（双缓冲差 1024B）；split>64 每 tile +76%；
maca_async_copy 判题机 WA；每 SM 只驻留 1 个 CTA。
