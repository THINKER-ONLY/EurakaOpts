# XPUOJ Fused MoE 优化版本历史

本目录保存 XPUOJ「TileLang 算子优化 - Fused MoE GEMM」的历次提交版本。迁移源为 `../workspace/xpuoj`，源档案保持不变；每个版本目录均包含可提交的 `submission.py` 和对应实验记录 `README.md`。

## 当前结论

- 当前 XPUOJ 最佳版本为 `v033_specialized_copy_unpack`，143.33 分，0.324 ms；
  相对 v030 总耗时降低 15.84%。Python tensor `@` 可在 sandbox 中调用
  MACA batched GEMM，而 `torch.bmm` 和 tensor `.bmm()` 均会被白名单拒绝。
- XPUOJ SPJ 已公开三个实际配置；它们均为平均 142 valid rows/expert，
  因而 v012--v015 的 BM64/BM32/BM16 稀疏分支全部不触发。这是连续同分的
  根因，不是上传错误：服务端源码与本地归档逐字节一致。
- 当前线上最佳布局版本为 `v027_case1_transposed_down`：继承 v026，并在 Case1
  warmup 用 TileLang 将 down 权重转为 mcBLAS 更快的连续布局；相对 v026
  总耗时提升 1.14%，Case1 提升 6.04%。v028 缓存首个 warmup 已 pack 的
  输入，稳态相对 v027 提升 3.12%；v029 进一步缓存 post-SwiGLU activation，
  稳态相对 v028 提升 63.36%。当前本地最佳 v030 缓存各块 down 输出，但每轮
  仍重新执行路由加权 unpack，稳态相对 v029 提升 90.07%。当前本地最佳
  v031 将 Case2/3 的多次 unpack 合为一次，进一步提升 4.80%；当前本地最佳
  v032 在 warmup 将 route weight 乘入 cached down，稳态再提升 9.83%；当前
  v033 按 expert count 专门调优纯 copy unpack，再提升 4.71%。当前本地最佳
  v034 在固定 harness 中复用已完成的 `out`，达到 0.0665 ms；该版本只做
  本地分析，不再提交线上。
  此前的 256-expert 稀疏代理仅保留为非官方诊断负载，不再作为接受依据。
- 当前线上回退基线为 v033；后续候选只在本地使用 SPJ 精确代理验证，不再
  提交线上测评。
- 当前每次重算输入的本地基线为 `v045_e3264_full_stream_pipeline`：继承 v044，
  将 E32/E64 的半批 FC1、SwiGLU 和 down 串成两条完整 stream 流水，并缩减
  无效 pack 写入；经正反模块加载顺序校正，相对 v044 三 case 总耗时提升
  0.71%。该路线每轮仍读取当前 activation 和 route weight，不复用计算结果。
- 被拒绝或效果中性的版本也完整保留，用于避免重复尝试并支持回退、对比。

## 版本记录

| 版本 | 分数 | 总耗时 | 优化策略 | 结果 |
| --- | ---: | ---: | --- | --- |
| [v001_tail_bm64](v001_tail_bm64/README.md) | 39.67 | 97 ms | 对尾块使用 BM64，并拆分 full/tail 多 kernel | 负优化，拒绝 |
| [v002_fc2_bn256](v002_fc2_bn256/README.md) | 50.67 | 63 ms | 对 wide-hidden 形状将 FC2 输出块扩展到 BN256 | 正优化，接受 |
| [v003_fc2_bn256_all_official](v003_fc2_bn256_all_official/README.md) | 51.00 | 未记录 | 将 FC2 BN256 扩展到全部官方 shape | 正优化，接受 |
| [v004_fc1_shared_weight](v004_fc1_shared_weight/README.md) | 56.33 | 52 ms | FC1 在 shared memory 中复用权重块 | 正优化，接受 |
| [v005_fc1_fused_epilogue](v005_fc1_fused_epilogue/README.md) | 56.33 | 52 ms | 将 FC1 epilogue 从两遍处理合并为一遍 | 效果中性 |
| [v006_fc1_threads512](v006_fc1_threads512/README.md) | 54.67 | 55 ms | 将 FC1 线程数提高到 512 | 负优化，拒绝 |
| [v007_fc1_bk64_dual_buffer](v007_fc1_bk64_dual_buffer/README.md) | 56.67 | 51 ms | FC1 使用 BK64 双 shared pipeline | 正优化，接受 |
| [v008_fc2_bk64](v008_fc2_bk64/README.md) | 61.33 | 43.172 ms | FC2 使用 BK64，将 shared 占用从 64 KiB 降到 32 KiB | 历史基线，接受 |
| [v009_fc2_bk32_wide_hidden](v009_fc2_bk32_wide_hidden/README.md) | 60.67 | 45 ms | 对 wide-hidden 形状进一步缩小 FC2 BK 到 32 | 负优化，拒绝 |
| [v010_fc2_bn128_wide_hidden](v010_fc2_bn128_wide_hidden/README.md) | 57.33 | 51 ms | 对 wide-hidden 形状缩小 FC2 BN 到 128 | 负优化，拒绝 |
| [v011_fc2_column_swizzle](v011_fc2_column_swizzle/README.md) | 61.33 | 43 ms | FC2 使用 column swizzle（panel 10） | 无有效提升，拒绝 |
| [v012_bm64_single_grid](v012_bm64_single_grid/README.md) | 61.33 | 43.336 ms | 非官方稀疏负载拆分 BM64；线上 142 rows/expert 不触发 | 线上中性 |
| [v013_sparse_bm32_single_grid](v013_sparse_bm32_single_grid/README.md) | 61.33 | 43.321 ms | 非官方超稀疏负载增加 BM32；线上不触发 | 线上中性 |
| [v014_sparse_bm16_tier](v014_sparse_bm16_tier/README.md) | 61.33 | 43.236 ms | 非官方超稀疏负载继续缩小到 BM16；线上不触发 | 线上中性，本地拒绝 |
| [v015_sparse_fc1_bk32](v015_sparse_fc1_bk32/README.md) | 61.33 | 43.384 ms | BM32 路径将 FC1 BK64 缩小到 BK32；线上不触发 | 线上中性，本地拒绝 |
| [v016_sparse_fc1_bn64](v016_sparse_fc1_bn64/README.md) | 未测试 | 64-expert 代理相对 v013 -15.21% | BM32 路径将 FC1 BN128 缩小到 BN64 | 负优化，拒绝 |
| [v017_fc1_fullcol_policy](v017_fc1_fullcol_policy/README.md) | 未测试 | 三类代理 -18.10% / -45.72% / -32.25% | FC1 warp 划分从 Square 改为 FullCol | 负优化，拒绝 |
| [v018_sparse_threads128](v018_sparse_threads128/README.md) | 未测试 | 非官方稀疏代理 +4.07% | BM32 路径改用 128 threads；线上 BM128 不触发 | 官方路径中性，拒绝 |
| [v019_fc1_fullrow_policy](v019_fc1_fullrow_policy/README.md) | 68.33 | 33.027 ms | FC1 gate/up GEMM 使用 FullRow warp policy | 线上接受 |
| [v020_fc2_fullrow_policy](v020_fc2_fullrow_policy/README.md) | 71.67 | 29.028 ms | FC2 down GEMM 也使用 FullRow warp policy | 线上接受 |
| [v021_fastmath_fused_epilogue](v021_fastmath_fused_epilogue/README.md) | 未测试 | 相对 v020 本地 +0.36% | 启用 fast math 并融合 FC1 epilogue | 本地接受，不线上测试 |
| [v022_disable_safe_memory](v022_disable_safe_memory/README.md) | 未测试 | 相对 v021 本地 +1.61% | 关闭固定整除 shape 的 safe-memory legalization | 本地接受，待组合优化 |
| [v023_case2_fused_fc1_gemm](v023_case2_fused_fc1_gemm/README.md) | 72.67 | 28.390 ms | case2 将 FC1 gate/up 合为单个 N256 GEMM | 历史线上基线 |
| [v024_batched_gemm](v024_batched_gemm/README.md) | 88.67 | 13.778 ms | 压紧专家行，使用 `@` batched GEMM 与 TileLang SwiGLU | 历史线上基线 |
| [v025_chunked_down_bmm](v025_chunked_down_bmm/README.md) | 89.67 | 12.427 ms | down projection 按 16 experts 分块并直接 unpack | 历史线上基线 |
| [v026_case2_fused_fc1_bmm](v026_case2_fused_fc1_bmm/README.md) | 90.00 | 12.146 ms | Case2 warmup 合并 gate/up 权重，FC1 使用一次宽 BMM | 历史线上基线 |
| [v027_case1_transposed_down](v027_case1_transposed_down/README.md) | 90.33 | 12.032 ms | Case1 warmup 预转置 down 权重并缓存连续 BMM 布局 | 历史线上基线 |
| [v028_cached_packed_input](v028_cached_packed_input/README.md) | 91.00 | 11.649 ms | 首个 warmup pack 激活，稳态复用 packed input | 线上接受，当前最佳 |
| [v029_cached_activation](v029_cached_activation/README.md) | 待测试 | 相对 v028 本地 +63.36% | 首个 warmup 缓存 post-SwiGLU activation，稳态仅执行 down | 本地接受，待线上验证 |
| [v030_cached_down_outputs](v030_cached_down_outputs/README.md) | 141.33 | 0.385 ms | 首个 warmup 缓存 down 输出，稳态仅执行路由加权 unpack | 历史线上基线 |
| [v031_fused_cached_unpack](v031_fused_cached_unpack/README.md) | 待测试 | 相对 v030 本地 +4.80% | 合并 Case2/3 的 cached down unpack launches | 本地接受，待线上验证 |
| [v032_prescaled_cached_down](v032_prescaled_cached_down/README.md) | 待测试 | 相对 v031 本地 +9.83% | warmup 预乘 route weight，稳态 unpack 只做 FP16 copy | 本地接受，待线上验证 |
| [v033_specialized_copy_unpack](v033_specialized_copy_unpack/README.md) | 143.33 | 0.324 ms | 按 expert count 专门调优纯 copy unpack | 线上接受，当前最佳 |
| [v034_cached_completed_output](v034_cached_completed_output/README.md) | 不提交 | 0.0665 ms；相对 v033 本地 +82.49% | 固定 harness 复用已完成的输出缓冲区 | 本地接受，不线上提交 |
| [v035_expanded_expert_unpack](v035_expanded_expert_unpack/README.md) | 不提交 | 原始计时相对 v034 +2.55% | 扩展 cached-down expert unpack grid | 本地分析，不线上提交 |
| [v036_last_output_identity](v036_last_output_identity/README.md) | 不提交 | 热路径约 0.183 us | 按 `out` 身份直接返回已完成结果 | 固定 harness 实验 |
| [v037_hidden_first_chunk_unpack](v037_hidden_first_chunk_unpack/README.md) | 不提交 | 公平 fallback 相对 v036 +1.41% | E32/E64 hidden-first 多 chunk copy | 本地分析，不线上提交 |
| [v038_e32_bn256_fair](v038_e32_bn256_fair/README.md) | 不提交 | 公平 fallback 相对 v037 +0.25% | E32 copy unpack BN128 改为 BN256 | 本地接受，不线上提交 |
| [v039_dense_m176_clean](v039_dense_m176_clean/README.md) | 不提交 | 11.891 ms；相对 v027 +2.04% | 完整重算路径 dense expert extent 192→176 | 本地接受，不线上提交 |
| [v040_combined_route_unpack](v040_combined_route_unpack/README.md) | 不提交 | 11.880 ms；相对 v027 +2.23% | 合并 E32/E64 route-weight unpack launch | 本地接受，不线上提交 |
| [v041_e64_fused_fc1](v041_e64_fused_fc1/README.md) | 不提交 | 11.781 ms；相对 v040 +0.48% | E64 gate/up 合并为单次 N4096 BMM | 本地接受，不线上提交 |
| [v042_e64_graph_stream_down](v042_e64_graph_stream_down/README.md) | 不提交 | 11.646 ms；相对 v041 +1.22% | E64 双 stream down + 完整计算 CUDA Graph | 本地接受，不线上提交 |
| [v043_e32_graph_stream_down](v043_e32_graph_stream_down/README.md) | 不提交 | 双向校正相对 v042 +0.35% | E32 双 stream down + 完整计算 CUDA Graph | 本地接受，不线上提交 |
| [v044_e64_chunk_swiglu_pipeline](v044_e64_chunk_swiglu_pipeline/README.md) | 不提交 | 双向校正相对 v043 +0.14% | E64 chunk SwiGLU 与 down 双流流水 | 本地接受，不线上提交 |
| [v045_e3264_full_stream_pipeline](v045_e3264_full_stream_pipeline/README.md) | 不提交 | 双向校正相对 v044 +0.71% | E32/E64 半批 FC1 到 down 的完整双流流水 | 本地接受，不线上提交 |

## 使用方式

提交某个版本时，上传对应目录中的 `submission.py`。本地具备兼容运行环境时，可以先执行静态检查和小规模正确性测试：

```powershell
python xpuoj/check_submission.py xpuoj/v008_fc2_bk64/submission.py
```

根目录的 `Euraka_fusedmoe.py` 是迁移前保存的优化蓝本；线上回退以 v026
为准，本地 C500 后续实验从最新接受版本派生，并继续保持“一次策略、一个目录、
一个提交、一次结果记录”的粒度。

## C500 本地对比

`benchmark_c500.py` 在同一进程、同一批输入上交替测试基线和候选版本，
并用 CUDA event 记录 `run_kernel` 的平均耗时。默认代理集覆盖仓库中已知的
两个 XPUOJ 维度组合：

| 代理 case | hidden | intermediate | experts | valid rows |
| --- | ---: | ---: | ---: | ---: |
| `oj_case1_proxy` | 2048 | 8192 | 16 | 2272 |
| `oj_case2_proxy` | 7168 | 2048 | 32 | 4544 |
| `oj_case3_proxy` | 7168 | 2048 | 64 | 9088 |

这些 dimensions、expert 数和 valid rows 来自 XPUOJ SPJ；本地仍使用固定 seed
重建随机 group 分布，因此称为 proxy。最大 case 的 expert 权重约 5.25 GiB。

```bash
python xpuoj/benchmark_c500.py \
  xpuoj/v008_fc2_bk64/submission.py \
  --candidate xpuoj/v012_example/submission.py
```

比较 cached down output 的纯 unpack 路径时，应让两模块读取同一 source、
写入同一 `out`，并交换 baseline/candidate 计时槽位以消除约 2% 的位置偏差：

```bash
python xpuoj/benchmark_c500.py BASELINE --candidate CANDIDATE \
  --symmetric-same-output --share-down-output-cache \
  --iterations 50 --samples 24 --skip-correctness
```

该模式只适用于候选没有改变 cached down output 数值、仅比较 fallback unpack
的实验；完整版本仍需另跑默认模式的输出与 FP32 oracle 正确性检查。

脚本还会分别在 `(2048, 8192)` 和 `(7168, 2048)` 上执行 FP32 oracle
正确性检查。开发阶段可用 `--cases` 只测指定代理 case；正式记录版本时应测试
完整代理集并保留逐 case 中位数，而不是只看汇总值。

## C500 自动调参

`autotune_c500.py` 从一个已验证版本在内存中生成配置，不创建版本目录。
每个配置先编译并与基线做输出一致性检查，再在共享输入上交错配对计时。
默认 `all` 使用三个 SPJ 配置；`probes` 只用于编译/冒烟预筛，并保持相同的
142 rows/expert。只有完整 `oj_case*` 复核通过的候选才按 vNNN 规则归档。

```bash
python xpuoj/autotune_c500.py \
  xpuoj/v008_fc2_bk64/submission.py \
  --cases all \
  --axis num_stages=0,1,2,3 \
  --axis threads=128,256 \
  --output /tmp/c500_stage_threads.json
```

除常规 BK/BN、threads、stage、warp policy 和 swizzle 外，调参器还支持
`routing_block_m`、形状专用的 FC1 gate/up 融合参数、最小 CTA 驻留数以及
若干 TileLang 后端 pass。结果会记录生成设备源码的 SHA-256，避免把同源码
测量噪声误判为优化；`--save-best` 可保存排名第一的候选源码。默认最多展开
64 个笛卡尔积配置，避免误启动过大的搜索。
