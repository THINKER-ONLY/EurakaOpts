# MLA-OP

MLA decode 优化版本的独立备份。

每个子目录是该版本在 OJ 上提交过的文件副本，与 `mla_decode/v*/` 内容一致，
放在这里是为了在一处能直接取到任一版本的提交件。版本演进与实测数据见
`docs/mla_decode_optimization_report.md`。

## 终审材料包

`终审材料/` 是按赛题要求整理的终审提交材料，包含技术方案文档、
编译与测试日志、OJ 判题结果，以及与提交号 168869 逐字节一致的源代码。

```text
终审材料/
├── README.md            材料索引
├── 1-算法设计报告.md     技术方案、创新点、实现步骤、算法伪代码
├── 2-功能测试报告.md     参数设置、golden 定义、逐 case 精度对比
├── 3-性能测试报告.md     参数设置、逐 case 结果、瓶颈分析
├── 4-源代码/             完整实现 + 测试代码 + 编译运行说明
└── 附录/                 四份原始运行输出
```

## 技术方案与归档的对照

`终审材料/1-算法设计报告.md` 的探索历程共 19 步，每一步的实现都收在本目录：

| 步骤 | 关键改动 | display | 目录 |
| ---: | --- | ---: | --- |
| 1 | 官方模板归档 | — | `v001_official_baseline` |
| 2 | 8 路 context split 加 4 个输出分区 | — | `v002_split8_output_quarters` |
| 3 | split 数按 batch 选择 | — | `v003_batch_split_policy` |
| 4 | 前缀缩放权重 | — | `v012_prefix_scaled_weights` |
| 5 | 共享 KV 首块 | — | `v013_shared_kv_part0` |
| 6 | PV 与 value tile 载入流水 | — | `v025_b32_pv_pipeline` |
| 7 | 恢复 OJ 源码 | 96.83 | `v028_oj163109_96p83_recovered` |
| 8 | 去掉 max 跟踪 | 97.00 | `v029_nomax_fused_97p0` |
| 9 | 归一化推迟到归并 | 97.17 | `v030_defer_div_97p17` |
| 10 | 关闭 swizzle | 97.17 | `v031_noswz_97p17` |
| 11 | 混合路径 | 97.33 | `v033_hybrid_97p33` |
| 12 | PV 改 4 warp | 97.33 | `v035_pv4warp_97p33` |
| 13 | 归并用满 256 线程 | 97.50 | `v036_merge256_97p5` |
| 14 | 行和融进 PV 宏 | 97.50 | `v037_psum_97p5` |
| 15 | 去掉 prologue 屏障 | 97.50 | `v038_psum_noprobar_97p5` |
| 16 | split 边界按 tile 对齐 | 97.67 | `v039_split104_aligned_97p67` |
| 17 | 32 键整块寄存器暂存 | 97.83 | `v042_stage32_587` |
| 18 | 快速路径下放 + 两处 bank 冲突 | 98.00 | `v043_fast4096_588` |
| 19 | 2048 补 padding + 自旋改 volatile load | 98.17 | `v044_sole_first_589` |

以下两个目录不产生成绩，作为证据保留：

- `v041_merge_latency_study`：2048 的合并延迟诊断脚本
- `v045_rejected_qk_ksplit`：被否定的 QK K 维拆分实验，印证技术方案第 8 节的结论

## 全部备份版本

- `v001_official_baseline`
- `v002_split8_output_quarters`
- `v003_batch_split_policy`
- `v004_batch16_split8`
- `v005_adaptive_split_policy`
- `v006_partitioned_reduce`
- `v007_reduce_64threads`
- `v008_adaptive_reduce8`
- `v009_materialized_scores`
- `v010_materialized_weights`
- `v011_adaptive_norm_threads`
- `v012_prefix_scaled_weights`
- `v013_shared_kv_part0`
- `v014_fused_qk576`
- `v015_qk_clear_accum`
- `v016_cached_split_lse`
- `v017_adaptive_reduce_max_clear`
- `v018_shared_score_pv`
- `v019_direct_shared_score_pv`
- `v020_adaptive_qk_no_part0`
- `v021_adaptive_factored_softmax_scale`
- `v022_prescaled_long_q`
- `v023_adaptive_num_split`
- `v024_b1_materialized_scores`
- `v025_b32_pv_pipeline`
- `v026_b32_qk_part0`
- `v027_b32_8k_fused_split32`
- `v028_oj163109_96p83_recovered`
- `v029_nomax_fused_97p0`
- `v030_defer_div_97p17`
- `v031_noswz_97p17`
- `v033_hybrid_97p33`
- `v034_final_97p33`
- `v035_pv4warp_97p33`
- `v036_merge256_97p5`
- `v037_psum_97p5`
- `v038_psum_noprobar_97p5`
- `v039_split104_aligned_97p67`
- `v041_merge_latency_study`
- `v042_stage32_587`
- `v043_fast4096_588`
- `v044_sole_first_589`
- `v045_rejected_qk_ksplit`
