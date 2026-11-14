# MLA-OP

MLA decode 优化版本的独立备份。

每个子目录是该版本在 OJ 上提交过的文件副本，与 `mla_decode/v*/` 内容一致，
放在这里是为了在一处能直接取到任一版本的提交件。版本演进与实测数据见
`docs/mla_decode_optimization_report.md`。

## 本次会话的版本

| 版本 | 关键改动 | OJ display | 提交号 |
| --- | --- | ---: | --- |
| `v039_split104_aligned_97p67` | split 边界按 tile 对齐 + 104 CTA | 97.67 | 168534 |
| `v042_stage32_587` | 整块 32 键暂存进寄存器，LDG 提前、STS 延后 | 97.83 | 168733 |
| `v043_fast4096_588` | 4096/8192 改走 3-barrier fast path；修 O_shared/S_shared bank conflict | 98.00 | 168830 |
| `v044_sole_first_589` | 2048 路径补 O_shared padding；到达自旋改 volatile load | **98.17** | 168869 |

`v041_merge_latency_study` 不是可提交版本，是 2048 的合并延迟诊断脚本。

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
