# XPUOJ Fused MoE 优化版本历史

本目录保存 XPUOJ「TileLang 算子优化 - Fused MoE GEMM」的历次提交版本。迁移源为 `../workspace/xpuoj`，源档案保持不变；每个版本目录均包含可提交的 `submission.py` 和对应实验记录 `README.md`。

## 当前结论

- 当前最佳版本：`v008_fc2_bk64`，61.33 分，43 ms。
- `v011_fc2_column_swizzle` 同为 61.33 分，没有带来有效提升，因此仍以 v008 为后续 C500 调优基线。
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
| [v008_fc2_bk64](v008_fc2_bk64/README.md) | 61.33 | 43 ms | FC2 使用 BK64，将 shared 占用从 64 KiB 降到 32 KiB | 当前最佳，接受 |
| [v009_fc2_bk32_wide_hidden](v009_fc2_bk32_wide_hidden/README.md) | 60.67 | 45 ms | 对 wide-hidden 形状进一步缩小 FC2 BK 到 32 | 负优化，拒绝 |
| [v010_fc2_bn128_wide_hidden](v010_fc2_bn128_wide_hidden/README.md) | 57.33 | 51 ms | 对 wide-hidden 形状缩小 FC2 BN 到 128 | 负优化，拒绝 |
| [v011_fc2_column_swizzle](v011_fc2_column_swizzle/README.md) | 61.33 | 43 ms | FC2 使用 column swizzle（panel 10） | 无有效提升，拒绝 |

## 使用方式

提交某个版本时，上传对应目录中的 `submission.py`。本地具备兼容运行环境时，可以先执行静态检查和小规模正确性测试：

```powershell
python xpuoj/check_submission.py xpuoj/v008_fc2_bk64/submission.py
```

根目录的 `Euraka_fusedmoe.py` 是迁移前保存的优化蓝本；后续设备相关实验应从 v008 派生新版本，并继续保持“一次策略、一个目录、一个提交、一次分数记录”的粒度。
