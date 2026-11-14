# EurakaOpts

XPUOJ 赛题的内核优化存档。每个赛题一个目录，每个被接受的优化对应一个版本目录和一个提交。

## 目录

| 目录 | 内容 |
| --- | --- |
| `mla_decode/` | DeepSeek MLA Decode，按版本归档的提交件、研究笔记与实验脚本 |
| `xpuoj/` | fused MoE 与 native sparse attention 方向的版本归档 |
| `native_sparse_attention/` | 稀疏注意力的参考实现与试验 |
| `MLA-OP/` | MLA 各优化版本提交件的独立备份 |
| `tools/xpuoj/` | 向 XPUOJ 提交并拉取判题结果的客户端 |
| `docs/` | 优化报告 |

## 文档

- [MLA Decode 优化报告](docs/mla_decode_optimization_report.md)
- [MLA 版本表](mla_decode/README.md)

## 约定

- 每个版本一个目录，目录名形如 `vNNN_简述_成绩`
- 提交件统一命名 `submission.py`
- 生成物（`*.log`、`*.npy`、`*.pt`、`*.cu`、`__pycache__`）不入库
