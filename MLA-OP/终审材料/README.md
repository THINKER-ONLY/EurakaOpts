# 终审材料：DeepSeek MLA Decode

本目录是揭榜挂帅 XH-202608 赛题 “DeepSeek MLA Decode”的终审提交材料，按赛题要求的四份文件组织。

## 目录结构

```text
终审材料/
├── README.md                本文件
├── 1-算法设计报告.md         技术方案、创新点、实现步骤、算法伪代码
├── 2-功能测试报告.md         参数设置、golden 定义、逐 case 精度对比
├── 3-性能测试报告.md         参数设置、逐 case 结果、瓶颈分析
├── 4-源代码/
│   ├── README.md            编译、功能测试与性能测试的运行方式
│   ├── mla_decode_submission.py
│   ├── build_all.py
│   ├── test_functional.py
│   └── test_performance.py
└── 附录/
    ├── build.log            编译原始输出
    ├── functional_test.log  功能测试原始输出
    ├── performance_test.log 性能测试原始输出
    └── oj_results.md        判题机的逐 case 判题输出
```

## 四份材料

| 序 | 材料 | 文件 | 内容 |
| ---: | --- | --- | --- |
| 1 | 算法设计报告 | `1-算法设计报告.md` | Kernel 设计方案、技术路线、创新点、实现步骤、算法伪代码 |
| 2 | 功能测试报告 | `2-功能测试报告.md` | 6 个计分 case 的参数、golden 定义、精度比对结果 |
| 3 | 性能测试报告 | `3-性能测试报告.md` | 计时参数、本机与判题机结果、成本模型与瓶颈分析 |
| 4 | 源代码 | `4-源代码/` | 完整 TileLang 实现、测试代码、编译与运行说明 |

## 成绩

| 项目 | 值 |
| --- | --- |
| OJ display | 98.17（589/6） |
| mean_ratio | 98.3380 |
| 排名 | 第 1 名 |
| 提交号 | 168869 |
| 源码 md5 | `23ae084483f8a922c81feee644a4052d` |

逐 case：

| case | 基线 / ms | 内核 k / ms | ratio | display |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 0.419 | 0.012 | 97.22% | 97 |
| 4096 | 0.789 | 0.016 | 98.01% | 98 |
| 8192 | 1.526 | 0.023 | 98.52% | 98 |
| 16384 | 3.000 | 0.031 | 98.98% | 98 |
| 32768 | 5.958 | 0.050 | 99.17% | 99 |
| 65536 | 11.840 | 0.088 | 99.26% | 99 |

`4-源代码/mla_decode_submission.py` 与 OJ 提交号 168869 的文件逐字节一致。

## 快速复现

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
cd 4-源代码
python build_all.py         # 编译检查
python test_functional.py   # 功能测试
python test_performance.py  # 性能测试
```

三条命令分别在 1 分钟内完成，详细说明见 `4-源代码/README.md`。

## 说明

测试集合即赛题的 6 个计分 case，赛题未另行提供 golden 数据文件，功能测试以全精度 fp32 参考实现作为 golden，判据与理由见 `2-功能测试报告.md` 第 2 节。

附录保存的是原始运行输出，正文中的结论均可据此复核。
