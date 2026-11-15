# 终审材料包：DeepSeek MLA Decode

本目录是揭榜挂帅 XH-202608 赛题 “DeepSeek MLA Decode”的终审提交材料。

## 目录结构

```text
final_submission/
├── README.md                  本文件：材料说明、环境依赖、编译与运行方式
├── docs/
│   └── 技术方案.md                 技术方案文档（Kernel 设计、技术路线、创新点、实现步骤）
├── logs/
│   ├── build.log              编译日志：6 个计分 shape 的编译与首次运行
│   ├── functional_test.log    功能测试日志：数值正确性逐 shape 校验
│   ├── performance_test.log   性能测试日志：逐 shape 耗时与有效带宽
│   └── oj_results.md          OJ 判题结果：最终提交的原始输出与探索历程
└── src/
    ├── mla_decode_submission.py  提交到 OJ 的完整 TileLang 算子实现
    ├── build_all.py              编译检查
    ├── test_functional.py        功能测试
    └── test_performance.py       性能测试
```

## 成绩

| 项目 | 值 |
| --- | --- |
| OJ display | **98.17**（589/6） |
| mean_ratio | 98.3380 |
| 排名 | 第 1 名 |
| 提交号 | 168869 |
| 源码 md5 | `23ae084483f8a922c81feee644a4052d` |

`src/mla_decode_submission.py` 与 OJ 提交号 168869 的文件逐字节一致。

## 环境依赖

| 组件 | 版本 |
| --- | --- |
| GPU | MetaX C500（104 SM，64 KB 共享内存 / SM） |
| MACA | 3.7.1.5 |
| Kernel Mode Driver | 3.8.30 |
| Python | 3.12.11 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10+cuda.gitf549117c（沐曦分支） |

本机 TileLang 安装在 `/opt/tilelang-metax-v0.1.10`，需通过 `PYTHONPATH` 引入：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
```

## 编译

内核是单文件 TileLang 实现，没有独立的构建步骤。编译在首次调用 `run_kernel` 时由 `tilelang.jit` 自动完成，按 shape 特化并缓存。

执行编译检查：

```bash
cd src
python build_all.py
```

预期输出 6 个 shape 全部 `OK`，单次编译加首次运行约 600 毫秒～720 毫秒。完整输出见 `logs/build.log`。

## 运行测试

### 功能测试

```bash
cd src
python test_functional.py
```

对 6 个计分 shape 逐个校验：固定种子生成输入，与 fp32 参考实现比对绝对误差；换一组输入再跑一次，确认输出随输入变化；显式检查 NaN。容差为 `max_abs < 2e-3`。

实测最大绝对误差在 `5.7e-5`～`8.9e-5` 之间，全部通过。完整输出见 `logs/functional_test.log`。

### 性能测试

```bash
cd src
python test_performance.py
```

对 6 个计分 shape 计时：预热 20 次，每轮 50 次 launch，共 5 轮，取中位数，同时给出 KV 读取的有效带宽。

本机实测中位数：2048 为 15.18 µs，4096 为 18.41 µs，8192 为 25.46 µs，16384 为 35.33 µs，32768 为 53.19 µs，65536 为 91.06 µs。完整输出见 `logs/performance_test.log`。

### 关于本机与判题机的差异

本机 C500 是 50% 带宽切片，绝对耗时比判题机高 1 微秒～2 微秒。因此本机按官方基线折算的 display 为 97.83，而判题机实测为 98.17。

本机数据用于不同实现之间横向比较；最终成绩以 `logs/oj_results.md` 中的 OJ 判题结果为准。

## 可调参数

`test_functional.py` 与 `test_performance.py` 都支持 `--kernel PATH` 指定待测文件，便于与其它实现对比。

`test_performance.py` 另有 `--warmup`、`--iterations`、`--samples` 三个参数，默认值分别为 20、50、5。

## 说明

技术方案、创新点与各轮改进的实测依据见 `docs/技术方案.md`。优化过程中被实测否定的方向，在技术路线与创新点两节中一并说明。
