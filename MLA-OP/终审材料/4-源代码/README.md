# 源代码说明

本目录是 MLA Decode 内核的完整 TileLang 实现，以及配套的功能测试与性能测试代码。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `mla_decode_submission.py` | 算子实现，即提交到 OJ 的文件 |
| `build_all.py` | 编译检查，对 6 个计分 shape 各编译一次 |
| `test_functional.py` | 功能测试，与 fp32 参考实现逐元素比对 |
| `test_performance.py` | 性能测试，逐 shape 计时并折算有效带宽 |

`mla_decode_submission.py` 与 OJ 提交号 168869 的文件逐字节一致，md5 为 `23ae084483f8a922c81feee644a4052d`。

## 环境依赖

| 组件 | 版本 |
| --- | --- |
| GPU | MetaX C500，104 SM，每 SM 65,536 B 共享内存 |
| MACA | 3.7.1.5 |
| Kernel Mode Driver | 3.8.30 |
| Python | 3.12.11 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10+cuda.gitf549117c，沐曦分支 |

本机 TileLang 安装在 `/opt/tilelang-metax-v0.1.10`，需通过 `PYTHONPATH` 引入：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
```

除上述依赖外，内核不引入任何第三方库。

## 编译

内核是单文件 TileLang 实现，没有独立的构建步骤。编译在首次调用 `run_kernel` 时由 `tilelang.jit` 自动完成，按 shape 特化并缓存。

执行编译检查：

```bash
cd 4-源代码
python build_all.py
```

预期输出 6 个 shape 全部 `OK`，单次编译加首次运行约 600 至 720 毫秒。

## 功能测试

```bash
cd 4-源代码
python test_functional.py
```

对 6 个计分 shape 逐个校验：固定种子生成输入，与 fp32 参考实现比对绝对误差；换一组种子再跑一次，确认输出随输入变化；显式检查 NaN。

容差为 `max_abs < 2e-3`，实测最大绝对误差为 `8.941e-05`。

预期输出 6 个 shape 全部 `PASS`，末行 `结果：ALL PASS`。退出码 0 表示全部通过。

## 性能测试

```bash
cd 4-源代码
python test_performance.py
```

对 6 个计分 shape 计时：预热 20 次，每轮 50 次启动，共 5 轮，取中位数，同时给出 KV 读取的有效带宽，以及按官方基线折算的 display 估计。

预期输出 6 行测量结果，末行为 `display 估计均值`。

## 可调参数

两个测试脚本都支持以下参数：

| 参数 | 适用 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--kernel PATH` | 两者 | 同目录的 `mla_decode_submission.py` | 指定待测文件，便于与其它实现对比 |
| `--warmup N` | 性能测试 | 20 | 预热次数 |
| `--iterations N` | 性能测试 | 50 | 每轮迭代次数 |
| `--samples N` | 性能测试 | 5 | 采样轮数 |

示例，测另一份实现：

```bash
python test_performance.py --kernel ../其它实现.py --iterations 100
```

## 输入输出约定

`run_kernel` 的签名与赛题框架一致：

```python
run_kernel(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim)
```

所有张量为连续存放的 fp16，`kv_heads` 固定为 1，结果写回 `output`。

模块内部缓存已编译的内核，并按 shape 各存一份到达计数器，因此同一进程内可以连续调用不同 shape。

## 关于本机与判题机的差异

本机 C500 是 50% 带宽切片，绝对耗时比判题机高 1 至 2 微秒。本机按官方基线折算的 display 为 97.83，判题机为 98.17。

本机数据用于不同实现之间的横向比较，最终成绩以判题机为准，详见 `../3-性能测试报告.md`。
