# OJ 判题结果

本文件记录最终提交在 XPU-OJ 平台上的原始判题结果。

## 1. 最终提交

| 项目 | 值 |
| --- | --- |
| 提交号 | 168869 |
| 标签 | `v994_skew_spin` |
| 状态 | Accepted |
| 提交时间 | 2026-11-13T16:59:40 |
| display 总分 | **98.17** |
| mean_ratio | 98.3380 |
| 排名 | **第 1 名** |
| 源码 md5 | `23ae084483f8a922c81feee644a4052d` |

源码即本材料包中的 `src/mla_decode_submission.py`，与该次提交逐字节一致。

## 2. 逐 case 判题输出

判题机对同一份代码报告了 7 行结果，其中 kv_ctx=2048 出现两次（第二次不计分）。

```text
  97   97.22% base=0.419 kernel=0.012 x34.917  batch=1, kv_ctx=2048,  heads=16, kv_heads=1, dim=512, pe_dim=64
  97   97.22% base=0.419 kernel=0.012 x34.917  batch=1, kv_ctx=2048,  heads=16, kv_heads=1, dim=512, pe_dim=64
  98   98.01% base=0.789 kernel=0.016 x49.312  batch=1, kv_ctx=4096,  heads=16, kv_heads=1, dim=512, pe_dim=64
  98   98.52% base=1.526 kernel=0.023 x66.348  batch=1, kv_ctx=8192,  heads=16, kv_heads=1, dim=512, pe_dim=64
  98   98.98% base=3.000 kernel=0.031 x96.774  batch=1, kv_ctx=16384, heads=16, kv_heads=1, dim=512, pe_dim=64
  99   99.17% base=5.958 kernel=0.050 x119.160 batch=1, kv_ctx=32768, heads=16, kv_heads=1, dim=512, pe_dim=64
  99   99.26% base=11.84 kernel=0.088 x134.545 batch=1, kv_ctx=65536, heads=16, kv_heads=1, dim=512, pe_dim=64
display=98.17 mean_ratio=98.33805714285714
```

## 3. 计分换算

计分公式为 `ratio = base / (base + k)`、`display = floor(100 * ratio)`，总分为 6 个 `display` 的整数均值。

| case | base / ms | k / ms | ratio | display |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 0.419 | 0.012 | 97.216% | 97 |
| 4096 | 0.789 | 0.016 | 98.012% | 98 |
| 8192 | 1.526 | 0.023 | 98.515% | 98 |
| 16384 | 3.000 | 0.031 | 98.977% | 98 |
| 32768 | 5.958 | 0.050 | 99.168% | 99 |
| 65536 | 11.840 | 0.088 | 99.262% | 99 |

合计 `97 + 98 + 98 + 98 + 99 + 99 = 589`，`589 / 6 = 98.17`。

## 4. 版本演进

本方案在 OJ 上逐版实测，关键节点如下：

| 提交号 | 标签 | display | 主要改动 |
| --- | --- | ---: | --- |
| 168408 | `psum_noprobar` | 97.50 | psum 行和 + 去掉 prologue barrier |
| 168534 | `split104_aligned` | 97.67 | split 边界按 tile 对齐，`num_split` 提到 104 |
| 168733 | `stage32_regstage` | 97.83 | 32 键整块寄存器暂存，LDG 提前、STS 延后 |
| 168830 | `fast4096` | 98.00 | 4096/8192 改走 3 barrier fast path，修 O/S 的 bank conflict |
| **168869** | **`v994_skew_spin`** | **98.17** | 2048 路径补 `O_shared` padding，到达自旋改 volatile load |

## 5. 说明

判题机的内核耗时按微秒量化后再参与比例计算，因此相邻两档常常只差 1 微秒。例如 2048 档升至 98 分需要 `k ≤ 0.00855`，而当前为 0.012；16384 档升至 99 分需要 `k ≤ 0.0303`，当前为 0.031，仅差 1 微秒的量化步长。

上述未达档位的可行性已在 `docs/technical-solution.md` 第 8 节逐条给出实测依据。
