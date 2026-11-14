# XPUOJ 客户端

向 XPUOJ（LibreOJ）提交 TileLang 候选并等待判题结果的命令行工具。

## 文件

| 文件 | 用途 |
| --- | --- |
| `oj.py` | HTTP 客户端，含 PoW 挑战求解与 token 读取 |
| `loop.py` | 提交一个 `.py` 文件、轮询判题状态、把逐 case 结果写进 ledger |
| `precompile.py` | 提交前本地预编译，减少 CompileError 往返 |

## 认证

Token 从环境变量 `XPUOJ_TOKEN` 读取，缺省回落到 `~/.xpuoj_token`。

```bash
XPUOJ_TOKEN="$(cat ~/.xpuoj_token)" python loop.py submission.py --label my_label
```

不要把 token 写进任何被提交的文件。

## 输出

`loop.py` 把每次提交的 submissionId、逐 case 的 `kernel_ms`、`Score ratio` 与 `displayScore`
追加到 `--ledger` 指定的 JSONL（默认 `/root/lzy/workbench/oj_ledger.jsonl`）。
