> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA：SSH 验收与优化结果

日期：2026-10-26，Asia/Shanghai。

后续仅修改全局常量字面量的 OJ 兼容补丁及当前交付哈希，见 [OJ 修复记录](D:/workre/HeteMCP/docs/mla-oj-literal-fix-20261026.md)。下文为本轮完整 GPU 验收时的历史快照。

**原始 JSON 的 31 组用例全部通过；31 组几何平均加速 2.32345×，最高 15.16860×。**
这一结论仅覆盖仓库提供的 31 组用例。补充的 5 组边界测试为 1 组通过、4 组未满足原容差；完整 36 组脚本因此以非零状态结束。未放宽容差，也未将补充失败写成通过。

## 交付与备份

- 本地最终源码：[MLA.py](D:/workre/tzb/MLA.py)。
- 服务器最终源码：`/data/tilelang-metax/race_tests/mla/MLA.py`。
- 发布前本地副本：`D:\workre\tzb\MLA.before-validated-20261026-67f9c320b3.py`。
- 更早的历史备份 `MLA.original-20261026.py` 与 `MLA.before-ssh-20261026-a57c1c819a.py` 保留。
- 最终 SHA-256：`2e16438207a28bf711bc8007c85fcc76b5afb07b0afb7934d031d580fa3d9fbc`。
- 原始服务器测试文件 SHA-256：`f4b651387eeef921abdda892e13857ba60d6cfda81a7011267153aabb72f75db`；原测试未被修改。
- 原始用例 JSON SHA-256：`bff60f6360cf1d2e4364ed772a3b5f98155a2e1111c6ddeab7b6e3af40e7201a`。

DSH artifact、服务器交付文件和本地 MLA.py 已逐字节核对一致。源码头部的“尚待验证”描述生成时的状态；本报告记录之后对同一 SHA-256 源码完成的实际验证，源码未被手改。

## 环境与构建

- GPU：MetaX C500，当前实例为 50% compute、约 32000 MiB 显存切片。
- MACA：3.5.3；现有 PyTorch：`2.8.0+metax3.5.3.9`。
- Python 环境：`/data/mla-validation-env-20261026-a57c1c819a`，使用 system-site-packages 复用现有 PyTorch。
- 源码与本地构建：`/data/tilelang-metax`、`/data/tilelang-metax/build`。
- 初次基线执行因未安装 TileLang 失败；直接导入源码也确认缺少 `build/lib` 与 `build/tvm`。
- 用户明确授权构建后，初始化已有仓库子模块、补齐构建依赖，并成功构建 TVM FFI、TVM 与 MACA TileLang；导入和 GPU 识别通过。
- 首次依赖下载在 900 秒超时，保留失败记录；随后限定 `ml-dtypes>=0.5.1,<0.6`、`numpy<2` 并复用缓存，构建成功。

SSH 最初在握手前断开；恢复后按用户终端提供的 ED25519 指纹固定校验连接。未删除用户的全局 known_hosts，也未将 SSH 密码写入源码、任务说明或报告。

## 改动与作者分工

Codex 负责上层分析、真实 HeteMCP MCP 调度、环境构建、审阅和 SSH 验收。所有算子实现及修复均由 DSH 编写；最终复制其完整原始字节。生产模块只导入 tilelang 与 tilelang.language；torch 仅用于单独的测试适配器和原始参考程序。

核心优化是对小 batch、长 context 启用有界 split-KV 并行，保持 FP32 LSE 和稳定合并；同时保留 split/tail mask 与 16 行对齐的 head tile。最初候选的 256 线程导致 MetaX 64-lane warp、16×16 MMA 的布局推导出现除零；DSH 根据真实错误将两个主路径改为已证实合法的 128 线程、零级流水线。

| HeteMCP 任务 | 产物/结果 |
|---|---|
| `a7b2de1e-1e8e-4397-8c34-a8717a0eaaf9` | 初始候选；实际 GPU 编译失败，记录保留 |
| `4257637c-63da-4336-ab3d-9f20ab0c7071` | 独立 GPU 验收脚本 |
| `d5d02bcc-b021-4fac-a427-ccb2a1ac1b1d` | 修复后的最终算子；原始 31 组验收通过 |

这些任务通过 PostgreSQL-backed HeteMCP 执行。任务 succeeded 仅表示 Harness 与文件范围检查完成；GPU 结论来自下述独立测试。usage.measured=false，费用/token 未测量。

## 验收与计时方法

- 不改动原始 `ref_program`，同一组 float16 CUDA 输入分别执行基线和候选。
- 每组使用随机种子 0、42；容差保留 `rtol=2e-4`、`atol=1e-4`、最大不匹配比例 0.01，与实际服务器原 profiler 默认配置核对一致。
- 候选 output 先填 NaN，检查原地输出、有限值和每次输出的误差统计。
- 编译与输入/参考分配均在计时区间外；预热 5 次，使用同步 CUDA events 测量 5 批、每批 20 次，报告每次调用耗时的批次中位数。
- 两个实现分别保持各自调用契约：原始基线 out_idx=[4] 返回输出，候选接收外部分配的 output 并原地写入。
- 原始 31 组均有双方通过后的有效计时。额外边界用例不计时、不计入性能汇总。
- 小 batch 长序列收益明显；短序列及 batch16/32 基本持平，微小正负差异不应解释为稳定收益。

## 原始 31 组实测

固定 heads=16、kv_heads=1、dim=512、pe_dim=64；以下两实现均通过两个种子的数值检查。

| case | batch | context | split | 原始 ms | 候选 ms | 加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2048 | 1 | 0.978150 | 0.978368 | 0.9998× |
| 2 | 1 | 4096 | 1 | 1.995354 | 1.990886 | 1.0022× |
| 3 | 1 | 8192 | 2 | 4.543526 | 2.534310 | 1.7928× |
| 4 | 1 | 16384 | 4 | 9.079053 | 2.574118 | 3.5271× |
| 5 | 1 | 32768 | 8 | 18.163994 | 2.586291 | 7.0232× |
| 6 | 1 | 65536 | 16 | 36.363327 | 2.605402 | 13.9569× |
| 7 | 2 | 2048 | 1 | 1.146893 | 1.145050 | 1.0016× |
| 8 | 2 | 8192 | 2 | 5.017690 | 2.561203 | 1.9591× |
| 9 | 2 | 16384 | 4 | 10.071309 | 2.593818 | 3.8828× |
| 10 | 2 | 32768 | 8 | 20.149542 | 2.605453 | 7.7336× |
| 11 | 2 | 65536 | 16 | 40.157300 | 2.648000 | 15.1651× |
| 12 | 4 | 2048 | 1 | 1.289152 | 1.289920 | 0.9994× |
| 13 | 4 | 8192 | 2 | 5.099443 | 2.596237 | 1.9642× |
| 14 | 4 | 16384 | 4 | 10.183527 | 2.609549 | 3.9024× |
| 15 | 4 | 32768 | 8 | 20.324646 | 2.642176 | 7.6924× |
| 16 | 4 | 65536 | 16 | 40.772223 | 2.687936 | 15.1686× |
| 17 | 8 | 2048 | 1 | 1.296486 | 1.295578 | 1.0007× |
| 18 | 8 | 8192 | 2 | 5.150080 | 2.620864 | 1.9650× |
| 19 | 8 | 16384 | 4 | 10.285990 | 2.646669 | 3.8864× |
| 20 | 8 | 32768 | 8 | 20.523187 | 2.687898 | 7.6354× |
| 21 | 8 | 65536 | 16 | 40.845096 | 5.300173 | 7.7064× |
| 22 | 16 | 2048 | 1 | 1.307469 | 1.306317 | 1.0009× |
| 23 | 16 | 8192 | 1 | 5.183258 | 5.179034 | 1.0008× |
| 24 | 16 | 16384 | 1 | 10.341670 | 10.341517 | 1.0000× |
| 25 | 16 | 32768 | 1 | 20.673575 | 20.669901 | 1.0002× |
| 26 | 16 | 65536 | 1 | 41.172684 | 41.161243 | 1.0003× |
| 27 | 32 | 2048 | 1 | 1.317376 | 1.315034 | 1.0018× |
| 28 | 32 | 8192 | 1 | 5.230170 | 5.230477 | 0.9999× |
| 29 | 32 | 16384 | 1 | 10.469043 | 10.471962 | 0.9997× |
| 30 | 32 | 32768 | 1 | 20.875571 | 20.872487 | 1.0001× |
| 31 | 32 | 65536 | 1 | 41.810739 | 41.804916 | 1.0001× |

## 补充边界测试的限制

下列失败是与原始 FP16 参考及同一严格容差的数值差异，均已成功编译并输出有限值。差异原因尚未进一步验证；不将其归因于某种精度机制，不承诺这些短序列/不规则 head 配置满足当前容差。

| 补充用例 | 结果 |
|---|---|
| `extra-run-heads7-ctx33` | 未满足原容差 |
| `extra-run-heads17-ctx129` | 未满足原容差 |
| `extra-run-heads16-ctx8193` | 通过 |
| `extra-direct-split3-heads16-ctx65` | 未满足原容差 |
| `extra-direct-split7-heads16-ctx33` | 未满足原容差 |

最大绝对误差约 1.46e-3；详细每种子误差、失败堆栈、全部输出有限性和原容差不匹配比例均保存在完整 JSON。最终交付以原始 JSON 的 31 组覆盖为验收范围；补充失败仍是后续扩展覆盖时需要处理的事项。

## 复现与原始证据

独立脚本和各轮原件保存在服务器 `/data/tilelang-metax/race_tests/mla/codex-20261026-a7b2de1e-a57c1c819a/`。

```bash
export MACA_PATH=/opt/maca
export USE_MACA=ON
export PYTHONPATH=/data/tilelang-metax
export PATH=/data/mla-validation-env-20261026-a57c1c819a/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:$LD_LIBRARY_PATH
stage=/data/tilelang-metax/race_tests/mla/codex-20261026-a7b2de1e-a57c1c819a
python "$stage/MLA-validation.py" --candidate "$stage/MLA.round2.py" --baseline "$stage/test_tilelang_mla.before.py" --cases "$stage/test_cases_mla_batch_ctx.json" --output "$stage/reproduce-original31.json" --warmup 5 --repeats 20
```

追加 `--extra-boundaries` 会同时执行已记录的补充测试，当前仍有 4 组失败。

- [完整 GPU 结果](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/acceptance-round2.json)
- [Codex 独立汇总](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/acceptance-round2.supervisor.json)
- [发布与哈希记录](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/validated-publication-67f9c320b3.json)
- [成功构建记录](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/build-environment-v2-0371afbb.json)
- [构建输出](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/build-environment-v2-0371afbb-1.log)
- [初次 GPU 编译失败记录](D:/workre/HeteMCP/.heteromcp/mla-ssh-20261026/smoke-initial.json)

`.heteromcp` 为 Git 忽略目录；其任务、日志和产物为本机运行证据，不等于已提交到远端仓库。
