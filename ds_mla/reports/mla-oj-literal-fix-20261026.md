> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# MLA OJ 全局字面量兼容修复

2026-10-26，Asia/Shanghai。

OJ 在加载源码时使用 `ast.literal_eval` 校验全局赋值，拒绝 `_PARTIAL_ELEM_BUDGET = 8 * 1024 * 1024`，因此原提交尚未进入算子执行。OJ 输出的零耗时、零加速比不是本次算子的性能测试结果。

通过 PostgreSQL-backed HeteMCP 调度 DSH 完成最小修复：将上述右侧改为等值整数字面量 `8388608`。Codex 未手改算子实现；最终原样复制 DSH artifact。

- HeteMCP task：`a6355add-eabc-4767-9ffc-10b98cd9c8d4`。
- 源码 SHA-256：`ff56144e534081f3174685803c593acae5e816c38746e57071c62c39a6e00c48`。
- 原源码 SHA-256：`2e16438207a28bf711bc8007c85fcc76b5afb07b0afb7934d031d580fa3d9fbc`。
- 将旧 AST 中这一常量等值规范化后，整个模块 AST 与新版本完全相同；函数体、装饰器、签名、调度和数值算法均未变更。
- 全部八项全局赋值均通过 `ast.literal_eval`，包含合法的空字典 `_KERNEL_CACHE = {}`。
- MetaX C500 上 case 1（不分片）及 case 3（split）重新测试通过，各使用种子 0、42，并检查原容差、输出原地写入与 NaN/Inf。
- 测试服务器未安装 OJ 的 `triton_sandbox.SafeExecutor` 或 `tilelang_runner.py`，没有声称完整 OJ 提交通过；需要使用更新后的源码重新提交 OJ。

交付文件：[本地 MLA.py](D:/workre/tzb/MLA.py)；服务器 `/data/tilelang-metax/race_tests/mla/MLA.py`。两端已核对与 DSH artifact 字节一致。

本轮旧版备份：[MLA.before-oj-20261026-3012d1240c.py](D:/workre/tzb/MLA.before-oj-20261026-3012d1240c.py)；服务器保留同名备份。

本轮仅修复源码加载兼容性，未修改此前记录的四组附加短序列数值容差限制。原始 31 组完整 GPU 结果见 [上一轮验收报告](D:/workre/HeteMCP/docs/mla-ssh-acceptance-20261026.md)，它对应的是上一版源码哈希；本轮仅重跑了上述两组回归。

原始证据：[静态审阅](D:/workre/HeteMCP/.heteromcp/mla-oj-20261026/patch-review.json)、[GPU 回归](D:/workre/HeteMCP/.heteromcp/mla-oj-20261026/gpu-smoke.json)、[发布与备份](D:/workre/HeteMCP/.heteromcp/mla-oj-20261026/publication-3012d1240c.json)。
