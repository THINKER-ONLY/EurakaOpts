> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# OJ 483μs 后续实验（2026-10-28）

固定生产 SHA：def723b6f97f9cb2d0aa377c762be7e082a9fee40a98b705d942a53c34db0790。
用户报告 OJ 六点 20/35/39/64/114/211μs，合计483μs，96.67分。源码已通过真实OJ编译/评测；未独立获取提交SHA。相对上一轮490μs只少7μs，未重复测量，不能把SSH1.504×映射成OJ收益。
OJ GPU C500已由用户确认；全卡只是用户推测。SSH为50%C500，TileLang0.1.9；OJ为0.1.10，不能视作相同编译或调度环境。

所有实现与性能/验证脚本均由真实HeteMCP→DSH生成；Codex只做架构、编排、差分审阅、证据分析。实验阶段固定生产基线不变；通过验收后交付长ctx split256候选，见文末。

## 独立实验

1. QK512与PE64融合为576 reduction。DSH task `21579b62-7c12-4ac6-ab5c-45692de37b15`，attempt `c01cd541-1b79-49c4-a087-c338903b3c35`。候选SHA `044415dc089c2d646e1cc60f17d91c89db0ea6cf7fdab4028cdbb6418417bc29`。静态差分仅flashattn内部，PV宏与merge AST不变。核心六点双seed原容差通过，加速比0.996426/0.982865/1.001858/0.999567/0.999261/1.000772。无有效收益，否决，不跑31点、不发布。
2. merge输出tile64→32。DSH task `9e0a577a-45de-41f7-a9c2-73c31b7f8a80`，attempt `28cfb852-de56-4999-b639-931e4cd7c02d`，SHA `fbc9cb2c4e3f4aea1ebd8fa56bd598f26fde7924fb66fc19456efa2ea1f03024`。已逐字节确认只修改 `_MERGE_TILE_DIM = 64` 为32。目的：增加merge CTA、减小每线程片段，独立测量额外LSE读取/reduction的代价。核心六点双seed、原容差全部通过，加速比0.986335/0.991084/1.000142/0.999321/1.000000/0.999853。无有效收益，否决，不跑31点、不发布。证据 `fbc9cb2c4e-core.json` SHA cbfecffab3c2b03bb900013ae4fe6dfe60254eed92f00426d05160ddaae061d62；SSH job `mla-merge32-fbc9cb2c4e-core-20261028-83578871.json`。
3. 当前固定源码split64/128/256诊断。DSH task `a17e2bce-5237-460a-827c-349b36ee2bcb`，attempt `34d16615-57c3-47a1-b4b7-4a2caf3ee05c`，工具SHA `889a5911c20a3658371031443273a888688d707528bed50b3e5af7279e158c62`。不修改生产factory；直接公开flashattn，ctx8192/16384/32768/65536；原参考、原容差、seed0/42、预分配输出、200预热、7×100轮换/反转顺序计时。split256超过生产并行merge上限128，会走既有serial merge，因此结果测量的是完整公开kernel行为，不能纯归因attention并行度。四个长度×三个split均通过双seed原容差。结果见下表。证据 `889a5911c2-sweep.json` SHA 68a5ed9239140769d87bb3a16cb7e407277ed9b94a98a74b38a504a818868ab3；SSH job `mla-split-sweep-889a5911c2-20261028-a00062f2.json`。

证据目录：`.heteromcp/mla-fused-qk-20261028/`，SSH逐step日志：`.heteromcp/mla-ssh-20261026/`。

|ctx|split64 μs|split128 μs|split256 μs|256相对64加速|
|---:|---:|---:|---:|---:|
|8192|67.576|73.528|94.085|0.718098×|
|16384|120.325|125.379|148.966|0.807166×|
|32768|223.898|228.157|229.082|0.977384×|
|65536|430.323|431.086|385.528|1.116052×|

ctx65536约11.6%信号支持生成单一dispatch候选：仅base_grid1且ctx>=65536选256，其余调度不变。DSH task `d835fd3b-34cb-4f93-b29f-780577db412e`，attempt `fccd4655-9780-49f8-acf8-baa604009705`；候选SHA `ac9cdc1b083ef4301e302c1467076dcec8d784b44bd1a00c138ceec714d5dcbc`。已逐字节核对仅两处替换（split上限128→256、long/basegrid1分支ctx>=65536时target256）；不改kernel/merge/数值算法。独立shape网格确认batch1/head16/ctx65536由64→256，其他已选网格点不变。核心六点双seed通过，最长点434.806→390.159μs、1.114431×，七轮复现扫描收益；其他五点使用相同kernel，微小差异按噪声处理。core JSON SHA `4b637660ef22e1fd7f42147c593300f4f1eff3d22d33a1cf3cbaf343f6254bb05`。完整31/31双seed、原容差均通过；最长点435.584→393.075μs，1.108144×。完整报告SHA `c18ee1896f443432fe37651a81569782b137e75b1bb194695d8aee06fddc12695`；SSH job `mla-split256-ac9cdc1b08-full31-20261028-774a8d59.json`。其余30点dispatch/kernel不变，细小计时差异不作为收益。

## 交付与限制

决定保留长序列split256候选，原样DSH字节交付为MLA.py并保留def723备份；当前优化只具有SSH收益证据，尚未提交真实OJ，不宣称突破97分。当前所有DSH/GPU作业已终态。该调度依据shape和实测，不做设备/环境检测，也没有更改编译器或容差。

## OJ503反馈与否决

用户报告split256候选OJ六点20/35/39/65/114/230μs，总503μs，分数无进步。相对def723最长点211→230μs（+9.0%），总用时+20μs（+4.14%）。无重复测量，不声称统计显著，但这个结果已不支持保留候选。SSH扫描/核心/完整回归约11%收益并未迁移。否决split256，恢复已OJ验证的def723原样字节。回退只防止性能倒退，不算完成97分目标。原始反馈见 `oj503-observation.json`。

## OJ503之后补测的独立分离架构

此前DSH已成功生成但未运行的QK N64/K128 + PV N64/D128分离候选：task `1e39fcf3-a2c2-4e74-9bed-b59d1275ccae`，attempt `9c6d1bf6-a0f1-468f-b84a-5d04d60e0106`，SHA `518cba0add28cf1864e7bf319fb346f0c2eb2649803f67240abd7d42104b080c`。这是独立的多阶段架构实验，非在split256上继续叠加调参。补测以def723为固定基线，所有host helper AST相同，模块全局均literal；scratch64MiB上限，8192以下走旧fused原路径。阶段1以4×128 QK加PE形成FP32 logits；阶段2重复softmax但按4个输出维度tile处理PV；阶段3原merge。额外launch/全局读写的代价必须计入完整调用。未修改DSH文件。核心六点双seed配对运行，未发布。

补测结果：六点双seed原容差均通过；加速比 0.980415/0.980087/0.790239/0.826506/0.835781/0.850465×。中长四点均慢，性能否决，不跑31点、不发布、不请求OJ提交。JSON SHA `b9f7bbe036e2312d62a04bd0269626d1a837a1c05f14465fffec401b7ffd1911`；SSH job `mla-decoupled-518cba0add-core-20261028-7161b073.json`。

当前所有DSH/GPU任务终态；生产SHA仍def723。优化尚未达成97分。当前远端0.1.9开发编译器与OJ0.1.10、分片/物理负载和计时协议不同，至少两次SSH明显加速不能预测OJ收益；同版编译/资源与按kernel计时的证据应先于下一次性能交付。不将未知环境差异具体归咎于全卡，也不再以此猜测调split。
