> 历史报告（原时间截面，连接信息已脱敏）。文中的“当前/待测/生产版本”仅指写作当时；归档最新状态见 [README](../README.md) 和 [实验表](../docs/EXPERIMENTS.md)。原机器路径仅为溯源，仓库内文件见 [来源清单](../SOURCE_MANIFEST.json)。

# 延迟在线 softmax 分母归约（2026-10-30）

生产固定基线SHA5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c，OJ96.83、474μs。独立机制，不堆叠QK/B4/warp变化。保持原128threads/N32/KV一次读取/QKPE/PV64/在线max/acc_o缩放/FP16partial/FP32LSE及merge。仅ctx>=8192、headtile16对齐、pv_specialize且split>1。

数学：原分母 L_t=alpha_t*L_prev+sum_jP_tj；新FP32逐score位置D_tj=alpha_t*D_prevj+P_tj，最后L=sum_jD。每个lane保留自己的4个FP32分母累积，随runningmax变化缩放。去掉每个keyblock的一次跨线程sum归约，末尾一次归约；max归约仍每块执行，PV概率仍原FP16转换，分母始终FP32原acc_s、绝不用S_shared替代。FP32加法顺序改变须实际原容差验证，不放宽阈值。新增4FP32/thread的存活可能有资源代价。

真实HeteMCP→DSH task793a4e11-a7da-49e1-9bfa-21fce6722476 / attemptd9fdffb0-179e-438b-b50c-671a35bc5c1f，正式范围成功。产物SHA3b20aa59e4d7afc95e45bad700ddccf46381a9195f6582b98bd0187bfaf0cd34。去掉新factory条件、初始化/最终归约、新分支还原旧分支后整体AST完全等于生产基线；手审FP32表达式、分母归约位置、所有其他macro/no_split/merge/helpers不变。编译/性能待完成。

## 编译与资源

双方2048/8192/65536共6次compile/export成功，SSH作业e01720c0逐exec0，profileJSON SHAac9f49797c79fbca3b2e186b4537b9b73d1533d6248c61e09e376196a29c2931，ZIP91e7274dbf8f2724ceea7dbf9f6880dfbc80b3346e5c37e6cf53413d5f08812c。实际生成代码确认sum AllReduce128从keyloop内移至loop末尾，max仍loop内；非仅源码推断。长ctx动态shared57344→56832，mtreg244→246/streg28/private0，merge不变。额外FP32累积未造成private分配，但不能由此预测性能。六点核心验证进行中。

## 六点核心

6/6、seed0/42 finite原容差通过，7轮×100、200预热、预分配输出交替配对。

|点|基线μs|候选μs|加速比|
|---:|---:|---:|---:|
|1|24.371|24.008|1.015142|
|2|40.963|40.947|1.000375|
|3|72.453|72.000|1.006293|
|4|124.137|122.732|1.011451|
|5|228.751|224.945|1.016923|
|6|435.924|427.540|1.019610|

前两点原路径无改，不算收益。第3/4点部分轮次无收益（第4配对最低0.99825），第5/6所有配对轮次均加速，最长1.96%改善。收益小，不称已实现明显提分。核心JSON SHA96aa09582c49d5f232b6ab2226ce53d7af1c09432efbff85777c1eb1cb034f393，SSH作业d3646bbd逐step0。因存在完整调用收益，继续完整31点原容差回归，不预测OJ。

## 完整31点与交付

31/31、seed0/42全部finite并通过原rtol2e-4/atol1e-4/mismatch_ratio0.01；5轮×20、20预热、双方预分配输出交替配对。31点几何平均加速1.014005×；启用新路径的点加速范围1.003978–1.020850×。短ctx原路径不算有效收益。full31JSON SHA00cff47cf1ba32dfb8bb1db306df7b931aa0a8e53c50da5c549c6498d457d4b32，SSH作业cf4588df所有exec0。所测31点是原批量/上下文集合，不能声称覆盖全部泛化shape。

独立交付MLA.candidate-deferredsum-20261030.py，原字节SHA3b20aa59e4d7afc95e45bad700ddccf46381a9195f6582b98bd0187bfaf0cd34，OJ编译/正确性/性能待测。生产MLA.py保持5e3/96.83/474μs。本轮两个真实DSH任务与GPU作业均终态。该候选是约1%–2%的SSH收益，不称已明显提分或已97分，不能外推OJ。
