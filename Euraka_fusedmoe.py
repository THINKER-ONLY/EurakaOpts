import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *


# EN: TileLang JIT factory for the routed MoE compute kernel.
# CN: 用于 routed MoE 计算 kernel 的 TileLang JIT 工厂函数。
#
# EN: This file only defines the kernel implementation imported by the official
# EN: benchmark through RoutedMoEKernel. The benchmark still owns routing,
# EN: token regrouping, scatter_reduce, and end-to-end timing.
# CN: 本文件只定义官方 benchmark 通过 RoutedMoEKernel 导入的 kernel 实现。
# CN: routing、token regroup、scatter_reduce 和端到端计时仍由 benchmark 负责。
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def moe_forward_tilelang_routed(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    group_count,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
):
    # EN: SiLU(x) = x * sigmoid(x). TileLang exposes exp2, so sigmoid(x)
    # EN: is implemented as 1 / (1 + exp2(-x * log2(e))).
    # CN: SiLU(x) = x * sigmoid(x)。TileLang 提供 exp2，因此这里用
    # CN: 1 / (1 + exp2(-x * log2(e))) 来实现 sigmoid(x)。
    scale = 1.44269504  # log2(e)
    dtype = T.float16

    # EN: Model dimensions captured as compile-time constants for TileLang.
    # CN: 将模型维度捕获为 TileLang 的编译期常量。
    dhidden = d_hidden
    dexpert = d_expert
    n_routed_experts = n_routed_experts

    # EN: Number of block rows in the grouped expert-token space.
    # EN: The extra `group_count` blocks conservatively guard padded expert
    # EN: boundaries/tails used by the current host-side group_idx_for_bx map.
    # CN: grouped expert-token 空间中的 block 行数。
    # CN: 额外加上的 `group_count` 个 block 用于保守覆盖当前 host 侧
    # CN: group_idx_for_bx 映射中的 expert padding 边界和尾块。
    M = math.ceil(group_sum / block_token) + group_count
    accum_dtype = T.float32

    # EN: Tensor layout expected by fusedmoe_benchmark.MoE.forward:
    # EN: - input/up_logits/output are already grouped by expert on the host.
    # EN: - routed_expert_* contain one weight tensor per expert.
    # EN: - group metadata maps block-row bx back to its expert and real rows.
    # CN: fusedmoe_benchmark.MoE.forward 期望的张量布局：
    # CN: - input/up_logits/output 已经在 host 侧按 expert 分组。
    # CN: - routed_expert_* 为每个 expert 存一份权重张量。
    # CN: - group metadata 将 block 行 bx 映射回对应 expert 和真实行区间。
    input_shape = (group_sum, dhidden)
    intermediate_shape = (group_sum, dexpert)
    routed_expert_gate_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_up_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_down_shape = (n_routed_experts, dhidden, dexpert)
    routed_expert_weights_shape = group_sum
    group_sizes_shape = n_routed_experts

    @T.prim_func
    def kernel(
        input: T.Tensor(input_shape, dtype),  # type: ignore
        routed_expert_gate: T.Tensor(routed_expert_gate_shape, dtype),  # type: ignore
        routed_expert_up: T.Tensor(routed_expert_up_shape, dtype),  # type: ignore
        routed_expert_down: T.Tensor(routed_expert_down_shape, dtype),  # type: ignore
        routed_expert_weights: T.Tensor(routed_expert_weights_shape, dtype),  # type: ignore
        group_sizes: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_padded_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_idx_for_bx: T.Tensor((M,), T.int32),  # type: ignore
        up_logits: T.Tensor(intermediate_shape, dtype),  # type: ignore
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        # EN: Step 1: FC1 / gated projection.
        # CN: 第一步：FC1 / gated projection。
        #
        # EN: For each expert block:
        # EN:   gate = input @ W_gate.T
        # EN:   up   = input @ W_up.T
        # EN:   up_logits = SiLU(gate) * up
        # CN: 对每个 expert block：
        # CN:   gate = input @ W_gate.T
        # CN:   up   = input @ W_up.T
        # CN:   up_logits = SiLU(gate) * up
        #
        # EN: This stage keeps two FP32 accumulator tiles live at once, so it is
        # EN: usually more sensitive to register/shared-memory pressure than FC2.
        # CN: 该阶段同时保留两块 FP32 accumulator tile，因此通常比 FC2 更容易
        # CN: 受到寄存器和 shared memory 压力影响。
        with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            routed_expert_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            routed_expert_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)

            T.use_swizzle(10)

            # EN: Translate padded block-row index bx into the real grouped-token
            # EN: row range for its expert. `actual_rows` can be smaller than
            # EN: block_token on expert tails, and can be zero for padded blocks.
            # CN: 将带 padding 的 block 行索引 bx 转换成该 expert 的真实 grouped-token
            # CN: 行区间。expert 尾块上的 `actual_rows` 可能小于 block_token，
            # CN: padding block 上甚至可能为 0。
            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            # EN: K-loop over hidden dimension. The host has already transposed
            # EN: expert weights into [expert, output_dim, input_dim], so GEMM
            # EN: uses transpose_B=True to multiply input tile by weight.T.
            # CN: 沿 hidden 维度做 K-loop。host 已经将 expert 权重转成
            # CN: [expert, output_dim, input_dim]，所以 GEMM 使用 transpose_B=True
            # CN: 来计算 input tile 与 weight.T 的乘法。
            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(
                    input[m_start : m_start + block_token, k * block_dhidden : (k + 1) * block_dhidden],
                    input_shared,
                )
                T.copy(
                    routed_expert_gate[
                        cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
                    ],
                    routed_expert_gate_shared,
                )
                T.gemm(input_shared, routed_expert_gate_shared, gate_logits_local, transpose_B=True)
                T.copy(
                    routed_expert_up[
                        cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
                    ],
                    routed_expert_up_shared,
                )
                T.gemm(input_shared, routed_expert_up_shared, up_logits_local, transpose_B=True)

            # EN: Apply gated activation in registers/fragments before writing
            # EN: the intermediate tensor consumed by FC2. Padding rows are
            # EN: filtered at store time below.
            # CN: 在寄存器/fragment 中完成 gated activation，然后写出供 FC2 使用的
            # CN: 中间张量。padding 行会在下面 store 时被过滤掉。
            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            # EN: Store only valid rows. Expert tails and padded rows must not
            # EN: write past the real grouped-token range.
            # CN: 只写出有效行。expert 尾块和 padding 行不能越过真实 grouped-token
            # CN: 范围写入。
            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]

        # EN: Step 2: FC2 / down projection.
        # CN: 第二步：FC2 / down projection。
        #
        # EN: output = up_logits @ W_down.T
        # CN: output = up_logits @ W_down.T
        #
        # EN: The route weight is multiplied here so the later PyTorch
        # EN: scatter_reduce can simply sum contributions for tokens routed to
        # EN: multiple experts.
        # CN: route weight 在这里相乘，因此后续 PyTorch scatter_reduce 只需要
        # CN: 对路由到多个 expert 的 token contribution 求和即可。
        with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10)

            # EN: Reuse the same expert/block mapping as FC1 so FC2 reads the
            # EN: corresponding intermediate rows and down-projection weights.
            # CN: 复用与 FC1 相同的 expert/block 映射，使 FC2 读取对应的中间行
            # CN: 和 down-projection 权重。
            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

            # EN: K-loop over expert/intermediate dimension.
            # CN: 沿 expert/intermediate 维度做 K-loop。
            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(
                    up_logits[m_start : m_start + block_token, k * block_dexpert : (k + 1) * block_dexpert],
                    up_logits_shared,
                )
                T.copy(
                    routed_expert_down[
                        cur_group_idx, by * block_dhidden : (by + 1) * block_dhidden, k * block_dexpert : (k + 1) * block_dexpert
                    ],
                    routed_expert_down_shared,
                )
                T.gemm(up_logits_shared, routed_expert_down_shared, output_local, transpose_B=True)

            # EN: Write weighted expert output. Host-side scatter_reduce later
            # EN: sums rows with the same original token index.
            # CN: 写出乘过 routing weight 的 expert output。host 侧 scatter_reduce
            # CN: 之后会把原始 token index 相同的行累加起来。
            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]

    return kernel


class RoutedMoEKernel:
    """Fixed public wrapper expected by the official benchmark.

    EN: The benchmark constructs this class on every custom_kernel call and
    EN: then invokes __call__ with grouped token buffers, expert weights, group
    EN: metadata, and preallocated intermediate/output tensors. The method
    EN: signatures should remain compatible with fusedmoe_benchmark.py.

    CN: 官方 benchmark 每次调用 custom_kernel 时都会构造该类，然后用 grouped
    CN: token buffer、expert 权重、group metadata 以及预分配的中间/输出张量
    CN: 调用 __call__。方法签名应保持与 fusedmoe_benchmark.py 兼容。
    """

    def __init__(
        self,
        d_hidden: int,
        d_expert: int,
        n_routed_experts: int,
        group_sum: int,
        group_count: int,
        block_token: int = 128,
        block_dhidden: int = 128,
        block_dexpert: int = 128,
        threads: int = 256,
        num_stages: int = 1,
        backend: str = "tilelang",
    ):
        self.d_hidden = d_hidden
        self.d_expert = d_expert
        self.n_routed_experts = n_routed_experts
        self.group_sum = group_sum
        self.group_count = group_count
        self.block_token = block_token
        self.block_dhidden = block_dhidden
        self.block_dexpert = block_dexpert
        self.threads = threads
        self.num_stages = num_stages
        self.backend = backend

        # EN: Build the JIT-compiled TileLang implementation for this exact
        # EN: shape and tile configuration. A future optimization can cache this
        # EN: object at module scope because the official benchmark repeatedly
        # EN: constructs the wrapper with identical shape parameters.
        # CN: 为当前 shape 和 tile 配置构建 JIT 编译后的 TileLang 实现。由于官方
        # CN: benchmark 会用相同 shape 参数反复构造该 wrapper，后续可考虑在
        # CN: module scope 缓存该对象。
        self.impl = moe_forward_tilelang_routed(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            group_sum=group_sum,
            group_count=group_count,
            block_token=block_token,
            block_dhidden=block_dhidden,
            block_dexpert=block_dexpert,
            threads=threads,
            num_stages=num_stages,
        )

    def __call__(
        self,
        input,
        routed_expert_gate,
        routed_expert_up,
        routed_expert_down,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
        output,
    ):
        # EN: Thin forwarding layer. All tensors are allocated and populated by
        # EN: the official benchmark; this wrapper only launches the TileLang
        # EN: kernel.
        # CN: 这一层只是很薄的转发层。所有张量都由官方 benchmark 分配并填充；
        # CN: 该 wrapper 只负责启动 TileLang kernel。
        return self.impl(
            input,
            routed_expert_gate,
            routed_expert_up,
            routed_expert_down,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            group_idx_for_bx,
            up_logits,
            output,
        )
