import math

import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}


@tilelang.jit
def fused_moe_single_kernel(
    d_hidden,
    d_expert,
    n_routed_experts,
    padded_group_sum,
    raw_group_sum,
    group_count,
    num_blocks_m,
    block_token=128,
    block_hidden=128,
    block_expert=32,
    threads=256,
    num_stages=1,
):
    dtype = T.float16
    accum_dtype = T.float32
    scale = 1.44269504

    input_shape = (padded_group_sum, d_hidden)
    gate_shape = (n_routed_experts, d_expert, d_hidden)
    up_shape = (n_routed_experts, d_expert, d_hidden)
    down_shape = (n_routed_experts, d_hidden, d_expert)
    weights_shape = (raw_group_sum,)
    groups_shape = (group_count,)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),  # type: ignore
        gate_w: T.Tensor(gate_shape, dtype),  # type: ignore
        up_w: T.Tensor(up_shape, dtype),  # type: ignore
        down_w: T.Tensor(down_shape, dtype),  # type: ignore
        routed_expert_weights: T.Tensor((raw_group_sum,), dtype),  # type: ignore
        group_sizes: T.Tensor(groups_shape, T.int32),  # type: ignore
        group_offsets: T.Tensor(groups_shape, T.int32),  # type: ignore
        group_padded_offsets: T.Tensor(groups_shape, T.int32),  # type: ignore
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),  # type: ignore
        out: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks_m, T.ceildiv(d_hidden, block_hidden), threads=threads) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_hidden), dtype=dtype)
            fc1_shared = T.alloc_shared((block_expert, block_hidden), dtype=dtype)
            down_shared = T.alloc_shared((block_hidden, block_expert), dtype=dtype)
            gate_activated_shared = T.alloc_shared((block_token, block_expert), dtype=dtype)
            up_activated_shared = T.alloc_shared((block_token, block_expert), dtype=dtype)
            fc1_local = T.alloc_fragment((block_token, block_expert), dtype=accum_dtype)
            up_activated_local = T.alloc_fragment((block_token, block_expert), dtype=dtype)
            out_local = T.alloc_fragment((block_token, block_hidden), dtype=accum_dtype)

            expert_id = group_idx_for_bx[bx]
            padded_start = bx * block_token
            raw_start = group_offsets[expert_id] + padded_start - group_padded_offsets[expert_id]
            group_size = group_sizes[expert_id]
            token_offset = padded_start - group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            T.clear(out_local)
            for expert_tile in T.Pipelined(T.ceildiv(d_expert, block_expert), num_stages=num_stages):
                T.clear(fc1_local)
                for hidden_tile in T.Pipelined(T.ceildiv(d_hidden, block_hidden), num_stages=num_stages):
                    for i, j in T.Parallel(block_token, block_hidden):
                        hidden_col = hidden_tile * block_hidden + j
                        if (i < actual_rows) & (hidden_col < d_hidden):
                            input_shared[i, j] = stacked_expert_tokens[
                                padded_start + i,
                                hidden_col,
                            ]
                        else:
                            input_shared[i, j] = 0
                    for j in T.Parallel(block_hidden):
                        for i in T.Serial(block_expert):
                            expert_col = expert_tile * block_expert + i
                            hidden_col = hidden_tile * block_hidden + j
                            fc1_shared[i, j] = T.if_then_else(
                                (expert_col < d_expert) & (hidden_col < d_hidden),
                                gate_w[expert_id, expert_col, hidden_col],
                                0,
                            )
                    T.gemm(input_shared, fc1_shared, fc1_local, transpose_B=True)

                for i, j in T.Parallel(block_token, block_expert):
                    gate_value = fc1_local[i, j]
                    gate_activated_shared[i, j] = gate_value * (
                        1.0 / (1.0 + T.exp2(-gate_value * scale))
                    )

                T.clear(fc1_local)
                for hidden_tile in T.Pipelined(T.ceildiv(d_hidden, block_hidden), num_stages=num_stages):
                    for i, j in T.Parallel(block_token, block_hidden):
                        hidden_col = hidden_tile * block_hidden + j
                        if (i < actual_rows) & (hidden_col < d_hidden):
                            input_shared[i, j] = stacked_expert_tokens[
                                padded_start + i,
                                hidden_col,
                            ]
                        else:
                            input_shared[i, j] = 0
                    for j in T.Parallel(block_hidden):
                        for i in T.Serial(block_expert):
                            expert_col = expert_tile * block_expert + i
                            hidden_col = hidden_tile * block_hidden + j
                            fc1_shared[i, j] = T.if_then_else(
                                (expert_col < d_expert) & (hidden_col < d_hidden),
                                up_w[expert_id, expert_col, hidden_col],
                                0,
                            )
                    T.gemm(input_shared, fc1_shared, fc1_local, transpose_B=True)

                for i, j in T.Parallel(block_token, block_expert):
                    up_activated_shared[i, j] = (
                        fc1_local[i, j] * gate_activated_shared[i, j]
                    )

                for i, j in T.Parallel(block_token, block_expert):
                    up_activated_local[i, j] = up_activated_shared[i, j]

                for i in T.Parallel(block_hidden):
                    for j in T.Serial(block_expert):
                        hidden_col = by * block_hidden + i
                        expert_col = expert_tile * block_expert + j
                        down_shared[i, j] = T.if_then_else(
                            (hidden_col < d_hidden) & (expert_col < d_expert),
                            down_w[expert_id, hidden_col, expert_col],
                            0,
                        )
                T.gemm(up_activated_local, down_shared, out_local, transpose_B=True)
            for i, j in T.Parallel(block_token, block_hidden):
                if i < actual_rows:
                    out_local[i, j] = (
                        out_local[i, j]
                        * routed_expert_weights[raw_start + i]
                    )
                else:
                    out_local[i, j] = 0
            T.copy(
                out_local,
                out[
                    padded_start : padded_start + block_token,
                    by * block_hidden : (by + 1) * block_hidden,
                ],
            )

    return kernel


def run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
):
    d_hidden = int(stacked_expert_tokens.shape[1])
    d_expert = int(gate_w.shape[1])
    n_routed_experts = int(gate_w.shape[0])
    padded_group_sum = int(stacked_expert_tokens.shape[0])
    raw_group_sum = int(routed_expert_weights.shape[0])
    group_count = int(group_sizes.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])
    key = (d_hidden, d_expert, n_routed_experts, padded_group_sum, raw_group_sum, group_count, num_blocks_m)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = fused_moe_single_kernel(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            padded_group_sum=padded_group_sum,
            raw_group_sum=raw_group_sum,
            group_count=group_count,
            num_blocks_m=num_blocks_m,
        )
        _KERNEL_CACHE[key] = kernel
    kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        out,
    )
