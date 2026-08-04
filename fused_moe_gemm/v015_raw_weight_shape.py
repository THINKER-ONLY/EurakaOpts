import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def fused_moe_tilelang(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    group_count,
    raw_group_sum,
    num_blocks,
    block_token=64,
    block_dhidden=128,
    block_dexpert=64,
    threads=256,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    input_shape = (group_sum, d_hidden)
    gate_shape = (n_routed_experts, d_expert, d_hidden)
    down_shape = (n_routed_experts, d_hidden, d_expert)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(gate_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((raw_group_sum,), dtype),
        group_sizes: T.Tensor((group_count,), T.int32),
        group_offsets: T.Tensor((group_count,), T.int32),
        group_padded_offsets: T.Tensor((group_count,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks,), T.int32),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(num_blocks * 2, T.ceildiv(d_hidden, block_dhidden), threads=threads) as (bx, by):
            input_frag = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            projection_weight_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            down_weight_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            hidden_shared = T.alloc_shared((block_token, block_dexpert), dtype=dtype)
            gate_accum = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_accum = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            output_accum = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10)
            logical_bx = bx // 2
            expert_idx = group_idx_for_bx[logical_bx]
            expert_size = group_sizes[expert_idx]
            raw_start = group_offsets[expert_idx]
            padded_block_start = logical_bx * 128 + (bx % 2) * block_token
            padded_group_start = group_padded_offsets[expert_idx]
            row_in_expert = padded_block_start - padded_group_start
            valid_rows = T.max(0, T.min(block_token, expert_size - row_in_expert))

            T.clear(output_accum)
            for expert_col in T.serial(T.ceildiv(d_expert, block_dexpert)):
                T.clear(gate_accum)
                T.clear(up_accum)
                for hidden_col in T.serial(T.ceildiv(d_hidden, block_dhidden)):
                    T.clear(input_frag)
                    for i, j in T.Parallel(block_token, block_dhidden):
                        if i < valid_rows:
                            input_frag[i, j] = stacked_expert_tokens[
                                padded_block_start + i,
                                hidden_col * block_dhidden + j,
                            ]
                    T.copy(
                        gate_w[
                            expert_idx,
                            expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                            hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden,
                        ],
                        projection_weight_shared,
                    )
                    T.gemm(input_frag, projection_weight_shared, gate_accum, transpose_B=True)
                    T.copy(
                        up_w[
                            expert_idx,
                            expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                            hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden,
                        ],
                        projection_weight_shared,
                    )
                    T.gemm(input_frag, projection_weight_shared, up_accum, transpose_B=True)
                for i, j in T.Parallel(block_token, block_dexpert):
                    up_accum[i, j] = up_accum[i, j] * (
                        gate_accum[i, j] * (1.0 / (1.0 + T.exp2(-gate_accum[i, j] * scale)))
                    )
                T.copy(up_accum, hidden_shared)
                T.copy(
                    down_w[
                        expert_idx,
                        by * block_dhidden:(by + 1) * block_dhidden,
                        expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                    ],
                    down_weight_shared,
                )
                T.gemm(hidden_shared, down_weight_shared, output_accum, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < valid_rows:
                    out[padded_block_start + i, by * block_dhidden + j] = (
                        output_accum[i, j] * routed_expert_weights[raw_start + row_in_expert + i]
                    )
                else:
                    out[padded_block_start + i, by * block_dhidden + j] = 0.0

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
    kernel = fused_moe_tilelang(
        d_hidden=stacked_expert_tokens.shape[1],
        d_expert=gate_w.shape[1],
        n_routed_experts=gate_w.shape[0],
        group_sum=stacked_expert_tokens.shape[0],
        group_count=group_sizes.shape[0],
        raw_group_sum=routed_expert_weights.shape[0],
        num_blocks=group_idx_for_bx.shape[0],
    )
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
