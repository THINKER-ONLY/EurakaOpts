import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def fused_moe(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    group_count,
    num_blocks,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
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
        routed_expert_weights: T.Tensor((group_sum,), dtype),
        group_sizes: T.Tensor((group_count,), T.int32),
        group_offsets: T.Tensor((group_count,), T.int32),
        group_padded_offsets: T.Tensor((group_count,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks,), T.int32),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(num_blocks, T.ceildiv(d_hidden, block_dhidden), threads=threads) as (bx, by):
            input_local = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            weight_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            hidden_shared = T.alloc_shared((block_token, block_dexpert), dtype=dtype)
            gate_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            out_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10)
            padded_start = bx * block_token
            expert_idx = group_idx_for_bx[bx]
            valid_count = group_sizes[expert_idx]
            raw_start = group_offsets[expert_idx]
            expert_padded_start = group_padded_offsets[expert_idx]
            row_in_expert = padded_start - expert_padded_start
            actual_rows = T.max(0, T.min(block_token, valid_count - row_in_expert))
            compact_start = raw_start + row_in_expert

            T.clear(out_local)
            for expert_col in T.serial(T.ceildiv(d_expert, block_dexpert)):
                T.clear(gate_local)
                T.clear(up_local)
                for hidden_col in T.Pipelined(T.ceildiv(d_hidden, block_dhidden), num_stages=num_stages):
                    T.copy(
                        stacked_expert_tokens[
                            compact_start:compact_start + block_token,
                            hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden,
                        ],
                        input_local,
                    )
                    T.copy(
                        gate_w[
                            expert_idx,
                            expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                            hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden,
                        ],
                        weight_shared,
                    )
                    T.gemm(input_local, weight_shared, gate_local, transpose_B=True)
                    T.copy(
                        up_w[
                            expert_idx,
                            expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                            hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden,
                        ],
                        weight_shared,
                    )
                    T.gemm(input_local, weight_shared, up_local, transpose_B=True)

                for i, j in T.Parallel(block_token, block_dexpert):
                    up_local[i, j] = up_local[i, j] * (
                        gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                    )
                T.copy(up_local, hidden_shared)
                T.copy(
                    down_w[
                        expert_idx,
                        by * block_dhidden:(by + 1) * block_dhidden,
                        expert_col * block_dexpert:(expert_col + 1) * block_dexpert,
                    ],
                    weight_shared,
                )
                T.gemm(hidden_shared, weight_shared, out_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    out[compact_start + i, by * block_dhidden + j] = out_local[i, j] * routed_expert_weights[compact_start + i]

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
    kernel = fused_moe(
        d_hidden=stacked_expert_tokens.shape[1],
        d_expert=gate_w.shape[1],
        n_routed_experts=gate_w.shape[0],
        group_sum=stacked_expert_tokens.shape[0],
        group_count=group_sizes.shape[0],
        num_blocks=group_idx_for_bx.shape[0],
    )
    kernel(
        stacked_expert_tokens, gate_w, up_w, down_w, routed_expert_weights,
        group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx, out,
    )
