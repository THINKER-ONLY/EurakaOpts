import tilelang
import tilelang.language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def moe_forward_tilelang_routed(
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
        input: T.Tensor(input_shape, dtype),
        routed_expert_gate: T.Tensor(gate_shape, dtype),
        routed_expert_up: T.Tensor(gate_shape, dtype),
        routed_expert_down: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((group_sum,), dtype),
        group_sizes: T.Tensor((group_count,), T.int32),
        group_offsets: T.Tensor((group_count,), T.int32),
        group_padded_offsets: T.Tensor((group_count,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks,), T.int32),
        output: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(num_blocks, T.ceildiv(d_hidden, block_dhidden), threads=threads) as (bx, by):
            input_local = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            gate_local = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            up_local = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10)
            m_start_padded = bx * block_token
            cur_group_idx = group_idx_for_bx[bx]
            cur_group_size = group_sizes[cur_group_idx]
            raw_group_start = group_offsets[cur_group_idx]
            padded_group_start = group_padded_offsets[cur_group_idx]
            group_row = m_start_padded - padded_group_start
            actual_rows = T.max(0, T.min(block_token, cur_group_size - group_row))
            m_start = raw_group_start + group_row

            T.clear(output_local)
            for expert_col in T.serial(T.ceildiv(d_expert, block_dexpert)):
                T.clear(gate_local)
                T.clear(up_local)
                for hidden_col in T.Pipelined(T.ceildiv(d_hidden, block_dhidden), num_stages=num_stages):
                    T.copy(input[m_start:m_start + block_token, hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden], input_local)
                    T.copy(routed_expert_gate[cur_group_idx, expert_col * block_dexpert:(expert_col + 1) * block_dexpert, hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden], gate_shared)
                    T.gemm(input_local, gate_shared, gate_local, transpose_B=True)
                    T.copy(routed_expert_up[cur_group_idx, expert_col * block_dexpert:(expert_col + 1) * block_dexpert, hidden_col * block_dhidden:(hidden_col + 1) * block_dhidden], up_shared)
                    T.gemm(input_local, up_shared, up_local, transpose_B=True)
                for i, j in T.Parallel(block_token, block_dexpert):
                    gate_local[i, j] = gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                    up_local[i, j] = up_local[i, j] * gate_local[i, j]
                T.copy(routed_expert_down[cur_group_idx, by * block_dhidden:(by + 1) * block_dhidden, expert_col * block_dexpert:(expert_col + 1) * block_dexpert], down_shared)
                T.gemm(up_local, down_shared, output_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]
                else:
                    output[m_start + i, by * block_dhidden + j] = 0.0

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
    impl = moe_forward_tilelang_routed(
        d_hidden=stacked_expert_tokens.shape[1],
        d_expert=gate_w.shape[1],
        n_routed_experts=gate_w.shape[0],
        group_sum=stacked_expert_tokens.shape[0],
        group_count=group_sizes.shape[0],
        num_blocks=group_idx_for_bx.shape[0],
    )
    impl(stacked_expert_tokens, gate_w, up_w, down_w, routed_expert_weights,
         group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx, out)
