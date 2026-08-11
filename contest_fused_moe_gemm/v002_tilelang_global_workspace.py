import math
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def moe_forward_tilelang_routed(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    padded_group_sum,
    group_count,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert
    n_routed_experts = n_routed_experts

    M = math.ceil(group_sum / block_token) + group_count
    accum_dtype = T.float32

    input_shape = (padded_group_sum, dhidden)
    intermediate_shape = (padded_group_sum, dexpert)
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
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        up_logits = T.alloc_global(intermediate_shape, dtype=dtype)

        # Step 1: Compute gate and up logits
        with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            routed_expert_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            routed_expert_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)

            T.use_swizzle(10)

            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(gate_logits_local)
            T.clear(up_logits_local)

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

            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]

        # Step 2: Compute down logits
        with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10)

            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

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

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]

    return kernel


_KERNEL_CACHE = {}


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
    group_sum = int(routed_expert_weights.shape[0])
    group_count = int(group_sizes.shape[0])
    key = (
        d_hidden,
        d_expert,
        n_routed_experts,
        group_sum,
        padded_group_sum,
        group_count,
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = moe_forward_tilelang_routed(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            group_sum=group_sum,
            padded_group_sum=padded_group_sum,
            group_count=group_count,
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
