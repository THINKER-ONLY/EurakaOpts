import math

import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
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
    block_expert=128,
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
            input_shared = T.alloc_fragment((block_token, block_hidden), dtype=dtype)
            gate_shared = T.alloc_shared((block_expert, block_hidden), dtype=dtype)
            up_shared = T.alloc_shared((block_expert, block_hidden), dtype=dtype)
            down_shared = T.alloc_shared((block_hidden, block_expert), dtype=dtype)
            gate_local = T.alloc_fragment((block_token, block_expert), dtype=accum_dtype)
            up_local = T.alloc_fragment((block_token, block_expert), dtype=accum_dtype)
            out_local = T.alloc_fragment((block_token, block_hidden), dtype=accum_dtype)

            T.use_swizzle(10)
            expert_id = group_idx_for_bx[bx]
            padded_start = bx * block_token
            raw_start = group_offsets[expert_id] + padded_start - group_padded_offsets[expert_id]
            group_size = group_sizes[expert_id]
            token_offset = padded_start - group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            T.clear(out_local)
            for expert_tile in T.Pipelined(T.ceildiv(d_expert, block_expert), num_stages=num_stages):
                T.clear(gate_local)
                T.clear(up_local)
                for hidden_tile in T.Pipelined(T.ceildiv(d_hidden, block_hidden), num_stages=num_stages):
                    for i, j in T.Parallel(block_token, block_hidden):
                        if i < actual_rows:
                            input_shared[i, j] = stacked_expert_tokens[
                                padded_start + i,
                                hidden_tile * block_hidden + j,
                            ]
                        else:
                            input_shared[i, j] = 0
                    T.copy(
                        gate_w[
                            expert_id,
                            expert_tile * block_expert : (expert_tile + 1) * block_expert,
                            hidden_tile * block_hidden : (hidden_tile + 1) * block_hidden,
                        ],
                        gate_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            expert_tile * block_expert : (expert_tile + 1) * block_expert,
                            hidden_tile * block_hidden : (hidden_tile + 1) * block_hidden,
                        ],
                        up_shared,
                    )
                    T.gemm(input_shared, gate_shared, gate_local, transpose_B=True)
                    T.gemm(input_shared, up_shared, up_local, transpose_B=True)

                for i, j in T.Parallel(block_token, block_expert):
                    gate_value = gate_local[i, j]
                    gate_local[i, j] = gate_value * (1.0 / (1.0 + T.exp2(-gate_value * scale)))
                    up_local[i, j] = up_local[i, j] * gate_local[i, j]

                T.copy(
                    down_w[
                        expert_id,
                        by * block_hidden : (by + 1) * block_hidden,
                        expert_tile * block_expert : (expert_tile + 1) * block_expert,
                    ],
                    down_shared,
                )
                T.gemm(up_local, down_shared, out_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_hidden):
                if i < actual_rows:
                    out[padded_start + i, by * block_hidden + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + i]
                    )
                else:
                    out[padded_start + i, by * block_hidden + j] = 0

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
