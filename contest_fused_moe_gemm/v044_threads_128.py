import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    block_token = 128
    block_dhidden = 128
    block_dexpert = 128
    threads = 128
    num_stages = 1

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),  # type: ignore
        gate_w: T.Tensor(gate_shape, dtype),  # type: ignore
        up_w: T.Tensor(up_shape, dtype),  # type: ignore
        down_w: T.Tensor(down_shape, dtype),  # type: ignore
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),  # type: ignore
        group_sizes: T.Tensor((num_experts,), T.int32),  # type: ignore
        group_offsets: T.Tensor((num_experts + 1,), T.int32),  # type: ignore
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),  # type: ignore
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),  # type: ignore
        up_logits: T.Tensor(intermediate_shape, dtype),  # type: ignore
        out: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        with T.Kernel(
            num_blocks_m,
            T.ceildiv(intermediate, block_dexpert),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_fragment(
                (block_token, block_dhidden),
                dtype=dtype,
            )
            gate_shared = T.alloc_shared(
                (block_dexpert, block_dhidden),
                dtype=dtype,
            )
            up_shared = T.alloc_shared(
                (block_dexpert, block_dhidden),
                dtype=dtype,
            )
            gate_local = T.alloc_fragment(
                (block_token, block_dexpert),
                dtype=accum_dtype,
            )
            up_local = T.alloc_fragment(
                (block_token, block_dexpert),
                dtype=accum_dtype,
            )

            T.use_swizzle(10)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(
                0,
                T.min(
                    block_token,
                    group_size - (block_start - padded_start),
                ),
            )

            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(
                T.ceildiv(hidden, block_dhidden),
                num_stages=num_stages,
            ):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + block_token,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    input_shared,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * block_dexpert : (by + 1) * block_dexpert,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    gate_shared,
                )
                T.gemm(
                    input_shared,
                    gate_shared,
                    gate_local,
                    transpose_B=True,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * block_dexpert : (by + 1) * block_dexpert,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    up_shared,
                )
                T.gemm(
                    input_shared,
                    up_shared,
                    up_local,
                    transpose_B=True,
                )

            for i, j in T.Parallel(block_token, block_dexpert):
                gate_local[i, j] = gate_local[i, j] * (
                    1.0
                    / (
                        1.0
                        + T.exp(-gate_local[i, j])
                    )
                )
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[
                        block_start + i,
                        by * block_dexpert + j,
                    ] = up_local[i, j]

        with T.Kernel(
            num_blocks_m,
            T.ceildiv(hidden, block_dhidden),
            threads=threads,
        ) as (bx, by):
            up_shared = T.alloc_fragment(
                (block_token, block_dexpert),
                dtype=dtype,
            )
            down_shared = T.alloc_shared(
                (block_dhidden, block_dexpert),
                dtype=dtype,
            )
            out_local = T.alloc_fragment(
                (block_token, block_dhidden),
                dtype=accum_dtype,
            )

            T.use_swizzle(10)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(
                    block_token,
                    group_size - token_offset,
                ),
            )

            T.clear(out_local)

            for k in T.Pipelined(
                T.ceildiv(intermediate, block_dexpert),
                num_stages=num_stages,
            ):
                T.copy(
                    up_logits[
                        block_start : block_start + block_token,
                        k * block_dexpert : (k + 1) * block_dexpert,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * block_dhidden : (by + 1) * block_dhidden,
                        k * block_dexpert : (k + 1) * block_dexpert,
                    ],
                    down_shared,
                )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                )

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    out[block_start + i, by * block_dhidden + j] = (
                        out_local[i, j]
                        * routed_expert_weights[
                            raw_start + token_offset + i
                        ]
                    )

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _moe_forward_kernel(*key)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _get_workspace(stacked_expert_tokens, intermediate):
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
        str(stacked_expert_tokens.dtype),
    )
    up_logits = _WORKSPACE_CACHE.get(key)
    if up_logits is None:
        up_logits = torch.empty(
            (
                int(stacked_expert_tokens.shape[0]),
                int(intermediate),
            ),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = up_logits
    return up_logits


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
    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])

    up_logits = _get_workspace(
        stacked_expert_tokens,
        intermediate,
    )
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
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
        up_logits,
        out,
    )
