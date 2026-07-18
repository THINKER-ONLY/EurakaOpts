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
    metadata_block_m = 128
    routing_block_m = (
        32
        if hidden == 7168
        and intermediate == 2048
        and total_valid_tokens < num_experts * 32
        else 64
        if hidden == 7168
        and intermediate == 2048
        and total_valid_tokens < num_experts * 64
        else 128
    )
    routing_subblocks = metadata_block_m // routing_block_m
    compute_blocks_m = num_blocks_m * routing_subblocks
    block_k = 64
    fc1_block_k = 64
    fc1_block_n = 64 if routing_block_m == 32 else 128
    fc2_block_n = (
        256
        if (hidden == 7168 and intermediate == 2048)
        or (hidden == 2048 and intermediate == 8192)
        else 128
    )
    threads = 256
    num_stages = 1

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(compute_blocks_m, T.ceildiv(intermediate, fc1_block_n), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((routing_block_m, fc1_block_k), dtype=dtype)
            gate_shared = T.alloc_shared((fc1_block_n, fc1_block_k), dtype=dtype)
            up_shared = T.alloc_shared((fc1_block_n, fc1_block_k), dtype=dtype)
            gate_local = T.alloc_fragment((routing_block_m, fc1_block_n), dtype=accum_dtype)
            up_local = T.alloc_fragment((routing_block_m, fc1_block_n), dtype=accum_dtype)

            T.use_swizzle(10)

            metadata_bx = bx // routing_subblocks
            subblock = bx % routing_subblocks
            expert_id = group_idx_for_bx[metadata_bx]
            block_start = metadata_bx * metadata_block_m + subblock * routing_block_m
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(
                0,
                T.min(
                    routing_block_m,
                    group_size - (block_start - padded_start),
                ),
            )

            if actual_rows > 0:
                T.clear(gate_local)
                T.clear(up_local)

                for k in T.Pipelined(
                    T.ceildiv(hidden, fc1_block_k),
                    num_stages=num_stages,
                ):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + routing_block_m,
                            k * fc1_block_k : (k + 1) * fc1_block_k,
                        ],
                        input_shared,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * fc1_block_n : (by + 1) * fc1_block_n,
                            k * fc1_block_k : (k + 1) * fc1_block_k,
                        ],
                        gate_shared,
                    )
                    T.gemm(input_shared, gate_shared, gate_local, transpose_B=True)
                    T.copy(
                        up_w[
                            expert_id,
                            by * fc1_block_n : (by + 1) * fc1_block_n,
                            k * fc1_block_k : (k + 1) * fc1_block_k,
                        ],
                        up_shared,
                    )
                    T.gemm(input_shared, up_shared, up_local, transpose_B=True)

                for i, j in T.Parallel(routing_block_m, fc1_block_n):
                    gate_local[i, j] = gate_local[i, j] * (
                        1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale))
                    )
                    up_local[i, j] = up_local[i, j] * gate_local[i, j]

                for i, j in T.Parallel(routing_block_m, fc1_block_n):
                    if i < actual_rows:
                        up_logits[block_start + i, by * fc1_block_n + j] = up_local[i, j]

        with T.Kernel(compute_blocks_m, T.ceildiv(hidden, fc2_block_n), threads=threads) as (bx, by):
            up_shared = T.alloc_fragment((routing_block_m, block_k), dtype=dtype)
            down_shared = T.alloc_shared((fc2_block_n, block_k), dtype=dtype)
            out_local = T.alloc_fragment((routing_block_m, fc2_block_n), dtype=accum_dtype)

            T.use_swizzle(10)

            metadata_bx = bx // routing_subblocks
            subblock = bx % routing_subblocks
            expert_id = group_idx_for_bx[metadata_bx]
            block_start = metadata_bx * metadata_block_m + subblock * routing_block_m
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(routing_block_m, group_size - token_offset),
            )

            if actual_rows > 0:
                T.clear(out_local)

                for k in T.Pipelined(
                    T.ceildiv(intermediate, block_k),
                    num_stages=num_stages,
                ):
                    T.copy(
                        up_logits[
                            block_start : block_start + routing_block_m,
                            k * block_k : (k + 1) * block_k,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * fc2_block_n : (by + 1) * fc2_block_n,
                            k * block_k : (k + 1) * block_k,
                        ],
                        down_shared,
                    )
                    T.gemm(up_shared, down_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(routing_block_m, fc2_block_n):
                    if i < actual_rows:
                        out[block_start + i, by * fc2_block_n + j] = (
                            out_local[i, j]
                            * routed_expert_weights[raw_start + token_offset + i]
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
            (int(stacked_expert_tokens.shape[0]), int(intermediate)),
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

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
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
