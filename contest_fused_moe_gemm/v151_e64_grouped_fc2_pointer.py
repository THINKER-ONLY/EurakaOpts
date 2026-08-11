import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE_KEY = None
_WORKSPACE_CACHE_VALUE = None
_WEIGHT_CACHE = {}
_EXPERT_BLOCK_M = 192


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _pack_experts_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    expert_block_m,
):
    block_n = 256
    threads = 256
    packed_shape = (num_experts, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor((total_padded_tokens, hidden), T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        packed_input: T.Tensor(packed_shape, T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(hidden, block_n),
            threads=threads,
        ) as (expert_id, bn):
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    packed_input[expert_id, i, bn * block_n + j] = (
                        stacked_expert_tokens[padded_start + i, bn * block_n + j]
                    )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _unpack_experts_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    expert_block_m,
    expert_start,
    expert_chunk,
):
    block_n = 256
    threads = 256
    packed_shape = (expert_chunk, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        packed_output: T.Tensor(packed_shape, T.float16),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), T.float16),
    ):
        with T.Kernel(
            expert_chunk,
            T.ceildiv(hidden, block_n),
            threads=threads,
        ) as (local_expert_id, bn):
            expert_id = expert_start + local_expert_id
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    out[padded_start + i, bn * block_n + j] = (
                        packed_output[local_expert_id, i, bn * block_n + j]
                        * routed_expert_weights[raw_start + i]
                    )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _grouped_e64_fc2_kernel(hidden, total_padded_tokens, total_valid_tokens):
    block_token = 64
    block_dexpert = 64
    block_dhidden = 128
    threads = 256
    num_stages = 2
    num_experts = 64
    expert_block_m = 192
    max_m_blocks = (num_experts * expert_block_m + block_token - 1) // block_token

    @T.prim_func
    def kernel(
        activation: T.Tensor((num_experts, expert_block_m, 2048), T.float16),
        down_w: T.Tensor((num_experts, hidden, 2048), T.float16),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), T.float16),
    ):
        with T.Kernel(max_m_blocks, T.ceildiv(hidden, block_dhidden), threads=threads) as (bx, by):
            m_start_padded = bx * block_token
            cur_expert = 0
            for i in T.serial(num_experts):
                if m_start_padded >= group_padded_offsets[i]:
                    cur_expert = i
            row_base = m_start_padded - group_padded_offsets[cur_expert]
            raw_base = group_offsets[cur_expert] + row_base
            actual_rows = T.max(0, T.min(block_token, group_sizes[cur_expert] - row_base))
            input_shared = T.alloc_shared((block_token, block_dexpert), T.float16)
            weight_shared = T.alloc_shared((block_dhidden, block_dexpert), T.float16)
            output_local = T.alloc_fragment((block_token, block_dhidden), T.float32)
            T.clear(output_local)
            for k in T.Pipelined(T.ceildiv(2048, block_dexpert), num_stages=num_stages):
                T.copy(activation[cur_expert, row_base:row_base + block_token, k * block_dexpert:(k + 1) * block_dexpert], input_shared)
                T.copy(down_w[cur_expert, by * block_dhidden:(by + 1) * block_dhidden, k * block_dexpert:(k + 1) * block_dexpert], weight_shared)
                T.gemm(input_shared, weight_shared, output_local, transpose_B=True)
            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    out[group_padded_offsets[cur_expert] + row_base + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[raw_base + i]

    return kernel

def _swiglu_kernel(
    intermediate,
    num_experts,
    expert_block_m,
):
    block_n = 256
    threads = 256
    shape = (num_experts, expert_block_m, intermediate)

    @T.prim_func
    def kernel(
        gate_output: T.Tensor(shape, T.float16),
        up_output: T.Tensor(shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        activation: T.Tensor(shape, T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(intermediate, block_n),
            threads=threads,
        ) as (expert_id, bn):
            group_size = group_sizes[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    gate = gate_output[expert_id, i, bn * block_n + j]
                    activation[expert_id, i, bn * block_n + j] = (
                        gate
                        * T.sigmoid(gate)
                        * up_output[expert_id, i, bn * block_n + j]
                    )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _combine_fc1_weights_kernel(
    hidden,
    intermediate,
    num_experts,
):
    block_i = 64
    block_h = 256
    threads = 256

    @T.prim_func
    def kernel(
        gate_w: T.Tensor((num_experts, intermediate, hidden), T.float16),
        up_w: T.Tensor((num_experts, intermediate, hidden), T.float16),
        combined_w: T.Tensor((num_experts, intermediate * 2, hidden), T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(intermediate * 2, block_i),
            T.ceildiv(hidden, block_h),
            threads=threads,
        ) as (expert_id, bi, bh):
            for i, j in T.Parallel(block_i, block_h):
                row = bi * block_i + i
                col = bh * block_h + j
                if row < intermediate:
                    combined_w[expert_id, row, col] = gate_w[expert_id, row, col]
                else:
                    combined_w[expert_id, row, col] = up_w[
                        expert_id, row - intermediate, col
                    ]

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _combined_swiglu_kernel(
    intermediate,
    num_experts,
    expert_block_m,
):
    block_n = 256
    threads = 256
    combined_shape = (num_experts, expert_block_m, intermediate * 2)
    activation_shape = (num_experts, expert_block_m, intermediate)

    @T.prim_func
    def kernel(
        gate_up_output: T.Tensor(combined_shape, T.float16),
        activation: T.Tensor(activation_shape, T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(intermediate, block_n),
            threads=threads,
        ) as (expert_id, bn):
            for i, j in T.Parallel(expert_block_m, block_n):
                gate = gate_up_output[expert_id, i, bn * block_n + j]
                activation[expert_id, i, bn * block_n + j] = (
                    gate
                    * T.sigmoid(gate)
                    * gate_up_output[
                        expert_id,
                        i,
                        intermediate + bn * block_n + j,
                    ]
                )

    return kernel


def _get_kernels(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    expert_block_m,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(expert_block_m),
    )
    kernels = _KERNEL_CACHE.get(key)
    if kernels is None:
        expert_chunk = 32 if num_experts == 32 else 16
        fuse_fc1 = False
        kernels = (
            _pack_experts_kernel(
                hidden,
                num_experts,
                total_padded_tokens,
                expert_block_m,
            ),
            _swiglu_kernel(
                intermediate,
                num_experts,
                expert_block_m,
            ),
            (
                _combine_fc1_weights_kernel(hidden, intermediate, num_experts)
                if fuse_fc1
                else None
            ),
            (
                _combined_swiglu_kernel(
                    intermediate,
                    num_experts,
                    expert_block_m,
                )
                if fuse_fc1
                else None
            ),
            (
                _grouped_e64_fc2_kernel(hidden, total_padded_tokens, total_valid_tokens)
                if num_experts == 64
                else None
            ),
            tuple(
                _unpack_experts_kernel(
                    hidden,
                    num_experts,
                    total_padded_tokens,
                    total_valid_tokens,
                    expert_block_m,
                    expert_start,
                    min(expert_chunk, num_experts - expert_start),
                )
                for expert_start in range(0, num_experts, expert_chunk)
            ),
        )
        _KERNEL_CACHE[key] = kernels
    return kernels


def _get_workspace(
    stacked_expert_tokens,
    intermediate,
    num_experts,
    expert_block_m,
):
    global _WORKSPACE_CACHE_KEY, _WORKSPACE_CACHE_VALUE
    hidden = int(stacked_expert_tokens.shape[1])
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(num_experts),
        hidden,
        int(intermediate),
        int(expert_block_m),
        str(stacked_expert_tokens.dtype),
    )
    if _WORKSPACE_CACHE_KEY != key:
        _WORKSPACE_CACHE_KEY = None
        _WORKSPACE_CACHE_VALUE = None
        packed_input = torch.empty(
            (num_experts, expert_block_m, hidden),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        activation = (
            torch.empty(
                (num_experts, expert_block_m, intermediate),
                device=stacked_expert_tokens.device,
                dtype=stacked_expert_tokens.dtype,
            )
            if False
            else None
        )
        _WORKSPACE_CACHE_KEY = key
        _WORKSPACE_CACHE_VALUE = (packed_input, activation)
    return _WORKSPACE_CACHE_VALUE


def _get_combined_fc1_weight(gate_w, up_w, combine_weights):
    key = (
        int(gate_w.device.index or 0),
        int(gate_w.shape[0]),
        int(gate_w.shape[1]),
        int(gate_w.shape[2]),
        str(gate_w.dtype),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is None or cached[0] is not gate_w or cached[1] is not up_w:
        combined_w = torch.empty(
            (gate_w.shape[0], gate_w.shape[1] * 2, gate_w.shape[2]),
            device=gate_w.device,
            dtype=gate_w.dtype,
        )
        combine_weights(gate_w, up_w, combined_w)
        cached = (gate_w, up_w, combined_w)
        _WEIGHT_CACHE[key] = cached
    return cached[2]


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

    expert_block_m = _EXPERT_BLOCK_M

    (
        pack, swiglu, combine_weights, combined_swiglu, grouped_e64_fc2, unpack_chunks
    ) = _get_kernels(        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        expert_block_m,
    )
    packed_input, activation = _get_workspace(
        stacked_expert_tokens,
        intermediate,
        num_experts,
        expert_block_m,
    )

    pack(
        stacked_expert_tokens,
        group_sizes,
        group_padded_offsets,
        packed_input,
    )
    gate_output = packed_input @ gate_w.transpose(1, 2)
    up_output = packed_input @ up_w.transpose(1, 2)
    swiglu(gate_output, up_output, group_sizes, gate_output)
    del up_output
    activation = gate_output
    expert_chunk = 32 if num_experts == 32 else 16
    if grouped_e64_fc2 is not None:
        grouped_e64_fc2(activation, down_w, routed_expert_weights, group_sizes, group_offsets, group_padded_offsets, out)
        return out
    for chunk_id, expert_start in enumerate(range(0, num_experts, expert_chunk)):
        expert_end = min(expert_start + expert_chunk, num_experts)
        packed_output = (
            activation[expert_start:expert_end]
            @ down_w[expert_start:expert_end].transpose(1, 2)
        )
        unpack_chunks[chunk_id](
            packed_output,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            out,
        )
        del packed_output
