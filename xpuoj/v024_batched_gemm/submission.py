import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}
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
                else:
                    packed_input[expert_id, i, bn * block_n + j] = T.float16(0)

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
):
    block_n = 256
    threads = 256
    packed_shape = (num_experts, expert_block_m, hidden)

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
            num_experts,
            T.ceildiv(hidden, block_n),
            threads=threads,
        ) as (expert_id, bn):
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    out[padded_start + i, bn * block_n + j] = (
                        packed_output[expert_id, i, bn * block_n + j]
                        * routed_expert_weights[raw_start + i]
                    )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
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
        activation: T.Tensor(shape, T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(intermediate, block_n),
            threads=threads,
        ) as (expert_id, bn):
            for i, j in T.Parallel(expert_block_m, block_n):
                gate = gate_output[expert_id, i, bn * block_n + j]
                activation[expert_id, i, bn * block_n + j] = (
                    gate
                    * T.sigmoid(gate)
                    * up_output[expert_id, i, bn * block_n + j]
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
        kernels = (
            _pack_experts_kernel(
                hidden,
                num_experts,
                total_padded_tokens,
                expert_block_m,
            ),
            _unpack_experts_kernel(
                hidden,
                num_experts,
                total_padded_tokens,
                total_valid_tokens,
                expert_block_m,
            ),
            _swiglu_kernel(
                intermediate,
                num_experts,
                expert_block_m,
            ),
        )
        _KERNEL_CACHE[key] = kernels
    return kernels


def _get_workspace(
    stacked_expert_tokens,
    num_experts,
    expert_block_m,
):
    hidden = int(stacked_expert_tokens.shape[1])
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(num_experts),
        hidden,
        int(expert_block_m),
        str(stacked_expert_tokens.dtype),
    )
    packed_input = _WORKSPACE_CACHE.get(key)
    if packed_input is None:
        packed_input = torch.empty(
            (num_experts, expert_block_m, hidden),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = packed_input
    return packed_input


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

    pack, unpack, swiglu = _get_kernels(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        expert_block_m,
    )
    packed_input = _get_workspace(
        stacked_expert_tokens,
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
    swiglu(gate_output, up_output, gate_output)
    packed_output = gate_output @ down_w.transpose(1, 2)
    unpack(
        packed_output,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        out,
    )
