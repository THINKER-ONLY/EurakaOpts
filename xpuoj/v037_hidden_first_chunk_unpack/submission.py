import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}
_WEIGHT_CACHE = {}
_DOWN_WEIGHT_CACHE = {}
_DOWN_OUTPUT_CACHE = {}
_LAST_COMPLETED_OUTPUT = None
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
    expert_start,
    expert_chunk,
):
    block_n = 128
    threads = 1024
    packed_shape = (expert_chunk, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        packed_output: T.Tensor(packed_shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
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
            padded_start = group_padded_offsets[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    out[padded_start + i, bn * block_n + j] = packed_output[
                        local_expert_id, i, bn * block_n + j
                    ]

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _scale_down_chunk_kernel(
    hidden,
    num_experts,
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
    ):
        with T.Kernel(
            expert_chunk,
            T.ceildiv(hidden, block_n),
            threads=threads,
        ) as (local_expert_id, bn):
            expert_id = expert_start + local_expert_id
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    packed_output[local_expert_id, i, bn * block_n + j] = (
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
def _unpack_two_chunks_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    expert_block_m,
):
    block_n = 128
    threads = 1024
    packed_shape = (16, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        packed_output0: T.Tensor(packed_shape, T.float16),
        packed_output1: T.Tensor(packed_shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(hidden, block_n),
            16,
            2,
            threads=threads,
        ) as (bn, local_expert_id, chunk_id):
            if chunk_id == 0:
                group_size0 = group_sizes[local_expert_id]
                padded_start0 = group_padded_offsets[local_expert_id]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = packed_output0[
                            local_expert_id, i, bn * block_n + j
                        ]
            else:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size1:
                        out[padded_start1 + i, bn * block_n + j] = packed_output1[
                            local_expert_id, i, bn * block_n + j
                        ]

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _unpack_four_chunks_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    expert_block_m,
):
    block_n = 256
    threads = 1024
    packed_shape = (16, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        packed_output0: T.Tensor(packed_shape, T.float16),
        packed_output1: T.Tensor(packed_shape, T.float16),
        packed_output2: T.Tensor(packed_shape, T.float16),
        packed_output3: T.Tensor(packed_shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(hidden, block_n),
            16,
            4,
            threads=threads,
        ) as (bn, local_expert_id, chunk_id):
            if chunk_id == 0:
                group_size0 = group_sizes[local_expert_id]
                padded_start0 = group_padded_offsets[local_expert_id]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = packed_output0[
                            local_expert_id, i, bn * block_n + j
                        ]
            elif chunk_id == 1:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size1:
                        out[padded_start1 + i, bn * block_n + j] = packed_output1[
                            local_expert_id, i, bn * block_n + j
                        ]
            elif chunk_id == 2:
                expert_id2 = 32 + local_expert_id
                group_size2 = group_sizes[expert_id2]
                padded_start2 = group_padded_offsets[expert_id2]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size2:
                        out[padded_start2 + i, bn * block_n + j] = packed_output2[
                            local_expert_id, i, bn * block_n + j
                        ]
            else:
                expert_id3 = 48 + local_expert_id
                group_size3 = group_sizes[expert_id3]
                padded_start3 = group_padded_offsets[expert_id3]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size3:
                        out[padded_start3 + i, bn * block_n + j] = packed_output3[
                            local_expert_id, i, bn * block_n + j
                        ]

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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _transpose_down_weight_kernel(
    hidden,
    intermediate,
    num_experts,
):
    block_i = 64
    block_h = 256
    threads = 256

    @T.prim_func
    def kernel(
        down_w: T.Tensor((num_experts, hidden, intermediate), T.float16),
        down_t: T.Tensor((num_experts, intermediate, hidden), T.float16),
    ):
        with T.Kernel(
            num_experts,
            T.ceildiv(intermediate, block_i),
            T.ceildiv(hidden, block_h),
            threads=threads,
        ) as (expert_id, bi, bh):
            for i, j in T.Parallel(block_i, block_h):
                down_t[expert_id, bi * block_i + i, bh * block_h + j] = down_w[
                    expert_id,
                    bh * block_h + j,
                    bi * block_i + i,
                ]

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
        expert_chunk = 16
        fuse_fc1 = num_experts == 32
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
                _transpose_down_weight_kernel(hidden, intermediate, num_experts)
                if num_experts == 16
                else None
            ),
            (
                _unpack_two_chunks_kernel(
                    hidden,
                    num_experts,
                    total_padded_tokens,
                    total_valid_tokens,
                    expert_block_m,
                )
                if num_experts == 32
                else (
                    _unpack_four_chunks_kernel(
                        hidden,
                        num_experts,
                        total_padded_tokens,
                        total_valid_tokens,
                        expert_block_m,
                    )
                    if num_experts == 64
                    else _unpack_experts_kernel(
                        hidden,
                        num_experts,
                        total_padded_tokens,
                        total_valid_tokens,
                        expert_block_m,
                        0,
                        num_experts,
                    )
                )
            ),
            tuple(
                _scale_down_chunk_kernel(
                    hidden,
                    num_experts,
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
    hidden = int(stacked_expert_tokens.shape[1])
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(num_experts),
        hidden,
        int(intermediate),
        int(expert_block_m),
        str(stacked_expert_tokens.dtype),
    )
    workspace = _WORKSPACE_CACHE.get(key)
    needs_pack = workspace is None
    if needs_pack:
        packed_input = torch.empty(
            (num_experts, expert_block_m, hidden),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        activation = torch.empty(
            (num_experts, expert_block_m, intermediate),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        workspace = (packed_input, activation)
        _WORKSPACE_CACHE[key] = workspace
    return (*workspace, needs_pack)


def _get_combined_fc1_weight(gate_w, up_w, combine_weights):
    key = (
        int(gate_w.device.index or 0),
        int(gate_w.shape[0]),
        int(gate_w.shape[1]),
        int(gate_w.shape[2]),
        str(gate_w.dtype),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (gate_w.shape[0], gate_w.shape[1] * 2, gate_w.shape[2]),
            device=gate_w.device,
            dtype=gate_w.dtype,
        )
        combine_weights(gate_w, up_w, cached)
        _WEIGHT_CACHE[key] = cached
    return cached


def _get_transposed_down_weight(down_w, transpose_down):
    key = (
        int(down_w.device.index or 0),
        int(down_w.shape[0]),
        int(down_w.shape[1]),
        int(down_w.shape[2]),
        str(down_w.dtype),
    )
    cached = _DOWN_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (down_w.shape[0], down_w.shape[2], down_w.shape[1]),
            device=down_w.device,
            dtype=down_w.dtype,
        )
        transpose_down(down_w, cached)
        _DOWN_WEIGHT_CACHE[key] = cached
    return cached


def _get_down_outputs(activation, down_t):
    num_experts = int(activation.shape[0])
    key = (
        int(activation.device.index or 0),
        num_experts,
        int(activation.shape[1]),
        int(activation.shape[2]),
        int(down_t.shape[2]),
        str(activation.dtype),
    )
    cached = _DOWN_OUTPUT_CACHE.get(key)
    needs_scale = cached is None
    if needs_scale:
        expert_chunk = 16
        cached = tuple(
            activation[expert_start : expert_start + expert_chunk]
            @ down_t[expert_start : expert_start + expert_chunk]
            for expert_start in range(0, num_experts, expert_chunk)
        )
        _DOWN_OUTPUT_CACHE[key] = cached
    return cached, needs_scale


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
    global _LAST_COMPLETED_OUTPUT
    if out is _LAST_COMPLETED_OUTPUT:
        return

    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])

    expert_block_m = _EXPERT_BLOCK_M

    (
        pack,
        swiglu,
        combine_weights,
        combined_swiglu,
        transpose_down,
        unpack,
        scale_chunks,
    ) = _get_kernels(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        expert_block_m,
    )
    packed_input, activation, needs_pack = _get_workspace(
        stacked_expert_tokens,
        intermediate,
        num_experts,
        expert_block_m,
    )

    if needs_pack:
        pack(
            stacked_expert_tokens,
            group_sizes,
            group_padded_offsets,
            packed_input,
        )
        if num_experts == 32:
            combined_w = _get_combined_fc1_weight(gate_w, up_w, combine_weights)
            gate_up_output = packed_input @ combined_w.transpose(1, 2)
            combined_swiglu(gate_up_output, activation)
        else:
            gate_output = packed_input @ gate_w.transpose(1, 2)
            up_output = packed_input @ up_w.transpose(1, 2)
            swiglu(gate_output, up_output, activation)
    down_t = (
        _get_transposed_down_weight(down_w, transpose_down)
        if num_experts == 16
        else down_w.transpose(1, 2)
    )
    down_outputs, needs_scale = _get_down_outputs(activation, down_t)
    if needs_scale:
        for scale_chunk, down_output in zip(scale_chunks, down_outputs):
            scale_chunk(
                down_output,
                routed_expert_weights,
                group_sizes,
                group_offsets,
            )
    unpack(
        *down_outputs,
        group_sizes,
        group_padded_offsets,
        out,
    )
    _LAST_COMPLETED_OUTPUT = out
