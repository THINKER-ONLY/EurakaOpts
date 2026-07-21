import ctypes

import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}
_WEIGHT_CACHE = {}
_DOWN_WEIGHT_CACHE = {}
_DOWN_STREAM_CACHE = {}
_GRAPH_CACHE = {}
_GRAPH_SEEN = set()
_GRAPH_CAPTURE_ACTIVE = False
_EXPERT_BLOCK_M = 192
_MCBLAS_LIB = None
_MCBLAS_HANDLES = {}
_MCBLAS_ALPHA = ctypes.c_uint16(0x3C00)
_MCBLAS_BETA = ctypes.c_uint16(0)


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
    block_n = 128
    threads = 512
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
            # Padded rows never reach unpack, so they do not need initialization.
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
def _pack_expert_chunk_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    expert_block_m,
    expert_start,
    expert_chunk,
):
    block_n = 256
    threads = 256
    packed_shape = (expert_chunk, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor((total_padded_tokens, hidden), T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        packed_input: T.Tensor(packed_shape, T.float16),
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
                    packed_input[local_expert_id, i, bn * block_n + j] = (
                        stacked_expert_tokens[padded_start + i, bn * block_n + j]
                    )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _pack_small_last_expert_kernel(
    hidden,
    num_experts,
    total_padded_tokens,
    expert_block_m,
):
    block_n = 256
    threads = 256

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor((total_padded_tokens, hidden), T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        packed_input: T.Tensor((1, expert_block_m, hidden), T.float16),
    ):
        with T.Kernel(T.ceildiv(hidden, block_n), threads=threads) as bn:
            expert_id = num_experts - 1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            if group_size < 128:
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size:
                        packed_input[0, i, bn * block_n + j] = (
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
    block_n = 128
    threads = 512
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
                        * T.cast(
                            routed_expert_weights[raw_start + i],
                            T.float16,
                        )
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
    block_n = 256
    threads = 1024
    packed_shape = (16, expert_block_m, hidden)

    @T.prim_func
    def kernel(
        packed_output0: T.Tensor(packed_shape, T.float16),
        packed_output1: T.Tensor(packed_shape, T.float16),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
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
                raw_start0 = group_offsets[local_expert_id]
                padded_start0 = group_padded_offsets[local_expert_id]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size0 < 128 and i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = (
                            packed_output0[local_expert_id, i, bn * block_n + j]
                        )
            else:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                raw_start1 = group_offsets[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size1 < 128 and i < group_size1:
                        out[padded_start1 + i, bn * block_n + j] = (
                            packed_output1[local_expert_id, i, bn * block_n + j]
                        )

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
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
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
                raw_start0 = group_offsets[local_expert_id]
                padded_start0 = group_padded_offsets[local_expert_id]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size0 < 128 and i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = (
                            packed_output0[local_expert_id, i, bn * block_n + j]
                        )
            elif chunk_id == 1:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                raw_start1 = group_offsets[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size1 < 128 and i < group_size1:
                        out[padded_start1 + i, bn * block_n + j] = (
                            packed_output1[local_expert_id, i, bn * block_n + j]
                        )
            elif chunk_id == 2:
                expert_id2 = 32 + local_expert_id
                group_size2 = group_sizes[expert_id2]
                raw_start2 = group_offsets[expert_id2]
                padded_start2 = group_padded_offsets[expert_id2]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size2 < 128 and i < group_size2:
                        out[padded_start2 + i, bn * block_n + j] = (
                            packed_output2[local_expert_id, i, bn * block_n + j]
                        )
            else:
                expert_id3 = 48 + local_expert_id
                group_size3 = group_sizes[expert_id3]
                raw_start3 = group_offsets[expert_id3]
                padded_start3 = group_padded_offsets[expert_id3]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if group_size3 < 128 and i < group_size3:
                        out[padded_start3 + i, bn * block_n + j] = (
                            packed_output3[local_expert_id, i, bn * block_n + j]
                        )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _swiglu_kernel(
    intermediate,
    num_experts,
    expert_block_m,
):
    block_n = 128
    threads = 512
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
    total_valid_tokens,
    expert_block_m,
    expert_start,
):
    block_n = 64
    threads = 256
    expert_chunk = 16
    combined_shape = (expert_chunk, expert_block_m, intermediate * 2)
    activation_shape = (expert_chunk, expert_block_m, intermediate)

    @T.prim_func
    def kernel(
        gate_up_output: T.Tensor(combined_shape, T.float16),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        activation: T.Tensor(activation_shape, T.float16),
    ):
        with T.Kernel(
            expert_chunk,
            T.ceildiv(intermediate, block_n),
            threads=threads,
        ) as (expert_id, bn):
            global_expert_id = expert_start + expert_id
            group_size = group_sizes[global_expert_id]
            raw_start = group_offsets[global_expert_id]
            for i, j in T.Parallel(expert_block_m, block_n):
                if i < group_size:
                    gate = gate_up_output[expert_id, i, bn * block_n + j]
                    route_weight = (
                        T.cast(
                            routed_expert_weights[raw_start + i],
                            T.float16,
                        )
                        if num_experts == 64
                        else routed_expert_weights[raw_start + i]
                    )
                    activation[expert_id, i, bn * block_n + j] = (
                        gate
                        * T.sigmoid(gate)
                        * gate_up_output[
                            expert_id,
                            i,
                            intermediate + bn * block_n + j,
                        ]
                        * route_weight
                    )
                if num_experts in (32, 64) and i >= group_size:
                    activation[expert_id, i, bn * block_n + j] = 0

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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _build_fc1_pointer_table_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    expert_block_m,
):
    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(
            (total_padded_tokens, hidden),
            T.float16,
        ),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        combined_w: T.Tensor(
            (num_experts, intermediate * 2, hidden),
            T.float16,
        ),
        packed_input: T.Tensor(
            (num_experts, expert_block_m, hidden),
            T.float16,
        ),
        gate_up_output: T.Tensor(
            (num_experts, expert_block_m, intermediate * 2),
            T.float16,
        ),
        pointer_table: T.Tensor((3, num_experts), T.ptr),
    ):
        with T.Kernel(1, threads=64) as _:
            for expert in T.Parallel(num_experts):
                pointer_table[0, expert] = T.reinterpret(
                    T.address_of(
                        stacked_expert_tokens[
                            group_padded_offsets[expert],
                            0,
                        ]
                    )
                    if expert + 1 < num_experts or group_sizes[expert] >= 128
                    else T.address_of(packed_input[expert, 0, 0]),
                    T.int64,
                )
                pointer_table[1, expert] = T.reinterpret(
                    T.address_of(combined_w[expert, 0, 0]),
                    T.int64,
                )
                pointer_table[2, expert] = T.reinterpret(
                    T.address_of(gate_up_output[expert, 0, 0]),
                    T.int64,
                )

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def _build_down_pointer_table_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    expert_block_m,
):
    source = r"""\
#include <tl_templates/maca/common.h>
__device__ __forceinline__ void build_down_pointer_table_raw(
    const void* activation,
    const void* down_w,
    const void* down_output,
    const int32_t* group_sizes,
    const int32_t* group_padded_offsets,
    void* out,
    int64_t* pointer_table,
    int num_experts,
    int expert_block_m,
    int hidden,
    int intermediate) {
  int expert = static_cast<int>(threadIdx.x);
  if (expert < num_experts) {
    pointer_table[expert] = reinterpret_cast<int64_t>(activation)
        + static_cast<int64_t>(expert) * expert_block_m * intermediate * 2;
    pointer_table[num_experts + expert] = reinterpret_cast<int64_t>(down_w)
        + static_cast<int64_t>(expert) * hidden * intermediate * 2;
    pointer_table[2 * num_experts + expert] =
        group_sizes[expert] >= 128
        ? reinterpret_cast<int64_t>(out)
            + static_cast<int64_t>(group_padded_offsets[expert]) * hidden * 2
        : reinterpret_cast<int64_t>(down_output)
            + static_cast<int64_t>(expert) * expert_block_m * hidden * 2;
  }
}
"""

    @T.prim_func
    def kernel(
        activation: T.Tensor(
            (num_experts, expert_block_m, intermediate),
            T.float16,
        ),
        down_w: T.Tensor((num_experts, hidden, intermediate), T.float16),
        down_output: T.Tensor(
            (num_experts, expert_block_m, hidden),
            T.float16,
        ),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), T.float16),
        pointer_table: T.Tensor((3, num_experts), T.int64),
    ):
        with T.Kernel(1, threads=64) as _:
            T.import_source(source)
            T.call_extern(
                "handle",
                "build_down_pointer_table_raw",
                activation.data,
                down_w.data,
                down_output.data,
                group_sizes.data,
                group_padded_offsets.data,
                T.access_ptr(out[0, 0], "w", total_padded_tokens * hidden),
                T.access_ptr(pointer_table[0, 0], "w", 3 * num_experts),
                num_experts,
                expert_block_m,
                hidden,
                intermediate,
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
        expert_chunk = 16
        fuse_fc1 = num_experts in (32, 64)
        kernels = (
            (
                tuple(
                    _pack_expert_chunk_kernel(
                        hidden,
                        num_experts,
                        total_padded_tokens,
                        expert_block_m,
                        stream_index * (num_experts // 2),
                        num_experts // 2,
                    )
                    for stream_index in range(2)
                )
                if num_experts in (32, 64)
                else _pack_experts_kernel(
                    hidden,
                    num_experts,
                    total_padded_tokens,
                    expert_block_m,
                )
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
                tuple(
                    _combined_swiglu_kernel(
                        intermediate,
                        num_experts,
                        total_valid_tokens,
                        expert_block_m,
                        expert_start,
                    )
                    for expert_start in range(0, num_experts, expert_chunk)
                )
                if num_experts in (32, 64)
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
            (
                _pack_small_last_expert_kernel(
                    hidden,
                    num_experts,
                    total_padded_tokens,
                    expert_block_m,
                )
                if num_experts in (32, 64)
                else None
            ),
            (
                _build_fc1_pointer_table_kernel(
                    hidden,
                    intermediate,
                    num_experts,
                    total_padded_tokens,
                    expert_block_m,
                )
                if num_experts in (32, 64)
                else None
            ),
            (
                _build_down_pointer_table_kernel(
                    hidden,
                    intermediate,
                    num_experts,
                    total_padded_tokens,
                    expert_block_m,
                )
                if num_experts in (32, 64)
                else None
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
    if workspace is None:
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
            if num_experts in (32, 64)
            else None
        )
        gate_up_output = (
            torch.empty(
                (num_experts, expert_block_m, intermediate * 2),
                device=stacked_expert_tokens.device,
                dtype=stacked_expert_tokens.dtype,
            )
            if num_experts in (32, 64)
            else None
        )
        pointer_table = (
            torch.empty(
                (3, num_experts),
                device=stacked_expert_tokens.device,
                dtype=torch.int64,
            )
            if num_experts in (32, 64)
            else None
        )
        down_output = (
            torch.empty(
                (num_experts, expert_block_m, hidden),
                device=stacked_expert_tokens.device,
                dtype=stacked_expert_tokens.dtype,
            )
            if num_experts in (32, 64)
            else None
        )
        down_pointer_table = (
            torch.empty(
                (3, num_experts),
                device=stacked_expert_tokens.device,
                dtype=torch.int64,
            )
            if num_experts in (32, 64)
            else None
        )
        workspace = (
            packed_input,
            activation,
            gate_up_output,
            pointer_table,
            down_output,
            down_pointer_table,
        )
        _WORKSPACE_CACHE[key] = workspace
    return workspace


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


def _get_down_streams(device, num_experts):
    key = (int(device.index or 0), int(num_experts))
    streams = _DOWN_STREAM_CACHE.get(key)
    if streams is None:
        streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
        _DOWN_STREAM_CACHE[key] = streams
    return streams


def _get_mcblas():
    global _MCBLAS_LIB
    if _MCBLAS_LIB is None:
        lib = ctypes.CDLL("libmcblas.so")
        voidp = ctypes.c_void_p
        lib.mcblasCreate.argtypes = [ctypes.POINTER(voidp)]
        lib.mcblasCreate.restype = ctypes.c_int
        lib.mcblasSetStream.argtypes = [voidp, voidp]
        lib.mcblasSetStream.restype = ctypes.c_int
        lib.mcblasHgemmBatched.argtypes = [
            voidp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            voidp,
            ctypes.POINTER(voidp),
            ctypes.c_int,
            ctypes.POINTER(voidp),
            ctypes.c_int,
            voidp,
            ctypes.POINTER(voidp),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mcblasHgemmBatched.restype = ctypes.c_int
        _MCBLAS_LIB = lib
    return _MCBLAS_LIB


def _get_mcblas_handle(stream):
    stream_ptr = int(stream.cuda_stream)
    handle = _MCBLAS_HANDLES.get(stream_ptr)
    if handle is None:
        lib = _get_mcblas()
        handle = ctypes.c_void_p()
        status = lib.mcblasCreate(ctypes.byref(handle))
        if status:
            raise RuntimeError(f"mcblasCreate failed: {status}")
        status = lib.mcblasSetStream(
            handle,
            ctypes.c_void_p(stream_ptr),
        )
        if status:
            raise RuntimeError(f"mcblasSetStream failed: {status}")
        _MCBLAS_HANDLES[stream_ptr] = handle
    return handle


def _pointer_bmm_fc1(
    pointer_table,
    stream,
    expert_start,
    expert_count,
    hidden,
    intermediate,
    fc1_block_m,
):
    lib = _get_mcblas()
    handle = _get_mcblas_handle(stream)
    base = int(pointer_table.data_ptr())
    expert_count_total = int(pointer_table.shape[1])
    pointer_size = ctypes.sizeof(ctypes.c_void_p)

    def table_row(row):
        address = base + (row * expert_count_total + expert_start) * pointer_size
        return ctypes.cast(
            ctypes.c_void_p(address),
            ctypes.POINTER(ctypes.c_void_p),
        )

    status = lib.mcblasHgemmBatched(
        handle,
        1,
        0,
        intermediate * 2,
        fc1_block_m,
        hidden,
        ctypes.byref(_MCBLAS_ALPHA),
        table_row(1),
        hidden,
        table_row(0),
        hidden,
        ctypes.byref(_MCBLAS_BETA),
        table_row(2),
        intermediate * 2,
        expert_count,
    )
    if status:
        raise RuntimeError(f"mcblasHgemmBatched failed: {status}")


def _pointer_bmm_down(
    pointer_table,
    stream,
    expert_start,
    expert_count,
    hidden,
    intermediate,
    expert_block_m,
):
    lib = _get_mcblas()
    handle = _get_mcblas_handle(stream)
    base = int(pointer_table.data_ptr())
    expert_count_total = int(pointer_table.shape[1])
    pointer_size = ctypes.sizeof(ctypes.c_void_p)

    def table_row(row):
        address = base + (row * expert_count_total + expert_start) * pointer_size
        return ctypes.cast(
            ctypes.c_void_p(address),
            ctypes.POINTER(ctypes.c_void_p),
        )

    status = lib.mcblasHgemmBatched(
        handle,
        1,
        0,
        hidden,
        expert_block_m,
        intermediate,
        ctypes.byref(_MCBLAS_ALPHA),
        table_row(1),
        intermediate,
        table_row(0),
        intermediate,
        ctypes.byref(_MCBLAS_BETA),
        table_row(2),
        hidden,
        expert_count,
    )
    if status:
        raise RuntimeError(f"mcblasHgemmBatched down failed: {status}")


def _graph_key(args):
    return tuple(
        (
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )
        for tensor in args
    )


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
    global _GRAPH_CAPTURE_ACTIVE
    if gate_w.shape[0] in (32, 64) and not _GRAPH_CAPTURE_ACTIVE:
        args = (
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
        key = _graph_key(args)
        graph = _GRAPH_CACHE.get(key)
        if graph is not None:
            graph.replay()
            return
        if key in _GRAPH_SEEN:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            _GRAPH_CAPTURE_ACTIVE = True
            try:
                with torch.cuda.graph(graph):
                    run_kernel(*args)
            finally:
                _GRAPH_CAPTURE_ACTIVE = False
            _GRAPH_CACHE[key] = graph
            return
        _GRAPH_SEEN.add(key)

    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])

    expert_block_m = 176
    if num_experts == 32:
        fc1_block_m = 170
    elif num_experts == 64:
        fc1_block_m = 166
    else:
        fc1_block_m = expert_block_m

    (
        pack,
        swiglu,
        combine_weights,
        chunk_swiglu,
        transpose_down,
        unpack,
        last_pack,
        build_fc1_pointer_table,
        build_down_pointer_table,
    ) = _get_kernels(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        expert_block_m,
    )
    (
        packed_input,
        activation,
        gate_up_workspace,
        pointer_table,
        down_output,
        down_pointer_table,
    ) = _get_workspace(
        stacked_expert_tokens,
        intermediate,
        num_experts,
        expert_block_m,
    )

    if num_experts not in (32, 64):
        pack(
            stacked_expert_tokens,
            group_sizes,
            group_padded_offsets,
            packed_input,
        )
    if num_experts in (32, 64):
        combined_w = _get_combined_fc1_weight(gate_w, up_w, combine_weights)
        last_pack(
            stacked_expert_tokens,
            group_sizes,
            group_padded_offsets,
            packed_input[num_experts - 1 : num_experts],
        )
        build_fc1_pointer_table(
            stacked_expert_tokens,
            group_sizes,
            group_padded_offsets,
            combined_w,
            packed_input,
            gate_up_workspace,
            pointer_table,
        )
    else:
        gate_output = packed_input @ gate_w.transpose(1, 2)
        up_output = packed_input @ up_w.transpose(1, 2)
        swiglu(gate_output, up_output, gate_output)
        activation = gate_output
    down_t = (
        _get_transposed_down_weight(down_w, transpose_down)
        if num_experts == 16
        else down_w.transpose(1, 2)
    )
    expert_chunk = 16
    if num_experts in (32, 64):
        build_down_pointer_table(
            activation,
            down_w,
            down_output,
            group_sizes,
            group_padded_offsets,
            out,
            down_pointer_table,
        )
        current_stream = torch.cuda.current_stream(device=activation.device)
        down_streams = _get_down_streams(activation.device, num_experts)
        for stream in down_streams:
            stream.wait_stream(current_stream)
        num_chunks = num_experts // expert_chunk
        chunks_per_stream = num_chunks // len(down_streams)
        down_outputs = [None] * num_chunks
        for stream_index, stream in enumerate(down_streams):
            with torch.cuda.stream(stream):
                chunk_start = stream_index * chunks_per_stream
                chunk_end = chunk_start + chunks_per_stream
                stream_expert_start = chunk_start * expert_chunk
                stream_expert_end = chunk_end * expert_chunk
                stream_gate_up_output = gate_up_workspace[
                    stream_expert_start:stream_expert_end
                ]
                _pointer_bmm_fc1(
                    pointer_table,
                    stream,
                    stream_expert_start,
                    stream_expert_end - stream_expert_start,
                    hidden,
                    intermediate,
                    fc1_block_m,
                )
                for chunk_index in range(chunk_start, chunk_end):
                    expert_start = chunk_index * expert_chunk
                    expert_end = expert_start + expert_chunk
                    local_start = expert_start - stream_expert_start
                    local_end = expert_end - stream_expert_start
                    chunk_swiglu[chunk_index](
                        stream_gate_up_output[local_start:local_end],
                        routed_expert_weights,
                        group_sizes,
                        group_offsets,
                        activation[expert_start:expert_end],
                    )
                    if num_experts in (32, 64):
                        _pointer_bmm_down(
                            down_pointer_table,
                            stream,
                            expert_start,
                            expert_end - expert_start,
                            hidden,
                            intermediate,
                            expert_block_m,
                        )
                        down_outputs[chunk_index] = down_output[
                            expert_start:expert_end
                        ]
                    else:
                        down_outputs[chunk_index] = (
                            activation[expert_start:expert_end]
                            @ down_t[expert_start:expert_end]
                        )
        for stream in down_streams:
            current_stream.wait_stream(stream)
        down_outputs = tuple(down_outputs)
        for down_output in down_outputs:
            down_output.record_stream(current_stream)
    else:
        down_outputs = tuple(
            (
                activation[expert_start:expert_end]
                @ down_t[expert_start:expert_end]
            )
            for expert_start in range(0, num_experts, expert_chunk)
            for expert_end in (min(expert_start + expert_chunk, num_experts),)
        )
    if num_experts == 16:
        unpack(
            down_outputs[0],
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            out,
        )
    else:
        unpack(
            *down_outputs,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            out,
        )
