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
                    if i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = (
                            packed_output0[local_expert_id, i, bn * block_n + j]
                        )
            else:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                raw_start1 = group_offsets[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size1:
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
                    if i < group_size0:
                        out[padded_start0 + i, bn * block_n + j] = (
                            packed_output0[local_expert_id, i, bn * block_n + j]
                        )
            elif chunk_id == 1:
                expert_id1 = 16 + local_expert_id
                group_size1 = group_sizes[expert_id1]
                raw_start1 = group_offsets[expert_id1]
                padded_start1 = group_padded_offsets[expert_id1]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size1:
                        out[padded_start1 + i, bn * block_n + j] = (
                            packed_output1[local_expert_id, i, bn * block_n + j]
                        )
            elif chunk_id == 2:
                expert_id2 = 32 + local_expert_id
                group_size2 = group_sizes[expert_id2]
                raw_start2 = group_offsets[expert_id2]
                padded_start2 = group_padded_offsets[expert_id2]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size2:
                        out[padded_start2 + i, bn * block_n + j] = (
                            packed_output2[local_expert_id, i, bn * block_n + j]
                        )
            else:
                expert_id3 = 48 + local_expert_id
                group_size3 = group_sizes[expert_id3]
                raw_start3 = group_offsets[expert_id3]
                padded_start3 = group_padded_offsets[expert_id3]
                for i, j in T.Parallel(expert_block_m, block_n):
                    if i < group_size3:
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
                if num_experts == 64
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
        workspace = (packed_input, activation)
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

    (
        pack,
        swiglu,
        combine_weights,
        chunk_swiglu,
        transpose_down,
        unpack,
    ) = _get_kernels(
        hidden,
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

    if num_experts != 64:
        pack(
            stacked_expert_tokens,
            group_sizes,
            group_padded_offsets,
            packed_input,
        )
    if num_experts in (32, 64):
        combined_w = _get_combined_fc1_weight(gate_w, up_w, combine_weights)
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
                if num_experts == 64:
                    pack[stream_index](
                        stacked_expert_tokens,
                        group_sizes,
                        group_padded_offsets,
                        packed_input[stream_expert_start:stream_expert_end],
                    )
                stream_gate_up_output = (
                    packed_input[stream_expert_start:stream_expert_end]
                    @ combined_w[stream_expert_start:stream_expert_end].transpose(
                        1, 2
                    )
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
