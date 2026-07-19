import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_size,
    groups,
    selected_blocks,
):
    scale = float((dim**-0.5) * 1.44269504)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]

    dtype = T.float16
    accum_dtype = T.float32
    block_t = min(128, tilelang.math.next_power_of_2(dim))

    assert tilelang.cdiv(dim, block_t) == 1, (
        "The key dimension can not be larger than 128"
    )

    S = selected_blocks
    G = groups
    BS = block_size
    BK = BV = block_t
    threads = (
        256
        if dim >= 64 and block_size >= 64
        else (128 if dim >= 64 and block_size >= 32 else 64)
    )
    gemm_policy = (
        T.GemmWarpPolicy.FullCol
        if S == 1 or (S in (2, 4, 8) and BS == 16 and dim == 64)
        else T.GemmWarpPolicy.FullRow
    )
    gather_n = S * BS

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            seq_len,
            tilelang.cdiv(dim, BV),
            batch * head_kv,
            threads=threads,
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = K_shared
            if threads >= 128:
                S_shared = T.alloc_shared([G, BS], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            if threads == 64:
                acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            if S > 1:
                scores_max_prev = T.alloc_fragment([G], accum_dtype)
                scores_scale = T.alloc_fragment([G], accum_dtype)
                logsum = T.alloc_fragment([G], accum_dtype)

            i_t = bx
            i_v = by
            i_bh = bz
            i_b = i_bh // head_kv
            i_h = i_bh % head_kv

            T.copy(
                Q[i_b, i_t, i_h * G : (i_h + 1) * G, :],
                Q_shared,
            )
            if S > 1:
                T.fill(acc_o, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0)

            for s in T.serial(S):
                i_s = BlockIndices[i_b, i_t, i_h, s] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(
                                i_t >= i_s + j,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=gemm_policy,
                    )

                    if S > 1:
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                        for i in T.Parallel(G):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                    else:
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    if S > 1:
                        for i in T.Parallel(G):
                            logsum[i] = (
                                logsum[i] * scores_scale[i] + scores_sum[i]
                            )
                    if threads >= 128:
                        T.copy(acc_s, S_shared)
                    else:
                        T.copy(acc_s, acc_s_cast)

                    if S > 1:
                        for i, j in T.Parallel(G, BV):
                            acc_o[i, j] *= scores_scale[i]

                    T.copy(
                        V[
                            i_b,
                            i_s : i_s + BS,
                            i_h,
                            i_v * BV : (i_v + 1) * BV,
                        ],
                        V_shared,
                    )
                    if threads >= 128:
                        if S > 1:
                            T.gemm(
                                S_shared,
                                V_shared,
                                acc_o,
                                policy=gemm_policy,
                            )
                        else:
                            T.gemm(
                                S_shared,
                                V_shared,
                                acc_o,
                                policy=gemm_policy,
                                clear_accum=True,
                            )
                    else:
                        if S > 1:
                            T.gemm(
                                acc_s_cast,
                                V_shared,
                                acc_o,
                                policy=gemm_policy,
                            )
                        else:
                            T.gemm(
                                acc_s_cast,
                                V_shared,
                                acc_o,
                                policy=gemm_policy,
                                clear_accum=True,
                            )

            if S > 1:
                for i, j in T.Parallel(G, BV):
                    acc_o[i, j] /= logsum[i]
            else:
                for i, j in T.Parallel(G, BV):
                    acc_o[i, j] /= scores_sum[i]
            T.copy(
                acc_o,
                Output[
                    i_b,
                    i_t,
                    i_h * G : (i_h + 1) * G,
                    i_v * BV : (i_v + 1) * BV,
                ],
            )

    @T.prim_func
    def gathered_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            seq_len,
            tilelang.cdiv(dim, BV),
            batch * head_kv,
            threads=128,
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            KV_shared = T.alloc_shared([gather_n, BK], dtype)
            S_shared = T.alloc_shared([G, gather_n], dtype)

            acc_s = T.alloc_fragment([G, gather_n], accum_dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)

            i_t = bx
            i_v = by
            i_bh = bz
            i_b = i_bh // head_kv
            i_h = i_bh % head_kv

            T.copy(
                Q[i_b, i_t, i_h * G : (i_h + 1) * G, :],
                Q_shared,
            )

            for s in T.serial(S):
                i_s = BlockIndices[i_b, i_t, i_h, s] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(
                        K[i_b, i_s : i_s + BS, i_h, :],
                        KV_shared[s * BS : (s + 1) * BS, :],
                    )
                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, s * BS + j] = T.if_then_else(
                                i_t >= i_s + j,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, s * BS + j] = 0
                else:
                    for j, k in T.Parallel(BS, BK):
                        KV_shared[s * BS + j, k] = 0
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, s * BS + j] = -T.infinity(acc_s.dtype)

            T.gemm(
                Q_shared,
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=gemm_policy,
            )

            T.reduce_max(acc_s, scores_max, dim=1, clear=True)
            for i, j in T.Parallel(G, gather_n):
                acc_s[i, j] = T.exp2(
                    acc_s[i, j] * scale - scores_max[i] * scale
                )
            T.reduce_sum(acc_s, scores_sum, dim=1)
            T.copy(acc_s, S_shared)

            for s in T.serial(S):
                i_s = BlockIndices[i_b, i_t, i_h, s] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(
                        V[
                            i_b,
                            i_s : i_s + BS,
                            i_h,
                            i_v * BV : (i_v + 1) * BV,
                        ],
                        KV_shared[s * BS : (s + 1) * BS, :],
                    )
                else:
                    for j, k in T.Parallel(BS, BV):
                        KV_shared[s * BS + j, k] = 0

            T.gemm(
                S_shared,
                KV_shared,
                acc_o,
                policy=gemm_policy,
                clear_accum=True,
            )
            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= scores_sum[i]
            T.copy(
                acc_o,
                Output[
                    i_b,
                    i_t,
                    i_h * G : (i_h + 1) * G,
                    i_v * BV : (i_v + 1) * BV,
                ],
            )

    if S == 2 and BS == 16 and dim >= 64:
        return gathered_kernel
    return kernel


_KERNEL_CACHE = {}


def _get_kernel(B, seq_len, H, HQ, D, S, block_size, is_causal):
    groups = HQ // H
    key = (B, seq_len, H, HQ, D, S, block_size, int(is_causal))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = native_sparse_attention(
            batch=B,
            heads=HQ,
            seq_len=seq_len,
            dim=D,
            is_causal=bool(is_causal),
            block_size=block_size,
            groups=groups,
            selected_blocks=S,
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def run_kernel(
    q,
    k,
    v,
    block_indices,
    output,
    B,
    seq_len,
    H,
    HQ,
    D,
    S,
    block_size,
    is_causal,
):
    kernel = _get_kernel(
        int(B),
        int(seq_len),
        int(H),
        int(HQ),
        int(D),
        int(S),
        int(block_size),
        int(is_causal),
    )
    kernel(q, k, v, block_indices, output)
