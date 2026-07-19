import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_n,
    block_h,
    num_split,
    softmax_scale,
):
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = min(block_h, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split], dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        with T.Kernel(
            batch,
            heads // min(block_h, kv_group_num),
            num_split,
            threads=256,
        ) as (bid, hid, bz):
            Q_shared = T.alloc_shared([block_h, dim], dtype)
            S_shared = T.alloc_shared([block_h, block_n], dtype)
            Q_pe_shared = T.alloc_shared([block_h, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_n, dim], dtype)
            K_pe_shared = T.alloc_shared([block_n, pe_dim], dtype)
            O_shared = T.alloc_shared([block_h, dim], dtype)
            acc_s = T.alloc_fragment([block_h, block_n], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_h, block_n], dtype)
            acc_o = T.alloc_fragment([block_h, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_h], accum_dtype)
            scores_scale = T.alloc_fragment([block_h], accum_dtype)
            scores_sum = T.alloc_fragment([block_h], accum_dtype)
            logsum = T.alloc_fragment([block_h], accum_dtype)
            cur_kv_head = hid // (kv_group_num // block_h)

            T.use_swizzle(10)
            T.copy(
                Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_shared,
            )
            T.copy(
                Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_pe_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv // num_split, block_n)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = (seqlen_kv // num_split) * bz + k * block_n
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_n
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)

                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_h):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_h, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_h):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_h, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    acc_s_cast,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for i, j in T.Parallel(block_h, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_h):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(
                logsum,
                glse[
                    bid,
                    hid * valid_block_h : (hid + 1) * valid_block_h,
                    bz,
                ],
            )
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                output_partial[
                    bid,
                    hid * valid_block_h : (hid + 1) * valid_block_h,
                    bz,
                    :,
                ],
            )

        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)

            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, hid, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local

            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = output_partial[bz, hid, k, i]
                lse_local_split = glse[bz, hid, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local

            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(
            heads // min(block_h, kv_group_num), batch, threads=128
        ) as (hid, bid):
            Q_shared = T.alloc_shared([block_h, dim], dtype)
            S_shared = T.alloc_shared([block_h, block_n], dtype)
            Q_pe_shared = T.alloc_shared([block_h, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_n, dim], dtype)
            K_pe_shared = T.alloc_shared([block_n, pe_dim], dtype)
            O_shared = T.alloc_shared([block_h, dim], dtype)
            acc_s = T.alloc_fragment([block_h, block_n], accum_dtype)
            acc_o = T.alloc_fragment([block_h, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_h], accum_dtype)
            scores_scale = T.alloc_fragment([block_h], accum_dtype)
            scores_sum = T.alloc_fragment([block_h], accum_dtype)
            logsum = T.alloc_fragment([block_h], accum_dtype)
            cur_kv_head = hid // (kv_group_num // block_h)

            T.copy(
                Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_shared,
            )
            T.copy(
                Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_pe_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_n)
            for k in T.Pipelined(loop_range, num_stages=0):
                T.copy(
                    KV[
                        bid,
                        k * block_n : (k + 1) * block_n,
                        cur_kv_head,
                        :,
                    ],
                    KV_shared,
                )
                T.copy(
                    K_pe[
                        bid,
                        k * block_n : (k + 1) * block_n,
                        cur_kv_head,
                        :,
                    ],
                    K_pe_shared,
                )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_h):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_h, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_h):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_h, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    S_shared,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for i, j in T.Parallel(block_h, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[
                    bid,
                    hid * valid_block_h : (hid + 1) * valid_block_h,
                    :,
                ],
            )

    if num_split > 1:
        return main_split
    return main_no_split


_KERNEL_CACHE = {}


def _get_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    block_n = 32
    block_h = min(16, heads // kv_heads)
    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5
    key = (
        batch,
        heads,
        kv_heads,
        kv_ctx,
        dim,
        pe_dim,
        block_n,
        block_h,
        num_split,
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = flashattn(
            batch,
            heads,
            kv_heads,
            kv_ctx,
            dim,
            pe_dim,
            block_n,
            block_h,
            num_split,
            softmax_scale,
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def run_kernel(
    q,
    q_pe,
    kv,
    k_pe,
    output,
    batch,
    heads,
    kv_heads,
    kv_ctx,
    dim,
    pe_dim,
):
    kernel = _get_kernel(
        int(batch),
        int(heads),
        int(kv_heads),
        int(kv_ctx),
        int(dim),
        int(pe_dim),
    )
    kernel(q, q_pe, kv, k_pe, output)
