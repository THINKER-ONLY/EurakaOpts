# Pure-TileLang MLA decode kernel.  Assumptions: the five input/output tensors
# are contiguous 16-bit tensors with the layouts declared below, kv_head_num is
# exactly 1, and a split count never exceeds the KV sequence length.  Proposed
# improvements (theoretical only): balanced split ranges instead of floor-sized
# chunks, explicit masking of split/tail score columns, removal of the redundant
# shared -> fragment cast in the split path, two-stage pipelining and 256 threads
# in the no-split path, FP32 split-LSE storage, and a head tile that always
# divides the head count.  No TileLang/CUDA compiler or benchmark was available,
# so these changes have not been compile-tested or performance-tuned on hardware.

import tilelang
import tilelang.language as T


def _usable_block_h(heads, block_h):
    """Largest tile row count <= block_h that covers every query head."""
    limit = min(int(block_h), int(heads))
    for candidate in range(limit, 0, -1):
        if heads % candidate == 0:
            return candidate
    return 1


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
    assert kv_head_num == 1, "kv_head_num must be 1"
    assert heads > 0
    assert seqlen_kv > 0
    assert block_n > 0
    assert num_split >= 1
    if num_split > 1:
        assert num_split <= seqlen_kv

    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = _usable_block_h(heads, min(block_h, kv_group_num))

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # FP32 LSE avoids an avoidable half-precision round trip before the
        # split-combination kernel.  Partial outputs stay in the output dtype.
        glse = T.alloc_global([batch, heads, num_split], accum_dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        with T.Kernel(
            batch,
            heads // valid_block_h,
            num_split,
            threads=256,
        ) as (bid, hid, bz):
            Q_shared = T.alloc_shared([valid_block_h, dim], dtype)
            S_shared = T.alloc_shared([valid_block_h, block_n], dtype)
            Q_pe_shared = T.alloc_shared([valid_block_h, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_n, dim], dtype)
            K_pe_shared = T.alloc_shared([block_n, pe_dim], dtype)
            O_shared = T.alloc_shared([valid_block_h, dim], dtype)
            acc_s = T.alloc_fragment([valid_block_h, block_n], accum_dtype)
            acc_o = T.alloc_fragment([valid_block_h, dim], accum_dtype)
            scores_max = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_scale = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_sum = T.alloc_fragment([valid_block_h], accum_dtype)
            logsum = T.alloc_fragment([valid_block_h], accum_dtype)

            # kv_head_num == 1, so every query head uses the only KV head.
            cur_kv_head = 0

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

            # Balanced split boundaries: the union is [0, seqlen_kv) with no
            # overlap, including when num_split does not divide seqlen_kv.
            split_len_max = (seqlen_kv + num_split - 1) // num_split
            loop_range = T.ceildiv(split_len_max, block_n)
            kv_start = (seqlen_kv * bz) // num_split
            kv_end = (seqlen_kv * (bz + 1)) // num_split

            for k in T.Pipelined(loop_range, num_stages=2):
                kv_block_start = kv_start + k * block_n
                T.copy(
                    KV[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    KV_shared,
                )
                T.copy(
                    K_pe[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
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

                # A block can cross this split's end even though it is still
                # inside the complete KV tensor.  Mask those columns so split
                # results remain disjoint.
                for i, j in T.Parallel(valid_block_h, block_n):
                    acc_s[i, j] = T.if_then_else(
                        kv_block_start + j < kv_end,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(valid_block_h):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(valid_block_h, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(valid_block_h):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(valid_block_h, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    S_shared,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for i, j in T.Parallel(valid_block_h, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(valid_block_h):
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
            for k in T.serial(num_split):
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
            heads // valid_block_h,
            batch,
            threads=256,
        ) as (hid, bid):
            Q_shared = T.alloc_shared([valid_block_h, dim], dtype)
            S_shared = T.alloc_shared([valid_block_h, block_n], dtype)
            Q_pe_shared = T.alloc_shared([valid_block_h, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_n, dim], dtype)
            K_pe_shared = T.alloc_shared([block_n, pe_dim], dtype)
            O_shared = T.alloc_shared([valid_block_h, dim], dtype)
            acc_s = T.alloc_fragment([valid_block_h, block_n], accum_dtype)
            acc_o = T.alloc_fragment([valid_block_h, dim], accum_dtype)
            scores_max = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_scale = T.alloc_fragment([valid_block_h], accum_dtype)
            scores_sum = T.alloc_fragment([valid_block_h], accum_dtype)
            logsum = T.alloc_fragment([valid_block_h], accum_dtype)

            cur_kv_head = 0

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
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_block_start = k * block_n
                T.copy(
                    KV[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    KV_shared,
                )
                T.copy(
                    K_pe[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
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

                for i, j in T.Parallel(valid_block_h, block_n):
                    acc_s[i, j] = T.if_then_else(
                        kv_block_start + j < seqlen_kv,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(valid_block_h):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(valid_block_h, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(valid_block_h):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(valid_block_h, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    S_shared,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for i, j in T.Parallel(valid_block_h, dim):
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
    assert kv_heads == 1, "kv_head_num must be 1"
    block_n = 32
    block_h = _usable_block_h(heads, min(16, heads // kv_heads))
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
