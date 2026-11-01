# Pure-TileLang MLA decode kernel (single-file candidate).
#
# STATUS: This candidate has NOT been compiled, numerically validated, or
# benchmarked on the MetaX target. No TileLang compiler, GPU, remote baseline
# or device properties were available while it was written. Nothing below is
# a measured performance claim and the candidate must not be called validated
# or faster.
#
# Assumptions (public contract):
#   * q, q_pe and output are contiguous 16-bit tensors shaped
#     [batch, heads, dim], [batch, heads, pe_dim] and [batch, heads, dim].
#   * kv and k_pe are contiguous 16-bit tensors shaped
#     [batch, seqlen_kv, kv_head_num, dim] and
#     [batch, seqlen_kv, kv_head_num, pe_dim].
#   * kv_head_num == 1, so every query head uses the single KV head.
#   * output is written in place by run_kernel; run_kernel allocates nothing.
#   * num_split <= seqlen_kv, so every split range is non-empty.
#
# Conservative changes versus the already published candidate:
#   * A shape-only split heuristic (long context, few batch/head CTAs) enables
#     the existing split path. It is an unmeasured hypothesis about grid
#     parallelism, not a measured win. Short or high-batch shapes stay on the
#     no-split path. Split scratch is bounded and tiny splits are not asked for.
#   * The query-head tile is a padded 16-row (MMA-legal) tile instead of an
#     arbitrary divisor of heads. If heads is not a multiple of the tile, padded
#     query rows are zero-filled on load and their stores are masked, so real
#     heads are covered exactly once while GEMM always sees a legal tile.
#   * Score-column tail masking is emitted only when the compile-time block
#     layout is misaligned; aligned full blocks carry no mask loop.
#   * FP32 softmax/LSE accumulation, balanced split ranges, FP32 split LSE and
#     the bounded stable split merge are retained.
#   * The no-split path keeps 256 threads and two pipeline stages. This is an
#     inherited conservative choice for MetaX portability, not a tuning result.

import tilelang
import tilelang.language as T


# Smallest row tile kept for the Q/KV GEMMs. TileLang's MMA paths expect a
# legal tensor-core tile; 16 rows is the conservative choice used here.
_HEAD_TILE = 16

# Split heuristic bounds. These are shape-only and unmeasured.
_MAX_SPLIT_BATCH = 8          # keep high-batch shapes on the no-split path
_SPLIT_TRIGGER_CTX = 8192     # only long context is considered
_MIN_SPLIT_LEN = 2048         # never create splits shorter than this
_MAX_SPLIT = 16               # hard cap on split count
_MAX_SPLIT_CTAS = 32          # do not split if there is already this much grid
_PARTIAL_ELEM_BUDGET = 8 * 1024 * 1024  # fp16 partials, elements (16 MiB)


def _usable_block_h(heads, block_h):
    """Return a 16-row-aligned query-head tile.

    The previous helper chose the largest divisor of heads, which can be
    smaller than a legal MMA tile (or can force a 1-row tile). Rounding up to
    a multiple of 16 keeps the GEMM legal; the kernel masks the padded query
    rows when heads is not a multiple of the tile.
    """
    raw = max(1, min(int(block_h), int(heads)))
    if raw <= _HEAD_TILE:
        return _HEAD_TILE
    return ((raw + _HEAD_TILE - 1) // _HEAD_TILE) * _HEAD_TILE


def _select_num_split(batch, heads, kv_ctx, dim):
    """Shape-only, deliberately unmeasured split-count heuristic.

    Long context with few CTAs leaves the no-split grid with only
    batch * ceil(heads / 16) blocks. Splitting the KV range raises parallelism
    for that case. The hypotheses (context threshold, minimum split length,
    scratch budget) are conservative and are not benchmark results.
    """
    head_blocks = (int(heads) + _HEAD_TILE - 1) // _HEAD_TILE
    if batch > _MAX_SPLIT_BATCH or kv_ctx < _SPLIT_TRIGGER_CTX:
        return 1
    if batch * head_blocks > _MAX_SPLIT_CTAS:
        return 1

    target_len = 2 * _MIN_SPLIT_LEN
    n = (int(kv_ctx) + target_len - 1) // target_len
    n = min(n, int(kv_ctx) // _MIN_SPLIT_LEN)
    n = min(n, _MAX_SPLIT)
    # Bound the fp16 output_partial scratch (batch * heads * n * dim elements).
    n = min(n, _PARTIAL_ELEM_BUDGET // max(1, int(batch) * int(heads) * int(dim)))
    if n < 2:
        return 1
    return int(n)


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
    head_blocks = (heads + valid_block_h - 1) // valid_block_h
    pad_heads = (heads % valid_block_h) != 0

    # Compile-time decisions: skip per-element mask loops for full blocks.
    needs_tail_mask = (seqlen_kv % block_n) != 0
    split_len_max = (seqlen_kv + num_split - 1) // num_split
    needs_split_mask = not (
        num_split > 1
        and (seqlen_kv % num_split == 0)
        and ((seqlen_kv // num_split) % block_n == 0)
    )
    loop_range_split = (split_len_max + block_n - 1) // block_n
    loop_range_full = (seqlen_kv + block_n - 1) // block_n

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # FP32 LSE avoids a half-precision round trip before the merge.
        glse = T.alloc_global([batch, heads, num_split], accum_dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        with T.Kernel(
            batch,
            head_blocks,
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

            # Masked head-tail load; only used when heads is not tile-aligned.
            if pad_heads:
                T.fill(Q_shared, 0)
                T.fill(Q_pe_shared, 0)
                for i, j in T.Parallel(valid_block_h, dim):
                    Q_shared[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
                for i, j in T.Parallel(valid_block_h, pe_dim):
                    Q_pe_shared[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q_pe[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
            else:
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
            kv_start = (seqlen_kv * bz) // num_split
            kv_end = (seqlen_kv * (bz + 1)) // num_split

            for k in T.Pipelined(loop_range_split, num_stages=2):
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

                # A block can cross the end of this split. Mask those columns
                # so split results stay disjoint. Aligned shapes skip this.
                if needs_split_mask:
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

            T.copy(acc_o, O_shared)

            # Masked head-tail stores; aligned shapes use a plain T.copy.
            if pad_heads:
                for i in T.Parallel(valid_block_h):
                    if hid * valid_block_h + i < heads:
                        glse[bid, hid * valid_block_h + i, bz] = logsum[i]
                for i, j in T.Parallel(valid_block_h, dim):
                    if hid * valid_block_h + i < heads:
                        output_partial[
                            bid, hid * valid_block_h + i, bz, j
                        ] = O_shared[i, j]
            else:
                T.copy(
                    logsum,
                    glse[
                        bid,
                        hid * valid_block_h : (hid + 1) * valid_block_h,
                        bz,
                    ],
                )
                T.copy(
                    O_shared,
                    output_partial[
                        bid,
                        hid * valid_block_h : (hid + 1) * valid_block_h,
                        bz,
                        :,
                    ],
                )

        # Stable bounded merge of the split partials. heads is the exact real
        # head count here, so no padded rows participate in the merge.
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
                lse_local_split = glse[bz, hid, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dim):
                    po_local[i] = output_partial[bz, hid, k, i]
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
            head_blocks,
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

            if pad_heads:
                T.fill(Q_shared, 0)
                T.fill(Q_pe_shared, 0)
                for i, j in T.Parallel(valid_block_h, dim):
                    Q_shared[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
                for i, j in T.Parallel(valid_block_h, pe_dim):
                    Q_pe_shared[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q_pe[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
            else:
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

            for k in T.Pipelined(loop_range_full, num_stages=2):
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

                # Only the final block can be partial; aligned shapes skip it.
                if needs_tail_mask:
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

            if pad_heads:
                for i, j in T.Parallel(valid_block_h, dim):
                    if hid * valid_block_h + i < heads:
                        Output[bid, hid * valid_block_h + i, j] = O_shared[i, j]
            else:
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
    # Shape-only heuristic; see the module comment. This is not measured.
    num_split = _select_num_split(batch, heads, kv_ctx, dim)
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
