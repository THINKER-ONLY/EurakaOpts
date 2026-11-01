# Pure-TileLang MLA decode kernel (single-file production module).
#
# Split dispatch is shape-only. Contexts below 2048 use the no-split path.
# Larger contexts derive a split count from the CTA-grid target (128 CTAs for
# short contexts or multi-CTA batches, 64 for longer single-CTA contexts),
# bounded by the 32-key minimum split length and a 16 MiB FP16 output-partial
# scratch budget.
#
# Fixed assumptions: contiguous FP16 q/q_pe/kv/k_pe/output tensors with
# kv_head_num == 1, in-place output, 128-thread blocks, and block_n == 32.
# Validation evidence is maintained externally by artifact SHA; no benchmark
# result or acceptance claim is embedded here.
#
# The guarded long-context experiment separates score production (QK + PE,
# full-sequence FP32 logits in global scratch) from PV accumulation (128-wide
# output-dimension tiles). It is enabled only at compile time for dim == 512,
# pe_dim == 64, num_split > 1, seqlen_kv >= 8192, and a bounded logits scratch.
# Every other shape uses the original paths below unchanged.

import tilelang
import tilelang.language as T


# Smallest row tile kept for the Q/KV GEMMs. TileLang's MMA paths expect a
# legal tensor-core tile; 16 rows is the conservative choice used here.
_HEAD_TILE = 16

# Shape-only split dispatch constants.
_SPLIT_MIN_CTX = 2048
_SPLIT_SHORT_CTX = 4096
_SPLIT_TARGET_LONG = 64
_SPLIT_TARGET_SHORT = 128
_SPLIT_MAX = 128
_SPLIT_MIN_KEYS = 32
_PARTIAL_ELEM_BUDGET = 8388608

# Parallel split-merge tiling constants.
_MERGE_TILE_DIM = 64
_MERGE_MIN_SPLIT = 32
_MERGE_MAX_SPLIT = 128

# Long-context QK/PV decoupling constants.
_DECOUPLE_DIM = 512
_DECOUPLE_PE_DIM = 64
_DECOUPLE_MIN_CTX = 8192
_DECOUPLE_PV_BLOCK_N = 64
_DECOUPLE_PV_DIM_TILE = 128
_DECOUPLE_LOGIT_BUDGET_BYTES = 67108864


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
    """Choose a split count from the shape-only CTA-grid heuristic.

    Short contexts (one or a few KV tiles per split) and multi-CTA base grids
    use a 128-CTA target; longer contexts with a single base CTA use a 64-CTA
    target. The result is bounded by the maximum split count, a minimum of 32
    keys per split, and the FP16 output-partial scratch budget.
    """
    ctx = int(kv_ctx)
    if ctx < _SPLIT_MIN_CTX:
        return 1

    head_blocks = (int(heads) + _HEAD_TILE - 1) // _HEAD_TILE
    base_grid = int(batch) * head_blocks
    if base_grid <= 0:
        return 1
    if ctx <= _SPLIT_SHORT_CTX or base_grid > 1:
        target_ctas = _SPLIT_TARGET_SHORT
    else:
        target_ctas = _SPLIT_TARGET_LONG

    n = (target_ctas + base_grid - 1) // base_grid
    n = min(n, _SPLIT_MAX)
    n = min(n, ctx // _SPLIT_MIN_KEYS)
    n = min(
        n,
        _PARTIAL_ELEM_BUDGET
        // max(1, int(batch) * int(heads) * int(dim)),
    )
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

    # Host-side metadata for the tiled split merge. The padded split count is
    # the next power of two so the merge fragment has a static shape. Split
    # counts above the threshold keep the original serial merge fallback.
    merge_tile = _MERGE_TILE_DIM
    merge_num_tiles = (dim + merge_tile - 1) // merge_tile
    merge_dim_tail = (dim % merge_tile) != 0
    merge_padded_split = 1
    while merge_padded_split < num_split:
        merge_padded_split *= 2

    # Compile-time-only guard for the QK/PV decoupled experiment. The scratch
    # is the only added global buffer: batch * heads * seqlen_kv FP32 logits.
    score_scratch_bytes = int(batch) * int(heads) * int(seqlen_kv) * 4
    use_decoupled = (
        int(dim) == _DECOUPLE_DIM
        and int(pe_dim) == _DECOUPLE_PE_DIM
        and int(num_split) > 1
        and int(seqlen_kv) >= _DECOUPLE_MIN_CTX
        and score_scratch_bytes <= _DECOUPLE_LOGIT_BUDGET_BYTES
    )

    # Compile-time host metadata for the decoupled stages.
    score_blocks = (seqlen_kv + block_n - 1) // block_n
    score_tail_mask = (seqlen_kv % block_n) != 0
    pv_block_n = _DECOUPLE_PV_BLOCK_N
    pv_dim_tile = _DECOUPLE_PV_DIM_TILE
    pv_dim_tiles = dim // pv_dim_tile
    pv_loop_range = (split_len_max + pv_block_n - 1) // pv_block_n
    pv_needs_mask = not (
        (seqlen_kv % num_split == 0)
        and (((seqlen_kv // num_split) % pv_block_n) == 0)
    )
    pv_masked_logits = pad_heads or pv_needs_mask
    pv_masked_kv = pv_needs_mask

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

        # 128 threads = 2 x 64-lane warps. The M16,N32 QK GEMM below uses a
        # legal 1x2 MACA MMA partition; 256 threads would require 1x4 and fail.
        with T.Kernel(
            batch,
            head_blocks,
            num_split,
            threads=128,
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

            # num_stages=0 matches the passing baseline and avoids introducing
            # shared-memory double buffering as an additional unknown.
            for k in T.Pipelined(loop_range_split, num_stages=0):
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
                if loop_range_split > 1:
                    # A non-first tile with an unchanged running max has
                    # scores_scale exactly 1.0; then the rescale is a no-op.
                    for i, j in T.Parallel(valid_block_h, dim):
                        if scores_scale[i] != 1.0:
                            acc_o[i, j] *= scores_scale[i]
                else:
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

        # Parallel tiled merge of the split partials. Each CTA owns one
        # (head, output-dim tile, batch) triple, so the split reduction is
        # spread across CTAs while output-dimension loads stay coalesced. The
        # serial merge remains below the minimum split count, where small
        # reductions are faster serially, and above the maximum, where the
        # padded [split, tile] fragment would be too large.
        if num_split < _MERGE_MIN_SPLIT or num_split > _MERGE_MAX_SPLIT:
            # Stable bounded serial merge, retained as the public-API fallback.
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
        else:
            with T.Kernel(
                heads,
                merge_num_tiles,
                batch,
                threads=64,
            ) as (hid, did, bz):
                lse_frag = T.alloc_fragment([1, merge_padded_split], accum_dtype)
                lse_max = T.alloc_fragment([1], accum_dtype)
                lse_sum = T.alloc_fragment([1], accum_dtype)
                w_shared = T.alloc_shared([merge_padded_split], accum_dtype)
                po_frag = T.alloc_fragment(
                    [merge_padded_split, merge_tile], accum_dtype
                )
                o_frag = T.alloc_fragment([merge_tile], accum_dtype)

                # Load all split LSE values in parallel. Inactive padded rows
                # are set to -inf and later receive exactly zero weight.
                for i, j in T.Parallel(1, merge_padded_split):
                    lse_frag[i, j] = T.if_then_else(
                        j < num_split,
                        glse[bz, hid, T.min(j, num_split - 1)],
                        -T.infinity(accum_dtype),
                    )

                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(lse_frag, lse_max, dim=1, clear=False)

                for i, j in T.Parallel(1, merge_padded_split):
                    lse_frag[i, j] = T.exp2(lse_frag[i, j] - lse_max[i])

                T.reduce_sum(lse_frag, lse_sum, dim=1)

                # Normalized weights are shared so the split index can be
                # broadcast into the [split, tile] accumulation fragment. At
                # least one real split exists, so lse_sum is positive.
                for i, j in T.Parallel(1, merge_padded_split):
                    w_shared[j] = lse_frag[i, j] / lse_sum[i]

                if merge_dim_tail:
                    for k, i in T.Parallel(merge_padded_split, merge_tile):
                        col = did * merge_tile + i
                        po_frag[k, i] = T.if_then_else(
                            k < num_split,
                            T.if_then_else(
                                col < dim,
                                output_partial[
                                    bz,
                                    hid,
                                    T.min(k, num_split - 1),
                                    T.min(col, dim - 1),
                                ],
                                0,
                            ),
                            0,
                        ) * w_shared[k]
                else:
                    for k, i in T.Parallel(merge_padded_split, merge_tile):
                        po_frag[k, i] = T.if_then_else(
                            k < num_split,
                            output_partial[
                                bz,
                                hid,
                                T.min(k, num_split - 1),
                                did * merge_tile + i,
                            ],
                            0,
                        ) * w_shared[k]

                T.reduce_sum(po_frag, o_frag, dim=0)

                if merge_dim_tail:
                    for i in T.Parallel(merge_tile):
                        if did * merge_tile + i < dim:
                            Output[bz, hid, did * merge_tile + i] = o_frag[i]
                else:
                    for i in T.Parallel(merge_tile):
                        Output[bz, hid, did * merge_tile + i] = o_frag[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # Same legal 128-thread / 2-warp configuration as the passing baseline.
        with T.Kernel(
            head_blocks,
            batch,
            threads=128,
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

            # num_stages=0 matches the passing baseline and avoids introducing
            # shared-memory double buffering as an additional unknown.
            for k in T.Pipelined(loop_range_full, num_stages=0):
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
                if loop_range_full > 1:
                    # A non-first tile with an unchanged running max has
                    # scores_scale exactly 1.0; then the rescale is a no-op.
                    for i, j in T.Parallel(valid_block_h, dim):
                        if scores_scale[i] != 1.0:
                            acc_o[i, j] *= scores_scale[i]
                else:
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

    if use_decoupled:

        @T.prim_func
        def main_split_decoupled(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            # Added scratch: unscaled QK+PE logits, one FP32 plane per head.
            logits = T.alloc_global([batch, heads, seqlen_kv], accum_dtype)
            glse = T.alloc_global([batch, heads, num_split], accum_dtype)
            output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

            # Stage 1: produce raw scores. Each CTA owns one 32-key block and
            # all 16 query rows for one head tile. Q is re-read per sequence
            # block; there is no output accumulator and no loop-carried state.
            with T.Kernel(
                batch,
                head_blocks,
                score_blocks,
                threads=128,
            ) as (bid, hid, sb):
                Q_shared = T.alloc_shared([valid_block_h, dim], dtype)
                Q_pe_shared = T.alloc_shared([valid_block_h, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_n, dim], dtype)
                K_pe_shared = T.alloc_shared([block_n, pe_dim], dtype)
                acc_s = T.alloc_fragment([valid_block_h, block_n], accum_dtype)

                cur_kv_head = 0

                # Masked head-tail load; identical shape handling to baseline.
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

                score_block_start = sb * block_n

                # Tail-safe KV loads. Aligned shapes take the coalesced copy.
                if score_tail_mask:
                    for i, j in T.Parallel(block_n, dim):
                        kv_idx = score_block_start + i
                        KV_shared[i, j] = T.if_then_else(
                            kv_idx < seqlen_kv,
                            KV[
                                bid,
                                T.min(kv_idx, seqlen_kv - 1),
                                cur_kv_head,
                                j,
                            ],
                            0,
                        )
                    for i, j in T.Parallel(block_n, pe_dim):
                        kv_idx = score_block_start + i
                        K_pe_shared[i, j] = T.if_then_else(
                            kv_idx < seqlen_kv,
                            K_pe[
                                bid,
                                T.min(kv_idx, seqlen_kv - 1),
                                cur_kv_head,
                                j,
                            ],
                            0,
                        )
                else:
                    T.copy(
                        KV[
                            bid,
                            score_block_start : score_block_start + block_n,
                            cur_kv_head,
                            :,
                        ],
                        KV_shared,
                    )
                    T.copy(
                        K_pe[
                            bid,
                            score_block_start : score_block_start + block_n,
                            cur_kv_head,
                            :,
                        ],
                        K_pe_shared,
                    )

                # Raw QK + PE, exactly the baseline FP32 score formation.
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

                if score_tail_mask:
                    for i, j in T.Parallel(valid_block_h, block_n):
                        acc_s[i, j] = T.if_then_else(
                            score_block_start + j < seqlen_kv,
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )

                # No scale is applied before the global store. Only valid head
                # rows and valid keys are written when either boundary is tail.
                if pad_heads or score_tail_mask:
                    for i, j in T.Parallel(valid_block_h, block_n):
                        if hid * valid_block_h + i < heads:
                            if score_block_start + j < seqlen_kv:
                                logits[
                                    bid,
                                    hid * valid_block_h + i,
                                    score_block_start + j,
                                ] = acc_s[i, j]
                else:
                    T.copy(
                        acc_s,
                        logits[
                            bid,
                            hid * valid_block_h : (hid + 1) * valid_block_h,
                            score_block_start : score_block_start + block_n,
                        ],
                    )

            # Stage 2: PV and stable online softmax. The output-dimension tile
            # is flattened into the head-block grid axis to stay within the
            # three T.Kernel dimensions. All four value CTAs repeat the same
            # cheap softmax over the produced logits; none repeat QK.
            with T.Kernel(
                batch,
                head_blocks * pv_dim_tiles,
                num_split,
                threads=128,
            ) as (bid, hd, bz):
                pv_hid = hd // pv_dim_tiles
                pv_did = hd % pv_dim_tiles

                S_shared = T.alloc_shared([valid_block_h, pv_block_n], dtype)
                KV_shared = T.alloc_shared([pv_block_n, pv_dim_tile], dtype)
                O_shared = T.alloc_shared([valid_block_h, pv_dim_tile], dtype)
                acc_s = T.alloc_fragment([valid_block_h, pv_block_n], accum_dtype)
                acc_o = T.alloc_fragment([valid_block_h, pv_dim_tile], accum_dtype)
                scores_max = T.alloc_fragment([valid_block_h], accum_dtype)
                scores_max_prev = T.alloc_fragment([valid_block_h], accum_dtype)
                scores_scale = T.alloc_fragment([valid_block_h], accum_dtype)
                scores_sum = T.alloc_fragment([valid_block_h], accum_dtype)
                logsum = T.alloc_fragment([valid_block_h], accum_dtype)

                cur_kv_head = 0

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                pv_kv_start = (seqlen_kv * bz) // num_split
                pv_kv_end = (seqlen_kv * (bz + 1)) // num_split

                for k in T.Pipelined(pv_loop_range, num_stages=0):
                    kv_block_start = pv_kv_start + k * pv_block_n

                    # Load already-produced raw logits into a FP32 fragment.
                    if pv_masked_logits:
                        for i, j in T.Parallel(valid_block_h, pv_block_n):
                            head_idx = pv_hid * valid_block_h + i
                            key_idx = kv_block_start + j
                            acc_s[i, j] = T.if_then_else(
                                head_idx < heads,
                                T.if_then_else(
                                    key_idx < seqlen_kv,
                                    logits[
                                        bid,
                                        T.min(head_idx, heads - 1),
                                        T.min(key_idx, seqlen_kv - 1),
                                    ],
                                    0.0,
                                ),
                                0.0,
                            )
                    else:
                        for i, j in T.Parallel(valid_block_h, pv_block_n):
                            acc_s[i, j] = logits[
                                bid,
                                pv_hid * valid_block_h + i,
                                kv_block_start + j,
                            ]

                    # Mask split tails and any sequence tail. Unaligned shapes
                    # only; aligned shapes skip the loop entirely.
                    if pv_needs_mask:
                        for i, j in T.Parallel(valid_block_h, pv_block_n):
                            acc_s[i, j] = T.if_then_else(
                                kv_block_start + j < pv_kv_end,
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
                    for i, j in T.Parallel(valid_block_h, pv_block_n):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    for i in T.Parallel(valid_block_h):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    if pv_loop_range > 1:
                        for i, j in T.Parallel(valid_block_h, pv_dim_tile):
                            if scores_scale[i] != 1.0:
                                acc_o[i, j] *= scores_scale[i]
                    else:
                        for i, j in T.Parallel(valid_block_h, pv_dim_tile):
                            acc_o[i, j] *= scores_scale[i]

                    # Stage 2's B tile is 128 wide, not the full 512.
                    if pv_masked_kv:
                        for i, j in T.Parallel(pv_block_n, pv_dim_tile):
                            key_idx = kv_block_start + i
                            KV_shared[i, j] = T.if_then_else(
                                key_idx < seqlen_kv,
                                KV[
                                    bid,
                                    T.min(key_idx, seqlen_kv - 1),
                                    cur_kv_head,
                                    pv_did * pv_dim_tile + j,
                                ],
                                0,
                            )
                    else:
                        T.copy(
                            KV[
                                bid,
                                kv_block_start : kv_block_start + pv_block_n,
                                cur_kv_head,
                                pv_did * pv_dim_tile : (pv_did + 1) * pv_dim_tile,
                            ],
                            KV_shared,
                        )

                    T.gemm(
                        S_shared,
                        KV_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                for i, j in T.Parallel(valid_block_h, pv_dim_tile):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(valid_block_h):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(acc_o, O_shared)

                # Only output tile zero writes the per-split FP32 LSE.
                for i in T.Parallel(valid_block_h):
                    if pv_did == 0:
                        if pv_hid * valid_block_h + i < heads:
                            glse[bid, pv_hid * valid_block_h + i, bz] = logsum[i]

                # FP16 partial output in the existing split-partial layout.
                if pad_heads:
                    for i, j in T.Parallel(valid_block_h, pv_dim_tile):
                        if pv_hid * valid_block_h + i < heads:
                            output_partial[
                                bid,
                                pv_hid * valid_block_h + i,
                                bz,
                                pv_did * pv_dim_tile + j,
                            ] = O_shared[i, j]
                else:
                    T.copy(
                        O_shared,
                        output_partial[
                            bid,
                            pv_hid * valid_block_h : (pv_hid + 1) * valid_block_h,
                            bz,
                            pv_did * pv_dim_tile : (pv_did + 1) * pv_dim_tile,
                        ],
                    )

            # Stage 3: the existing split merge, copied unchanged in behavior.
            if num_split < _MERGE_MIN_SPLIT or num_split > _MERGE_MAX_SPLIT:
                # Stable bounded serial merge, retained as the public-API fallback.
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
            else:
                with T.Kernel(
                    heads,
                    merge_num_tiles,
                    batch,
                    threads=64,
                ) as (hid, did, bz):
                    lse_frag = T.alloc_fragment([1, merge_padded_split], accum_dtype)
                    lse_max = T.alloc_fragment([1], accum_dtype)
                    lse_sum = T.alloc_fragment([1], accum_dtype)
                    w_shared = T.alloc_shared([merge_padded_split], accum_dtype)
                    po_frag = T.alloc_fragment(
                        [merge_padded_split, merge_tile], accum_dtype
                    )
                    o_frag = T.alloc_fragment([merge_tile], accum_dtype)

                    for i, j in T.Parallel(1, merge_padded_split):
                        lse_frag[i, j] = T.if_then_else(
                            j < num_split,
                            glse[bz, hid, T.min(j, num_split - 1)],
                            -T.infinity(accum_dtype),
                        )

                    T.fill(lse_max, -T.infinity(accum_dtype))
                    T.reduce_max(lse_frag, lse_max, dim=1, clear=False)

                    for i, j in T.Parallel(1, merge_padded_split):
                        lse_frag[i, j] = T.exp2(lse_frag[i, j] - lse_max[i])

                    T.reduce_sum(lse_frag, lse_sum, dim=1)

                    for i, j in T.Parallel(1, merge_padded_split):
                        w_shared[j] = lse_frag[i, j] / lse_sum[i]

                    if merge_dim_tail:
                        for k, i in T.Parallel(merge_padded_split, merge_tile):
                            col = did * merge_tile + i
                            po_frag[k, i] = T.if_then_else(
                                k < num_split,
                                T.if_then_else(
                                    col < dim,
                                    output_partial[
                                        bz,
                                        hid,
                                        T.min(k, num_split - 1),
                                        T.min(col, dim - 1),
                                    ],
                                    0,
                                ),
                                0,
                            ) * w_shared[k]
                    else:
                        for k, i in T.Parallel(merge_padded_split, merge_tile):
                            po_frag[k, i] = T.if_then_else(
                                k < num_split,
                                output_partial[
                                    bz,
                                    hid,
                                    T.min(k, num_split - 1),
                                    did * merge_tile + i,
                                ],
                                0,
                            ) * w_shared[k]

                    T.reduce_sum(po_frag, o_frag, dim=0)

                    if merge_dim_tail:
                        for i in T.Parallel(merge_tile):
                            if did * merge_tile + i < dim:
                                Output[bz, hid, did * merge_tile + i] = o_frag[i]
                    else:
                        for i in T.Parallel(merge_tile):
                            Output[bz, hid, did * merge_tile + i] = o_frag[i]

        return main_split_decoupled

    if num_split > 1:
        return main_split
    return main_no_split


_KERNEL_CACHE = {}


def _get_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    assert kv_heads == 1, "kv_head_num must be 1"
    block_n = 32
    block_h = _usable_block_h(heads, min(16, heads // kv_heads))
    # Shape-only dispatch; see module comment.
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
