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

import tilelang
import tilelang.language as T


# Smallest row tile kept for the Q/KV GEMMs. TileLang's MMA paths expect a
# legal tensor-core tile; 16 rows is the conservative choice used here.
_HEAD_TILE = 16

# Shape-only split dispatch constants.
_SPLIT_MIN_CTX = 2048
_SPLIT_SHORT_CTX = 4096
_SPLIT_TARGET_LONG = 64
_SPLIT_TARGET_SHORT = 64
_SPLIT_MAX = 128
_SPLIT_MIN_KEYS = 32
_PARTIAL_ELEM_BUDGET = 8388608

# Parallel split-merge tiling constants.
_MERGE_TILE_DIM = 64
_MERGE_MIN_SPLIT = 32
_MERGE_MAX_SPLIT = 128


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

    # Source-explicit PV path for the scored 16-row, 512-wide, 32-key tile.
    # Other shapes keep the original T.gemm PV path and layout inference.
    pv_specialize = (
        int(dim) == 512
        and valid_block_h == 16
        and int(block_n) == 32
    )

    if pv_specialize:

        @T.macro
        def pv_mfma(S_shared, KV_shared, pv_a, pv_b, acc_o, lane, warp):
            T.sync_threads()
            for pv_ki in T.serial(2):
                pv_a[0] = S_shared[
                    lane % 16, pv_ki * 16 + (lane // 16) * 4 + 0
                ]
                pv_a[1] = S_shared[
                    lane % 16, pv_ki * 16 + (lane // 16) * 4 + 1
                ]
                pv_a[2] = S_shared[
                    lane % 16, pv_ki * 16 + (lane // 16) * 4 + 2
                ]
                pv_a[3] = S_shared[
                    lane % 16, pv_ki * 16 + (lane // 16) * 4 + 3
                ]

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 0 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 0 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 0 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 0 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 0, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 1 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 1, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 2 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 2, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 3 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 3, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 4 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 4, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 5 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 5, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 6 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 6, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 7 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 7, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 8 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 8, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 9 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 9, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 10 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 10, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 11 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 11, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 12 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 12, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 13 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 13, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 14 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 14, dtype="float32x4")

                pv_b[0] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[1] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[2] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[3] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 15 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 15, dtype="float32x4")
            T.sync_threads()

    @T.macro
    def pbar(bar):
        prev = T.atomic_add(bar[0], 1, memory_order="acq_rel", return_prev=True)
        target = (prev // 128 + 1) * 128
        while T.atomic_add(bar[0], 0, memory_order="acq_rel", return_prev=True) < target:
            T.evaluate(0)


    @T.macro
    def pv_step(n, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp):
        pv_b[0] = KV_s[ki * 16 + jh * 4 + 0, warp * 256 + n * 16 + lane % 16]
        pv_b[1] = KV_s[ki * 16 + jh * 4 + 1, warp * 256 + n * 16 + lane % 16]
        pv_b[2] = KV_s[ki * 16 + jh * 4 + 2, warp * 256 + n * 16 + lane % 16]
        pv_b[3] = KV_s[ki * 16 + jh * 4 + 3, warp * 256 + n * 16 + lane % 16]
        T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4",
                   pv_b.data, 0, pv_a.data, 0, acc_o.data, n, dtype="float32x4")

    @T.macro
    def scale_i(acc_o, i, alpha):
        acc_o[i] *= alpha[0]

    @T.macro
    def div_i(acc_o, i, lsum):
        acc_o[i] = acc_o[i] / lsum[0]

    @T.macro
    def zero_i(acc_o, i):
        acc_o[i] = 0.0

    @T.macro
    def store_i(acc_o, i, output_partial, bid, h, bz, dim, warp, jh):
        output_partial[bid, h, bz, warp * 256 + (i // 4) * 16 + jh * 4 + i % 4] = acc_o[i]
    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split], accum_dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        # 256 threads stream Q/KV/K_pe into shared memory; the first 128 threads
        # run fully hand-written MMA paths (explicit lane indexing, no T.gemm),
        # so no warp-partition or fragment-layout inference is involved.
        with T.Kernel(batch, head_blocks, num_split, threads=256) as (bid, hid, bz):
            Q_s = T.alloc_shared([valid_block_h, dim], dtype)
            Qp_s = T.alloc_shared([valid_block_h, pe_dim], dtype)
            KV_s = T.alloc_shared([block_n, dim], dtype)
            Kp_s = T.alloc_shared([block_n, pe_dim], dtype)
            S_s = T.alloc_shared([valid_block_h, block_n], dtype)
            O_s = T.alloc_shared([valid_block_h, dim + 8], dtype)
            PS = T.alloc_shared([valid_block_h, 8], accum_dtype)
            bar = T.alloc_shared([1], T.int32)

            acc_s = T.alloc_local([4], accum_dtype)
            acc_o = T.alloc_local([64], accum_dtype)
            a_l = T.alloc_local([4], dtype)
            b_l = T.alloc_local([4], dtype)
            pv_a = T.alloc_local([4], dtype)
            pv_b = T.alloc_local([4], dtype)
            mcur = T.alloc_local([1], accum_dtype)
            lsum = T.alloc_local([1], accum_dtype)
            mloc = T.alloc_local([1], accum_dtype)
            mrow = T.alloc_local([1], accum_dtype)
            nm = T.alloc_local([1], accum_dtype)
            alpha = T.alloc_local([1], accum_dtype)
            psum = T.alloc_local([1], accum_dtype)
            prow = T.alloc_local([1], accum_dtype)

            tx = T.get_thread_binding(0)
            lane = tx % 64
            row = lane % 16
            jh = lane // 16
            warp = tx // 64
            wid = warp * 4 + jh
            cur_kv_head = 0

            if pad_heads:
                for i, j in T.Parallel(valid_block_h, dim):
                    Q_s[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
                for i, j in T.Parallel(valid_block_h, pe_dim):
                    Qp_s[i, j] = T.if_then_else(
                        hid * valid_block_h + i < heads,
                        Q_pe[bid, T.min(hid * valid_block_h + i, heads - 1), j],
                        0,
                    )
            else:
                T.copy(Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Q_s)
                T.copy(Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Qp_s)

            if tx == 0:
                bar[0] = 0
            T.sync_threads()

            if tx < 128:
                zero_i(acc_o, 0)
                zero_i(acc_o, 1)
                zero_i(acc_o, 2)
                zero_i(acc_o, 3)
                zero_i(acc_o, 4)
                zero_i(acc_o, 5)
                zero_i(acc_o, 6)
                zero_i(acc_o, 7)
                zero_i(acc_o, 8)
                zero_i(acc_o, 9)
                zero_i(acc_o, 10)
                zero_i(acc_o, 11)
                zero_i(acc_o, 12)
                zero_i(acc_o, 13)
                zero_i(acc_o, 14)
                zero_i(acc_o, 15)
                zero_i(acc_o, 16)
                zero_i(acc_o, 17)
                zero_i(acc_o, 18)
                zero_i(acc_o, 19)
                zero_i(acc_o, 20)
                zero_i(acc_o, 21)
                zero_i(acc_o, 22)
                zero_i(acc_o, 23)
                zero_i(acc_o, 24)
                zero_i(acc_o, 25)
                zero_i(acc_o, 26)
                zero_i(acc_o, 27)
                zero_i(acc_o, 28)
                zero_i(acc_o, 29)
                zero_i(acc_o, 30)
                zero_i(acc_o, 31)
                zero_i(acc_o, 32)
                zero_i(acc_o, 33)
                zero_i(acc_o, 34)
                zero_i(acc_o, 35)
                zero_i(acc_o, 36)
                zero_i(acc_o, 37)
                zero_i(acc_o, 38)
                zero_i(acc_o, 39)
                zero_i(acc_o, 40)
                zero_i(acc_o, 41)
                zero_i(acc_o, 42)
                zero_i(acc_o, 43)
                zero_i(acc_o, 44)
                zero_i(acc_o, 45)
                zero_i(acc_o, 46)
                zero_i(acc_o, 47)
                zero_i(acc_o, 48)
                zero_i(acc_o, 49)
                zero_i(acc_o, 50)
                zero_i(acc_o, 51)
                zero_i(acc_o, 52)
                zero_i(acc_o, 53)
                zero_i(acc_o, 54)
                zero_i(acc_o, 55)
                zero_i(acc_o, 56)
                zero_i(acc_o, 57)
                zero_i(acc_o, 58)
                zero_i(acc_o, 59)
                zero_i(acc_o, 60)
                zero_i(acc_o, 61)
                zero_i(acc_o, 62)
                zero_i(acc_o, 63)
                mcur[0] = -T.infinity(accum_dtype)
                lsum[0] = 0.0

            kv_start = (seqlen_kv * bz) // num_split
            kv_end = (seqlen_kv * (bz + 1)) // num_split

            for k in T.serial(loop_range_split):
                kv_block_start = kv_start + k * block_n
                T.copy(
                    KV[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    KV_s,
                )
                T.copy(
                    K_pe[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    Kp_s,
                )
                T.sync_threads()

                if tx < 128:
                    for c in T.serial(4):
                        acc_s[c] = 0.0
                    for kk in T.serial(dim // 16):
                        a_l[0] = Q_s[lane % 16, kk * 16 + jh * 4 + 0]
                        a_l[1] = Q_s[lane % 16, kk * 16 + jh * 4 + 1]
                        a_l[2] = Q_s[lane % 16, kk * 16 + jh * 4 + 2]
                        a_l[3] = Q_s[lane % 16, kk * 16 + jh * 4 + 3]
                        b_l[0] = KV_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 0]
                        b_l[1] = KV_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 1]
                        b_l[2] = KV_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 2]
                        b_l[3] = KV_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 3]
                        T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4",
                                   b_l.data, 0, a_l.data, 0, acc_s.data, 0, dtype="float32x4")
                    for kk in T.serial(pe_dim // 16):
                        a_l[0] = Qp_s[lane % 16, kk * 16 + jh * 4 + 0]
                        a_l[1] = Qp_s[lane % 16, kk * 16 + jh * 4 + 1]
                        a_l[2] = Qp_s[lane % 16, kk * 16 + jh * 4 + 2]
                        a_l[3] = Qp_s[lane % 16, kk * 16 + jh * 4 + 3]
                        b_l[0] = Kp_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 0]
                        b_l[1] = Kp_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 1]
                        b_l[2] = Kp_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 2]
                        b_l[3] = Kp_s[warp * 16 + lane % 16, kk * 16 + jh * 4 + 3]
                        T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4",
                                   b_l.data, 0, a_l.data, 0, acc_s.data, 0, dtype="float32x4")

                    if needs_split_mask:
                        for c in T.serial(4):
                            if kv_block_start + warp * 16 + jh * 4 + c >= kv_end:
                                acc_s[c] = -T.infinity(accum_dtype)

                    psum[0] = 0.0
                    for c in T.serial(4):
                        psum[0] += T.exp2(acc_s[c] * scale)
                    for c in T.serial(4):
                        S_s[lane % 16, warp * 16 + jh * 4 + c] = T.exp2(acc_s[c] * scale)
                    PS[row, wid] = psum[0]
                T.sync_threads()
                if tx < 128:

                    prow[0] = PS[row, 0]
                    for w2 in T.serial(7):
                        prow[0] += PS[row, w2 + 1]
                    lsum[0] += prow[0]

                    for ki in T.serial(block_n // 16):
                        pv_a[0] = S_s[lane % 16, ki * 16 + jh * 4 + 0]
                        pv_a[1] = S_s[lane % 16, ki * 16 + jh * 4 + 1]
                        pv_a[2] = S_s[lane % 16, ki * 16 + jh * 4 + 2]
                        pv_a[3] = S_s[lane % 16, ki * 16 + jh * 4 + 3]
                        pv_step(0, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(1, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(2, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(3, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(4, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(5, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(6, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(7, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(8, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(9, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(10, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(11, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(12, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(13, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(14, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                        pv_step(15, KV_s, pv_a, pv_b, acc_o, ki, jh, lane, warp)
                T.sync_threads()

            if tx < 128:
                div_i(acc_o, 0, lsum)
                div_i(acc_o, 1, lsum)
                div_i(acc_o, 2, lsum)
                div_i(acc_o, 3, lsum)
                div_i(acc_o, 4, lsum)
                div_i(acc_o, 5, lsum)
                div_i(acc_o, 6, lsum)
                div_i(acc_o, 7, lsum)
                div_i(acc_o, 8, lsum)
                div_i(acc_o, 9, lsum)
                div_i(acc_o, 10, lsum)
                div_i(acc_o, 11, lsum)
                div_i(acc_o, 12, lsum)
                div_i(acc_o, 13, lsum)
                div_i(acc_o, 14, lsum)
                div_i(acc_o, 15, lsum)
                div_i(acc_o, 16, lsum)
                div_i(acc_o, 17, lsum)
                div_i(acc_o, 18, lsum)
                div_i(acc_o, 19, lsum)
                div_i(acc_o, 20, lsum)
                div_i(acc_o, 21, lsum)
                div_i(acc_o, 22, lsum)
                div_i(acc_o, 23, lsum)
                div_i(acc_o, 24, lsum)
                div_i(acc_o, 25, lsum)
                div_i(acc_o, 26, lsum)
                div_i(acc_o, 27, lsum)
                div_i(acc_o, 28, lsum)
                div_i(acc_o, 29, lsum)
                div_i(acc_o, 30, lsum)
                div_i(acc_o, 31, lsum)
                div_i(acc_o, 32, lsum)
                div_i(acc_o, 33, lsum)
                div_i(acc_o, 34, lsum)
                div_i(acc_o, 35, lsum)
                div_i(acc_o, 36, lsum)
                div_i(acc_o, 37, lsum)
                div_i(acc_o, 38, lsum)
                div_i(acc_o, 39, lsum)
                div_i(acc_o, 40, lsum)
                div_i(acc_o, 41, lsum)
                div_i(acc_o, 42, lsum)
                div_i(acc_o, 43, lsum)
                div_i(acc_o, 44, lsum)
                div_i(acc_o, 45, lsum)
                div_i(acc_o, 46, lsum)
                div_i(acc_o, 47, lsum)
                div_i(acc_o, 48, lsum)
                div_i(acc_o, 49, lsum)
                div_i(acc_o, 50, lsum)
                div_i(acc_o, 51, lsum)
                div_i(acc_o, 52, lsum)
                div_i(acc_o, 53, lsum)
                div_i(acc_o, 54, lsum)
                div_i(acc_o, 55, lsum)
                div_i(acc_o, 56, lsum)
                div_i(acc_o, 57, lsum)
                div_i(acc_o, 58, lsum)
                div_i(acc_o, 59, lsum)
                div_i(acc_o, 60, lsum)
                div_i(acc_o, 61, lsum)
                div_i(acc_o, 62, lsum)
                div_i(acc_o, 63, lsum)
                if wid == 0:
                    glse[bid, hid * valid_block_h + row, bz] = T.log2(lsum[0])
                for idx in T.serial(64):
                    col = warp * 256 + (idx // 4) * 16 + jh * 4 + idx % 4
                    O_s[row, col] = T.cast(acc_o[idx], dtype)
            T.sync_threads()
            if tx < 128:
                if pad_heads:
                    for i, j in T.Parallel(valid_block_h, dim):
                        if hid * valid_block_h + i < heads:
                            output_partial[
                                bid, hid * valid_block_h + i, bz, j
                            ] = O_s[i, j]
                else:
                    for i, j in T.Parallel(valid_block_h, dim):
                        output_partial[bid, hid * valid_block_h + i, bz, j] = O_s[i, j]

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

            if pv_specialize:
                pv_a = T.alloc_local([4], dtype)
                pv_b = T.alloc_local([4], dtype)
                T.annotate_layout({
                    acc_o: T.Fragment(
                        [16, 512],
                        forward_thread_fn=lambda i, j: (j // 256) * 64 + ((j % 16) // 4) * 16 + i,
                        forward_index_fn=lambda i, j: ((j % 256) // 16) * 4 + j % 4,
                    )
                })
                tx_ns = T.get_thread_binding(0)
                lane_ns = tx_ns % 64
                warp_ns = tx_ns // 64

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
                if pv_specialize:
                    pv_mfma(
                        S_shared,
                        KV_shared,
                        pv_a,
                        pv_b,
                        acc_o,
                        lane_ns,
                        warp_ns,
                    )
                else:
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
_LAST_KEY = None
_LAST_KERNEL = None


def _build_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    block_n = 32
    block_h = _usable_block_h(heads, min(16, heads // kv_heads))
    num_split = _select_num_split(batch, heads, kv_ctx, dim)
    softmax_scale = (dim + pe_dim) ** -0.5
    return flashattn(
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
    global _LAST_KEY, _LAST_KERNEL
    key = (batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    if key == _LAST_KEY:
        kernel = _LAST_KERNEL
    else:
        kernel = _KERNEL_CACHE.get(key)
        if kernel is None:
            kernel = _build_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim)
            _KERNEL_CACHE[key] = kernel
        _LAST_KEY = key
        _LAST_KERNEL = kernel
    kernel(q, q_pe, kv, k_pe, output)
