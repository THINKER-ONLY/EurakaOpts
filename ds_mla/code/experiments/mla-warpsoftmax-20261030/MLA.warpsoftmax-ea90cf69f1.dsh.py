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

    # Experimental layout/communication path: only full 16-row head tiles with
    # split decode and long contexts. Any head-tail shape keeps the original
    # split path, and no-split decoding is untouched.
    warp_softmax = (
        pv_specialize
        and (int(heads) % _HEAD_TILE) == 0
        and num_split > 1
        and int(seqlen_kv) >= 8192
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

                pv_b[4] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[5] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[6] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 1 * 16 + lane % 16,
                ]
                pv_b[7] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 1 * 16 + lane % 16,
                ]

                pv_b[8] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[9] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[10] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 2 * 16 + lane % 16,
                ]
                pv_b[11] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 2 * 16 + lane % 16,
                ]

                pv_b[12] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[13] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[14] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 3 * 16 + lane % 16,
                ]
                pv_b[15] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 3 * 16 + lane % 16,
                ]

                pv_b[16] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[17] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[18] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 4 * 16 + lane % 16,
                ]
                pv_b[19] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 4 * 16 + lane % 16,
                ]

                pv_b[20] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[21] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[22] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 5 * 16 + lane % 16,
                ]
                pv_b[23] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 5 * 16 + lane % 16,
                ]

                pv_b[24] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[25] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[26] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 6 * 16 + lane % 16,
                ]
                pv_b[27] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 6 * 16 + lane % 16,
                ]

                pv_b[28] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[29] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[30] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 7 * 16 + lane % 16,
                ]
                pv_b[31] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 7 * 16 + lane % 16,
                ]

                pv_b[32] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[33] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[34] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 8 * 16 + lane % 16,
                ]
                pv_b[35] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 8 * 16 + lane % 16,
                ]

                pv_b[36] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[37] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[38] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 9 * 16 + lane % 16,
                ]
                pv_b[39] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 9 * 16 + lane % 16,
                ]

                pv_b[40] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[41] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[42] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 10 * 16 + lane % 16,
                ]
                pv_b[43] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 10 * 16 + lane % 16,
                ]

                pv_b[44] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[45] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[46] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 11 * 16 + lane % 16,
                ]
                pv_b[47] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 11 * 16 + lane % 16,
                ]

                pv_b[48] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[49] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[50] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 12 * 16 + lane % 16,
                ]
                pv_b[51] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 12 * 16 + lane % 16,
                ]

                pv_b[52] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[53] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[54] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 13 * 16 + lane % 16,
                ]
                pv_b[55] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 13 * 16 + lane % 16,
                ]

                pv_b[56] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[57] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[58] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 14 * 16 + lane % 16,
                ]
                pv_b[59] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 14 * 16 + lane % 16,
                ]

                pv_b[60] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 0,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[61] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 1,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[62] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 2,
                    warp * 256 + 15 * 16 + lane % 16,
                ]
                pv_b[63] = KV_shared[
                    pv_ki * 16 + (lane // 16) * 4 + 3,
                    warp * 256 + 15 * 16 + lane % 16,
                ]

                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 0, pv_a.data, 0, acc_o.data, 0, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 1, pv_a.data, 0, acc_o.data, 1, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 2, pv_a.data, 0, acc_o.data, 2, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 3, pv_a.data, 0, acc_o.data, 3, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 4, pv_a.data, 0, acc_o.data, 4, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 5, pv_a.data, 0, acc_o.data, 5, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 6, pv_a.data, 0, acc_o.data, 6, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 7, pv_a.data, 0, acc_o.data, 7, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 8, pv_a.data, 0, acc_o.data, 8, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 9, pv_a.data, 0, acc_o.data, 9, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 10, pv_a.data, 0, acc_o.data, 10, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 11, pv_a.data, 0, acc_o.data, 11, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 12, pv_a.data, 0, acc_o.data, 12, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 13, pv_a.data, 0, acc_o.data, 13, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 14, pv_a.data, 0, acc_o.data, 14, dtype="float32x4")
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", pv_b.data, 15, pv_a.data, 0, acc_o.data, 15, dtype="float32x4")
            T.sync_threads()

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

            if pv_specialize:
                pv_a = T.alloc_local([4], dtype)
                pv_b = T.alloc_local([64], dtype)
                T.annotate_layout({
                    acc_o: T.Fragment(
                        [16, 512],
                        forward_thread_fn=lambda i, j: (j // 256) * 64 + ((j % 16) // 4) * 16 + i,
                        forward_index_fn=lambda i, j: ((j % 256) // 16) * 4 + j % 4,
                    )
                })
                tx_s = T.get_thread_binding(0)
                lane_s = tx_s % 64
                warp_s = tx_s // 64

            if warp_softmax:
                # Warp-local FP32 softmax scratch. Only soft_s carries an
                # explicit thread mapping; every row reduction fragment is
                # inferred consistently from it.
                soft_s = T.alloc_fragment([valid_block_h, block_n], accum_dtype)
                sm_max = T.alloc_fragment([valid_block_h], accum_dtype)
                sm_max_prev = T.alloc_fragment([valid_block_h], accum_dtype)
                sm_scale = T.alloc_fragment([valid_block_h], accum_dtype)
                sm_sum = T.alloc_fragment([valid_block_h], accum_dtype)
                sm_logsum = T.alloc_fragment([valid_block_h], accum_dtype)
                score_exchange = T.alloc_shared(
                    [valid_block_h, block_n], accum_dtype
                )
                scale_shared = T.alloc_shared([valid_block_h], accum_dtype)
                denom_shared = T.alloc_shared([valid_block_h], accum_dtype)
                lse_shared = T.alloc_shared([valid_block_h], accum_dtype)
                T.annotate_layout({
                    soft_s: T.Fragment(
                        [valid_block_h, block_n],
                        forward_thread_fn=lambda i, j: (i // 8) * 64
                        + (j // 4) * 8
                        + i % 8,
                        forward_index_fn=lambda i, j: j % 4,
                    )
                })

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

            if warp_softmax:
                T.fill(sm_logsum, 0)
                T.fill(sm_max, -T.infinity(accum_dtype))

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

                if warp_softmax:
                    # Move the original FullCol FP32 QK fragment through
                    # shared memory into the warp-local row layout.
                    T.copy(acc_s, score_exchange)
                    T.sync_threads()
                    T.copy(score_exchange, soft_s)

                    T.copy(sm_max, sm_max_prev)
                    T.fill(sm_max, -T.infinity(accum_dtype))
                    T.reduce_max(soft_s, sm_max, dim=1, clear=False)
                    for i in T.Parallel(valid_block_h):
                        sm_max[i] = T.max(sm_max[i], sm_max_prev[i])
                    for i in T.Parallel(valid_block_h):
                        sm_scale[i] = T.exp2(
                            sm_max_prev[i] * scale - sm_max[i] * scale
                        )
                    for i, j in T.Parallel(valid_block_h, block_n):
                        soft_s[i, j] = T.exp2(
                            soft_s[i, j] * scale - sm_max[i] * scale
                        )

                    T.reduce_sum(soft_s, sm_sum, dim=1)
                    T.copy(soft_s, S_shared)
                    for i in T.Parallel(valid_block_h):
                        sm_logsum[i] = sm_logsum[i] * sm_scale[i] + sm_sum[i]
                    T.copy(sm_scale, scale_shared)
                    T.sync_threads()
                    if loop_range_split > 1:
                        # A non-first tile with an unchanged running max has
                        # scale exactly 1.0; then the rescale is a no-op.
                        for i, j in T.Parallel(valid_block_h, dim):
                            if scale_shared[i] != 1.0:
                                acc_o[i, j] *= scale_shared[i]
                    else:
                        for i, j in T.Parallel(valid_block_h, dim):
                            acc_o[i, j] *= scale_shared[i]
                else:
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
                if pv_specialize:
                    pv_mfma(
                        S_shared,
                        KV_shared,
                        pv_a,
                        pv_b,
                        acc_o,
                        lane_s,
                        warp_s,
                    )
                else:
                    T.gemm(
                        S_shared,
                        KV_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

            if warp_softmax:
                T.copy(sm_logsum, denom_shared)
                for i in T.Parallel(valid_block_h):
                    sm_logsum[i] = T.log2(sm_logsum[i]) + sm_max[i] * scale
                T.copy(sm_logsum, lse_shared)
                T.sync_threads()
                for i, j in T.Parallel(valid_block_h, dim):
                    acc_o[i, j] /= denom_shared[i]
                T.copy(acc_o, O_shared)

                # Masked head-tail stores; aligned shapes use a plain copy.
                if pad_heads:
                    for i in T.Parallel(valid_block_h):
                        if hid * valid_block_h + i < heads:
                            glse[bid, hid * valid_block_h + i, bz] = lse_shared[i]
                    for i, j in T.Parallel(valid_block_h, dim):
                        if hid * valid_block_h + i < heads:
                            output_partial[
                                bid, hid * valid_block_h + i, bz, j
                            ] = O_shared[i, j]
                else:
                    for i in T.Parallel(valid_block_h):
                        glse[bid, hid * valid_block_h + i, bz] = lse_shared[i]
                    T.copy(
                        O_shared,
                        output_partial[
                            bid,
                            hid * valid_block_h : (hid + 1) * valid_block_h,
                            bz,
                            :,
                        ],
                    )
            else:
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

            if pv_specialize:
                pv_a = T.alloc_local([4], dtype)
                pv_b = T.alloc_local([64], dtype)
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
