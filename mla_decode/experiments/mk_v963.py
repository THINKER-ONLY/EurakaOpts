"""v963: issue the merge's partial-sum loads BEFORE the LSE reduction so the
two independent post-spin memory chains share one latency instead of two."""
import re

SRC = "/root/lzy/workbench/v960_split104_aligned.py"
DST = "/root/lzy/workbench/v963_mergeearly.py"
s = open(SRC).read()

OLD = """            if fuse_merge and bz < heads * fuse_tiles:
                # LSE denominator: computed on the ws(0) half as before, then
                # published to a shared scalar for all 256 threads.
                with T.ws(0):
                    for j in T.Parallel(fuse_lse_pad):
                        fuse_lse[j] = T.if_then_else(
                            j < num_split,
                            T.exp2(glse[bid, fuse_head, T.min(j, num_split - 1)]),
                            T.cast(0.0, accum_dtype),
                        )
                    T.reduce_sum(fuse_lse, fuse_s, dim=0)
                    if fuse_tx == 0:
                        fuse_fsum[0] = fuse_s[0]
                # Split reduction across both halves of the block: lane
                # fuse_tx // 128 owns splits [half*32, half*32+32) of the
                # output column fuse_tx % 128.
                fuse_half = fuse_tx // fuse_tile_dim
                fuse_col = fuse_tx % fuse_tile_dim
                for k in T.unroll(fuse_nhalf):
                    fuse_buf[k] = output_partial[
                        bid,
                        fuse_head,
                        fuse_half * fuse_nhalf + k,
                        fuse_tile * fuse_tile_dim + fuse_col,
                    ]
"""

NEW = """            if fuse_merge and bz < heads * fuse_tiles:
                # Split reduction across both halves of the block: lane
                # fuse_tx // 128 owns splits [half*32, half*32+32) of the
                # output column fuse_tx % 128.
                #
                # These partial-sum loads and the LSE load below are both
                # gated only by the arrival spin, so they are independent.
                # Issue the (32-deep) partial-sum chain first and let the LSE
                # reduction run behind it: one memory latency instead of two.
                fuse_half = fuse_tx // fuse_tile_dim
                fuse_col = fuse_tx % fuse_tile_dim
                for k in T.unroll(fuse_nhalf):
                    fuse_buf[k] = output_partial[
                        bid,
                        fuse_head,
                        fuse_half * fuse_nhalf + k,
                        fuse_tile * fuse_tile_dim + fuse_col,
                    ]
                # LSE denominator: computed on the ws(0) half as before, then
                # published to a shared scalar for all 256 threads.
                with T.ws(0):
                    for j in T.Parallel(fuse_lse_pad):
                        fuse_lse[j] = T.if_then_else(
                            j < num_split,
                            T.exp2(glse[bid, fuse_head, T.min(j, num_split - 1)]),
                            T.cast(0.0, accum_dtype),
                        )
                    T.reduce_sum(fuse_lse, fuse_s, dim=0)
                    if fuse_tx == 0:
                        fuse_fsum[0] = fuse_s[0]
"""

assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW, 1)
open(DST, "w").write(s)
print("wrote", DST)
