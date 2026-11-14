"""Device-time ablation of v540. Each variant is produced by exact-text
replacement (asserted count==1), so we know which code actually changed."""
import os, sys, importlib.util
import torch
from torch.profiler import profile, ProfilerActivity

BASE = "/root/lzy/workbench/v540_merge256.py"
SRC = open(BASE).read()

KVCOPY = """                T.copy(
                    KV[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    KV_shared,
                )
"""
KPECOPY = """                T.copy(
                    K_pe[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    K_pe_shared,
                )
"""
QK = """                    T.gemm(
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
"""
PV = """                if pv_specialize:
                    pv_mfma4(
                        S_shared,
                        KV_shared,
                        pv_a,
                        pv_b,
                        acc_o,
                        lane_s,
                        warp_s,
                    )
"""
ROWS = """                for i in T.Parallel(valid_block_h):
                    tile_sum[0] = T.cast(0.0, accum_dtype)
                    for j in T.serial(block_n):
                        tile_sum[0] += T.cast(S_shared[i, j], accum_dtype)
                    logsum[i] = logsum[i] + tile_sum[0]
"""
PART = """                T.copy(
                    O_shared,
                    output_partial[
                        bid,
                        hid * valid_block_h : (hid + 1) * valid_block_h,
                        bz,
                        :,
                    ],
                )
"""
SPIN = """                if fuse_tx == 0:
                    fuse_old = T.atomic_add(
                        State[0], 1, memory_order="acq_rel", return_prev=True
                    )
                    fuse_target = (fuse_old // num_split + 1) * num_split
                    while (
                        T.atomic_add(
                            State[0], 0, memory_order="acq_rel", return_prev=True
                        )
                        < fuse_target
                    ):
                        T.evaluate(0)
"""
MERGE = """                for k in T.unroll(fuse_nhalf):
                    fuse_buf[k] = output_partial[
                        bid,
                        fuse_head,
                        fuse_half * fuse_nhalf + k,
                        fuse_tile * fuse_tile_dim + fuse_col,
                    ]
"""
EPI = """            for i, j in T.Parallel(valid_block_h, dim):
                    acc_o[i, j] /= logsum[i]
"""
EPI2 = """                for i, j in T.Parallel(valid_block_h, dim):
                    acc_o[i, j] /= logsum[i]
"""

QK_ONLY_MAIN = """                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
"""
EXP2 = """                    for i, j in T.Parallel(valid_block_h, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale)
"""
SCOPY = """                    T.copy(acc_s, S_shared)
"""
QLOAD = """                T.copy(
                    Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                    Q_shared,
                )
"""

# (old, new, occurrence_index) — index 0 is the main_split path.
VARIANTS = {
    "base": [],
    "nokv": [(KVCOPY, "                pass\n", 0), (KPECOPY, "                pass\n", 0)],
    "noqk": [(QK, "                    pass\n", 0)],
    "noqk512": [(QK_ONLY_MAIN, "                    pass\n", 0)],
    "noexp2": [(EXP2, "                    pass\n", 0)],
    "noScopy": [(SCOPY, "                    pass\n", 0)],
    "noqload": [(QLOAD, "                pass\n", 0)],
    "nopv": [(PV, "                if pv_specialize:\n                    pass\n", 0)],
    "norowsum": [(ROWS, "                pass\n", 0)],
    "nopartwrite": [(PART, "                pass\n", 0)],
    "nospin": [(SPIN, "                pass\n", 0)],
    "nomergeload": [(MERGE, "                pass\n", 0)],
}


def make(name, edits):
    s = SRC
    for old, new, idx in edits:
        n = s.count(old)
        assert n > idx, f"{name}: pattern appears {n} times, need idx {idx}"
        parts = s.split(old)
        s = old.join(parts[: idx + 1]) + new + old.join(parts[idx + 1:])
        assert s != SRC
    p = f"/tmp/abl/{name}.py"
    open(p, "w").write(s)
    return p


def load(p, name):
    spec = importlib.util.spec_from_file_location(f"abl_{name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dev_us(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = cnt = 0.0
    for e in prof.key_averages():
        if str(e.device_type).endswith("CUDA") and e.count:
            tot += e.self_device_time_total
            cnt += e.count
    return tot / max(1, cnt)


os.makedirs("/tmp/abl", exist_ok=True)
dim, pe_dim, heads, kv_heads, batch = 512, 64, 16, 1, 1
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
CTXS = [2048, 4096]

mods = {}
for name in only:
    try:
        mods[name] = load(make(name, VARIANTS[name]), name)
    except Exception as e:
        print(f"{name}: BUILD FAIL {type(e).__name__} {str(e)[:150]}")

res = {}
for ctx in CTXS:
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
    qpe = torch.randn(batch, heads, pe_dim, device="cuda", dtype=torch.float16)
    kv = torch.randn(batch, ctx, kv_heads, dim, device="cuda", dtype=torch.float16)
    kpe = torch.randn(batch, ctx, kv_heads, pe_dim, device="cuda", dtype=torch.float16)
    out = torch.empty(batch, heads, dim, device="cuda", dtype=torch.float16)
    for name, m in mods.items():
        try:
            f = lambda m=m: m.run_kernel(q, qpe, kv, kpe, out, batch, heads, kv_heads, ctx, dim, pe_dim)
            t = dev_us(f)
            res[(name, ctx)] = t
            print(f"ctx={ctx:6d} {name:14s} dev={t:7.2f}us")
        except Exception as e:
            print(f"ctx={ctx:6d} {name:14s} FAIL {type(e).__name__} {str(e)[:120]}")
    del q, qpe, kv, kpe, out
    torch.cuda.empty_cache()

print("\n=== deltas vs base ===")
for ctx in CTXS:
    b = res.get(("base", ctx))
    if not b:
        continue
    for name in only:
        t = res.get((name, ctx))
        if t:
            print(f"ctx={ctx:6d} {name:14s} d={t-b:+7.2f}us")
