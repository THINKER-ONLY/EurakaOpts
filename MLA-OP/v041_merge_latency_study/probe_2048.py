"""Device-time breakdown of v960 at ctx=2048 (single fused kernel).

Each ablation is an exact-text replacement so we know what actually changed.
Also sweeps the short-context split target.
"""
import os, sys, importlib.util
import torch
from torch.profiler import profile, ProfilerActivity

BASE = "/root/lzy/workbench/v960_split104_aligned.py"
SRC = open(BASE).read()

KVCOPY = """                T.copy(
                    K_pe[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    K_pe_shared,
                )
"""
QK = """                with T.ws(0):
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
"""
MERGE = """            if fuse_merge and bz < heads * fuse_tiles:
"""
PARTW = """                T.copy(
                    O_shared,
                    output_partial[
                        bid,
                        hid * valid_block_h : (hid + 1) * valid_block_h,
                        bz,
                        :,
                    ],
                )
"""
QLOAD = """                T.copy(
                    Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                    Q_shared,
                )
"""

KVCOPY2 = """                T.copy(
                    KV[bid, kv_block_start : kv_block_start + block_n, cur_kv_head, :],
                    KV_shared,
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
MREAD = """                for k in T.unroll(fuse_nhalf):
                    fuse_buf[k] = output_partial[
                        bid,
                        fuse_head,
                        fuse_half * fuse_nhalf + k,
                        fuse_tile * fuse_tile_dim + fuse_col,
                    ]
"""
LSE = """                    T.reduce_sum(fuse_lse, fuse_s, dim=0)
"""

VARIANTS = {
    "base": [],
    "nokpe": [(KVCOPY, "                pass\n", 0)],
    "nokv": [(KVCOPY2, "                pass\n", 0)],
    "noqk": [(QK, "                with T.ws(0):\n                    pass\n", 0)],
    "nomerge": [(MERGE, "            if False:\n", 0)],
    "nospin": [(SPIN, "                pass\n", 0)],
    "nomergeval": [(MREAD, "                for k in T.unroll(fuse_nhalf):\n                    fuse_buf[k] = T.cast(0.0, dtype)\n", 0)],
    "nolse": [(LSE, "                    fuse_s[0] = T.cast(1.0, accum_dtype)\n", 0)],
    "nopartw": [(PARTW, "                pass\n", 0)],
    "noqload": [(QLOAD, "                pass\n", 0)],
    "split32": [("_SPLIT_TARGET_SHORT = 64", "_SPLIT_TARGET_SHORT = 32", 0)],
    "split48": [("_SPLIT_TARGET_SHORT = 64", "_SPLIT_TARGET_SHORT = 48", 0)],
}


def make(name, edits):
    s = SRC
    for old, new, idx in edits:
        n = s.count(old)
        assert n > idx, f"{name}: pattern appears {n} times, need idx {idx}"
        parts = s.split(old)
        s = old.join(parts[: idx + 1]) + new + old.join(parts[idx + 1:])
        assert s != SRC
    p = f"/tmp/abl2/{name}.py"
    open(p, "w").write(s)
    return p


def load(p, name):
    spec = importlib.util.spec_from_file_location(f"p2_{name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dev_us(fn, iters=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


os.makedirs("/tmp/abl2", exist_ok=True)
dim, pe_dim, heads, kv_heads, batch = 512, 64, 16, 1, 1
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)

q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
qpe = torch.randn(batch, heads, pe_dim, device="cuda", dtype=torch.float16)
kv = torch.randn(batch, ctx, kv_heads, dim, device="cuda", dtype=torch.float16)
kpe = torch.randn(batch, ctx, kv_heads, pe_dim, device="cuda", dtype=torch.float16)
out = torch.empty(batch, heads, dim, device="cuda", dtype=torch.float16)

res = {}
for name in only:
    try:
        m = load(make(name, VARIANTS[name]), name)
        f = lambda m=m: m.run_kernel(q, qpe, kv, kpe, out, batch, heads, kv_heads, ctx, dim, pe_dim)
        t = dev_us(f)
        res[name] = t
        print(f"{name:10s} dev={t:7.2f}us")
    except Exception as e:
        print(f"{name:10s} FAIL {type(e).__name__} {str(e)[:160]}")
    sys.stdout.flush()

b = res.get("base")
if b:
    print("\n=== delta vs base ===")
    for name in only:
        if name in res:
            print(f"{name:10s} {res[name]-b:+7.2f}us")
