"""v964_stage32 verification: correctness fingerprint vs BASE + timing,
and proof that several contexts in ONE process no longer deadlock."""
import sys, importlib.util, torch

MODE = sys.argv[1]                      # base | var
PATHS = {
    "base": "/root/lzy/workbench/v960_split104_aligned.py",
    "v964": "/root/lzy/workbench/v964_stage32.py",
    "v965": "/root/lzy/workbench/v965_short512.py",
    "v966": "/root/lzy/workbench/v966_combined.py",
    "v967": "/root/lzy/workbench/v967_pad8.py",
    "v990": "/root/lzy/workbench/v990_long512.py",
    "v991": "/root/lzy/workbench/v991_full.py",
    "v992": "/root/lzy/workbench/v992_4096.py",
    "v993": "/root/lzy/workbench/v993_best.py",
}
PATH = PATHS[MODE]
CTXS = [int(x) for x in sys.argv[2].split(",")]

spec = importlib.util.spec_from_file_location(f"m_{MODE}", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

b, h, kvh, dim, pe = 1, 16, 1, 512, 64
for ctx in CTXS:
    torch.manual_seed(ctx)
    q = torch.randn(b, h, dim, device="cuda", dtype=torch.float16)
    qpe = torch.randn(b, h, pe, device="cuda", dtype=torch.float16)
    kv = torch.randn(b, ctx, kvh, dim, device="cuda", dtype=torch.float16)
    kpe = torch.randn(b, ctx, kvh, pe, device="cuda", dtype=torch.float16)
    out = torch.empty(b, h, dim, device="cuda", dtype=torch.float16)

    f = lambda: m.run_kernel(q, qpe, kv, kpe, out, b, h, kvh, ctx, dim, pe)
    for _ in range(20):
        f()
    torch.cuda.synchronize()

    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    best = 1e9
    for _ in range(3):
        s.record()
        for _ in range(300):
            f()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 300 * 1000)

    o = out.float()
    print(f"{MODE} ctx={ctx:6d} {best:7.2f}us nan={torch.isnan(out).any().item()} "
          f"absmax={o.abs().max().item():.6f} sum={o.sum().item():.6f} "
          f"sq={float((o * o).sum()):.6f}")
    sys.stdout.flush()
    del q, qpe, kv, kpe, out, o
    torch.cuda.empty_cache()
print("DONE", MODE)
