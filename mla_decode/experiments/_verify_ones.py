"""Verify + time: v540 baseline vs the ones-operand denominator (v541).

Same math as the MLA decode example:
  out = softmax( (q @ kv^T + q_pe @ k_pe^T) * (dim+pe_dim)**-0.5 ) @ kv
Measures device time the way the judge-facing notes do (self_device_time_total).
"""
import sys
sys.path.insert(0, "/data/tilelang-metax")
sys.path.insert(0, "/root/lzy/workbench")
import torch
import importlib

B, H, KVH, D, PD = 1, 16, 1, 512, 64
SCALE = (D + PD) ** -0.5


def make(ctx, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, D, dtype=torch.float16, generator=g).cuda().contiguous()
    qpe = torch.randn(B, H, PD, dtype=torch.float16, generator=g).cuda().contiguous()
    kv = torch.randn(B, ctx, KVH, D, dtype=torch.float16, generator=g).cuda().contiguous()
    kpe = torch.randn(B, ctx, KVH, PD, dtype=torch.float16, generator=g).cuda().contiguous()
    out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
    return q, qpe, kv, kpe, out


def ref(q, qpe, kv, kpe, ctx):
    s = (q[0].float() @ kv[0, :, 0].float().T
         + qpe[0].float() @ kpe[0, :, 0].float().T) * SCALE
    p = torch.softmax(s, dim=-1)
    return (p @ kv[0, :, 0].float()).half()


def dev_us(fn, n=300):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        if "main_split_kernel" in e.key:
            tot += e.self_device_time_total
    return tot / n


def main():
    ctxs = [int(a) for a in sys.argv[1:]] or [2048, 4096]
    mods = {}
    for name in ("v540_merge256", "_v542_del", "_v547_w4"):
        mods[name] = importlib.import_module(name)
    for ctx in ctxs:
        q, qpe, kv, kpe, out = make(ctx)
        r = ref(q, qpe, kv, kpe, ctx)
        res = {}
        for name, m in mods.items():
            o = out.clone()
            m.run_kernel(q, qpe, kv, kpe, o, B, H, KVH, ctx, D, PD)
            torch.cuda.synchronize()
            md = (o.float() - r.float()).abs().max().item()
            t = dev_us(lambda: m.run_kernel(q, qpe, kv, kpe, o, B, H, KVH, ctx, D, PD))
            res[name] = (md, t)
        base = 0.419 * ctx / 2048.0
        line = f"ctx={ctx:6d}  ref_scale={base:.3f}ms  "
        for name, (md, t) in res.items():
            d = int(100 * base / (base + t / 1000.0))
            line += f"| {name[-9:]:>9}: {t:6.2f}us d={d} maxdiff={md:.2e} "
        tb = res["v540_merge256"][1]
        tn = res["_v547_w4"][1]
        print(line + f"| delta={tn-tb:+.2f}us")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
