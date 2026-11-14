"""Lens probe: how many CTAs are really co-resident on the C500 at the real
shared footprint, and do 128 CTAs cost one wave or two?

Latency-bound per-CTA chain (dependent LDS loads), so co-residency == plateau.
"""
import sys, statistics
import tilelang, tilelang.language as T, torch


def build(grid, nshared_f32, iters):
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float32")):
            with T.Kernel(grid, threads=256) as (bx,):
                S = T.alloc_shared([nshared_f32], "float32")
                acc = T.alloc_local([1], "float32")
                tx = T.get_thread_binding(0)
                S[tx] = T.cast(bx + tx, "float32")
                T.sync_threads()
                acc[0] = T.cast(tx % 64, "float32")
                for _ in T.serial(iters):
                    acc[0] = S[T.cast(acc[0], "int32")]
                if bx == 0 and tx == 0:
                    A[0] = acc[0]
        return main
    return f()


def dev_us(k, a, reps=9):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(10):
        k(a)
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                k(a)
            torch.cuda.synchronize()
        for e in prof.key_averages():
            if str(e.device_type).endswith("CUDA"):
                vals.append(e.self_device_time_total / max(1, e.count))
    return statistics.median(vals) if vals else float("nan")


a = torch.zeros(1, device="cuda", dtype=torch.float32)

# calibrate: one CTA chain length
ITERS=200
cal = build(1, 1024, ITERS)
for _ in range(5):
    cal(a)
torch.cuda.synchronize()
print("calib grid=1 iters=200 dev_us =", round(dev_us(cal, a), 2))

nbig = 14080   # 56320 B  == real kernel footprint
nsmall = 1024  # 4096 B
for label, ns in (("57KB", nbig), ("4KB", nsmall)):
    print(f"--- shared {label}/CTA (latency chain, uniform work per CTA)")
    for g in (64, 96, 104, 112, 120, 128, 160, 208, 256):
        k = build(g, ns, ITERS)
        print(f"  grid={g:4d}  dev_us={dev_us(k, a):7.2f}")
        sys.stdout.flush()
