"""Verify (a) the real dynamic-shared size used by the probe kernel and
(b) how many CTAs of a long latency-bound chain really co-reside."""
import statistics, sys, re
import tilelang, tilelang.language as T, torch


def build(grid, ns, iters):
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float32")):
            with T.Kernel(grid, threads=256) as (bx,):
                S = T.alloc_shared([ns], "float32")
                acc = T.alloc_local([1], "float32")
                tx = T.get_thread_binding(0)
                for i in T.serial(0, ns, 256):
                    S[i + tx] = T.cast(bx + tx, "float32")   # touch whole buffer
                T.sync_threads()
                acc[0] = T.cast(tx % 64, "float32")
                for _ in T.serial(iters):
                    acc[0] = S[T.cast(acc[0], "int32")]
                if bx == 0 and tx == 0:
                    A[0] = acc[0]
        return main
    return f()


def dev_us(k, a, reps=7):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(6):
        k(a)
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                k(a)
            torch.cuda.synchronize()
        for e in prof.key_averages():
            if str(e.device_type).endswith("CUDA"):
                vals.append(e.self_device_time_total / max(1, e.count))
    return statistics.median(vals) if vals else float("nan")


a = torch.zeros(1, device="cuda", dtype=torch.float32)
NS = 14080   # 56,320 B, same as the real kernel
ITERS = 2000

k1 = build(104, NS, ITERS)
src = None
for attr in ("get_kernel_source", "get_source"):
    if hasattr(k1, attr):
        v = getattr(k1, attr)
        src = v() if callable(v) else v
        break
if isinstance(src, str):
    for line in src.splitlines():
        if ("__launch_bounds__" in line or "shared" in line.lower()
                or "dyn" in line.lower() or "<<<" in line):
            print("SRC:", line.strip()[:180])
print("calib grid=104 iters=2000 dev_us=", round(dev_us(k1, a), 2))
sys.stdout.flush()

for label, ns in (("56KB", NS), ("4KB", 1024)):
    print(f"--- shared {label}, latency chain iters={ITERS}")
    for g in (64, 104, 208, 312, 416, 624, 832):
        k = build(g, ns, ITERS)
        print(f"  grid={g:4d}  dev_us={dev_us(k, a):8.2f}")
        sys.stdout.flush()
