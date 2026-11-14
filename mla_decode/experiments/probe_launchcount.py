"""Count real device launches per submission call, and inspect the generated
host stub for extra launches/memsets (e.g. zero-fill of alloc_global scratch)."""
import importlib.util, sys, re
import torch

sys.path.insert(0, "/root/lzy/workbench")
spec = importlib.util.spec_from_file_location("sub", "/root/lzy/workbench/v540_merge256.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

ctx = 4096
dim, pe_dim, heads, kv_heads, batch = 512, 64, 16, 1, 1
q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
qpe = torch.randn(batch, heads, pe_dim, device="cuda", dtype=torch.float16)
kv = torch.randn(batch, ctx, kv_heads, dim, device="cuda", dtype=torch.float16)
kpe = torch.randn(batch, ctx, kv_heads, pe_dim, device="cuda", dtype=torch.float16)
out = torch.empty(batch, heads, dim, device="cuda", dtype=torch.float16)

for _ in range(3):
    sub.run_kernel(q, qpe, kv, kpe, out, batch, heads, kv_heads, ctx, dim, pe_dim)
torch.cuda.synchronize()

k = sub._LAST_KERNEL
print("kernel obj:", type(k))
for attr in ("get_kernel_source", "get_host_source", "get_source", "source", "host_source"):
    if hasattr(k, attr):
        try:
            v = getattr(k, attr)
            s = v() if callable(v) else v
            if isinstance(s, str):
                print(f"--- {attr}: {len(s)} chars")
                print("  __global__ count:", s.count("__global__"))
                print("  <<< count:", s.count("<<<"))
                print("  LaunchKernel:", s.count("LaunchKernel"), "cudaLaunchKernel:", s.count("cudaLaunchKernel"))
                print("  cudaMemset:", s.count("cudaMemset"), "memset:", s.count("memset"))
        except Exception as e:
            print(f"--- {attr}: {type(e).__name__} {str(e)[:120]}")

# profiler-based launch count
try:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            sub.run_kernel(q, qpe, kv, kpe, out, batch, heads, kv_heads, ctx, dim, pe_dim)
        torch.cuda.synchronize()
    ev = 0
    for e in prof.key_averages():
        if e.device_type is not None and str(e.device_type).endswith("CUDA"):
            print("KERNEL:", e.key[:70], "count=", e.count, "self_us=", round(e.self_device_time_total / max(1, e.count), 2))
            ev += e.count
    print("total device events:", ev)
except Exception as e:
    print("PROFILER FAIL", type(e).__name__, str(e)[:200])
