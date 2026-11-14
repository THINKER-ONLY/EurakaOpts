import tilelang, tilelang.language as T, torch, statistics
def build(kb):
    n = max(256, kb * 1024 // 2)   # fp16 elements
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float16")):
            with T.Kernel(64, threads=256) as (bx,):
                S = T.alloc_shared([n], "float16")
                tx = T.get_thread_binding(0)
                for i in T.serial(n // 256):
                    S[i * 256 + tx] = T.cast(1.0, "float16")
                T.sync_threads()
                for i in T.serial(n // 256):          # keep every element live
                    S[i * 256 + tx] = S[i * 256 + tx] + T.cast(1.0, "float16")
                T.sync_threads()
                if bx == 0 and tx == 0:
                    A[0] = S[0] + S[n - 1]
        return main
    return f()
a = torch.zeros(1, device="cuda", dtype=torch.float16)
for kb in (0, 8, 16, 32, 40, 48, 56, 64):
    try:
        k = build(kb)
        for _ in range(20): k(a)
        torch.cuda.synchronize()
        vals=[]
        for _ in range(40):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record()
            for _ in range(50): k(a)
            e.record(); torch.cuda.synchronize()
            vals.append(s.elapsed_time(e)/50*1000)
        print(f"shared={kb:3d}KB  launch median {statistics.median(vals):6.2f}us min {min(vals):6.2f}")
    except Exception as ex:
        print(f"shared={kb}KB FAIL {type(ex).__name__}: {str(ex)[:80]}")
