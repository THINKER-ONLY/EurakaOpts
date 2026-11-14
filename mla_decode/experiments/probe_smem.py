import tilelang, tilelang.language as T, torch
def build(kb):
    n_el = kb * 1024 // 2   # fp16 elements
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float16")):
            with T.Kernel(64, threads=256) as (bx,):
                S = T.alloc_shared([n_el], "float16")
                tx = T.get_thread_binding(0)
                for i in T.serial(n_el // 256):
                    S[i * 256 + tx] = T.cast(1.0, "float16")
                T.sync_threads()
                if bx == 0 and tx == 0:
                    A[0] = S[0]
        return main
    return f
for kb in (48, 56, 64, 68, 72, 80, 96):
    try:
        k = build(kb); a = torch.zeros(1, device="cuda", dtype=torch.float16)
        k(a); torch.cuda.synchronize()
        print(f"{kb}KB: OK")
    except Exception as e:
        print(f"{kb}KB: FAIL {type(e).__name__}: {str(e)[:120]}")
