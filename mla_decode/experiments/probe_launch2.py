import tilelang, tilelang.language as T, torch, statistics
def build(n):
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float16")):
            with T.Kernel(n, threads=256) as (bx,):
                if bx == 0:
                    A[0] = T.cast(1.0, "float16")
        return main
    return f()
a = torch.zeros(1, device="cuda", dtype=torch.float16)
for n in (1, 16, 32, 64, 128):
    try:
        k = build(n)
        for _ in range(20): k(a)
        torch.cuda.synchronize()
        vals=[]
        for _ in range(40):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record()
            for _ in range(50): k(a)
            e.record(); torch.cuda.synchronize()
            vals.append(s.elapsed_time(e)/50*1000)
        print(f"ctas={n:4d} launch median {statistics.median(vals):.2f}us min {min(vals):.2f}us")
    except Exception as ex:
        print(f"ctas={n} FAIL {type(ex).__name__}")
