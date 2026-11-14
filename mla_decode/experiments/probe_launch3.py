import tilelang, tilelang.language as T, torch, statistics
def build(n, th):
    @tilelang.jit
    def f():
        @T.prim_func
        def main(A: T.Tensor([1], "float16")):
            with T.Kernel(n, threads=th) as (bx,):
                if bx == 0:
                    A[0] = T.cast(1.0, "float16")
        return main
    return f()
a = torch.zeros(1, device="cuda", dtype=torch.float16)
for th in (32, 64, 128, 256, 512):
    try:
        k = build(64, th)
        for _ in range(20): k(a)
        torch.cuda.synchronize()
        vals=[]
        for _ in range(40):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record()
            for _ in range(50): k(a)
            e.record(); torch.cuda.synchronize()
            vals.append(s.elapsed_time(e)/50*1000)
        print(f"threads={th:4d} launch median {statistics.median(vals):.2f}us min {min(vals):.2f}us")
    except Exception as ex:
        print(f"threads={th} FAIL {type(ex).__name__}")
