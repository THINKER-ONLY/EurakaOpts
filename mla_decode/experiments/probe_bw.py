import torch, tilelang, time, statistics
import tilelang.language as T

@tilelang.jit
def build(ctx, threads, bn):
    per = ctx // 64
    @T.prim_func
    def main(X: T.Tensor([ctx, 512], "float16"), Y: T.Tensor([64, 512], "float16")):
        with T.Kernel(64, threads=threads) as bx:
            S = T.alloc_shared([bn, 512], "float16")
            for k in T.serial(per // bn):
                T.copy(X[bx * per + k * bn : bx * per + (k + 1) * bn, :], S)
            T.copy(S, Y[bx:bx+1, :]) if False else None
            for i, j in T.Parallel(1, 512):
                Y[bx, j] = S[0, j]
    return main

ctx = 16384  # 16MB
x = torch.randn(ctx, 512, device="cuda", dtype=torch.float16)
y = torch.zeros(64, 512, device="cuda", dtype=torch.float16)
for threads in (128, 256, 512):
    try:
        k = build(ctx, threads, 32)
        k(x, y); torch.cuda.synchronize()
        vals=[]
        for _ in range(5):
            st,en=torch.cuda.Event(True),torch.cuda.Event(True)
            st.record()
            for _ in range(20): k(x,y)
            en.record(); torch.cuda.synchronize()
            vals.append(st.elapsed_time(en)/20)
        ms=statistics.median(vals)
        print(f"threads={threads} warps={threads//64}: {ms*1000:.1f} us, read BW {16/1024/(ms/1000):.1f} GB/s")
    except Exception as e:
        print(f"threads={threads} FAIL {type(e).__name__}: {str(e)[:100]}")
