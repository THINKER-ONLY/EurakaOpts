import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build():
    @T.prim_func
    def main(X: T.Tensor([256], "float32"), Y: T.Tensor([256], "float32")):
        with T.Kernel(1, threads=256) as _:
            S0 = T.alloc_shared([64], "float32")
            S1 = T.alloc_shared([64], "float32")
            for k in T.serial(2):
                with T.ws(0):
                    for i in T.Parallel(64):
                        S0[i] = X[k * 128 + i]
                with T.ws(1):
                    for i in T.Parallel(64):
                        S1[i] = X[k * 128 + 64 + i]
                T.sync_threads()
                with T.ws(0):
                    for i in T.Parallel(64):
                        Y[k * 128 + i] = S0[i]
                        Y[k * 128 + 64 + i] = S1[i]
                T.sync_threads()
    return main

x=torch.arange(256,dtype=torch.float32,device="cuda")
y=torch.zeros(256,device="cuda")
try:
    k=build(); k(x,y); torch.cuda.synchronize()
    print("equal", torch.equal(x,y), "max_abs", (x-y).abs().max().item())
    print("y[:8]", y[:8].tolist(), "y[64:72]", y[64:72].tolist())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
