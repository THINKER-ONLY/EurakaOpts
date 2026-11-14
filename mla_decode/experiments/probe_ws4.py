import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build():
    @T.prim_func
    def main(X: T.Tensor([1, 16, 32], "float16"), Out: T.Tensor([16], "float32")):
        with T.Kernel(1, threads=256) as _:
            acc = T.alloc_fragment([16, 32], "float32")
            m = T.alloc_fragment([16], "float32")
            with T.ws(0):
                for i, j in T.Parallel(16, 32):
                    acc[i, j] = X[0, i, j]
                T.reduce_max(acc, m, dim=1, clear=True)
            T.sync_threads()
            with T.ws(0):
                T.copy(m, Out)
    return main

torch.manual_seed(0)
x=torch.randn(1,16,32,device="cuda",dtype=torch.float16)
o=torch.zeros(16,device="cuda",dtype=torch.float32)
try:
    k=build(); k(x,o); torch.cuda.synchronize()
    ref=x[0].float().max(dim=1).values
    print("max_abs", (o-ref).abs().max().item())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
