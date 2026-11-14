import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build():
    @T.prim_func
    def main(
        A: T.Tensor([1, 16, 512], "float16"),
        B: T.Tensor([1, 64, 2, 512], "float16"),
        Out: T.Tensor([16, 32], "float16"),
    ):
        with T.Kernel(1, threads=256) as _:
            A_s = T.alloc_shared([16, 512], "float16")
            B_s = T.alloc_shared([32, 512], "float16")
            C = T.alloc_fragment([16, 32], "float32")
            T.copy(A[0, :, :], A_s)
            T.clear(C)
            for k in T.serial(2):
                base = k * 32
                with T.ws(0):
                    T.copy(B[0, base:base+16, 0, :], B_s[0:16, :])
                with T.ws(1):
                    T.copy(B[0, base+16:base+32, 0, :], B_s[16:32, :])
                T.sync_threads()
                with T.ws(0):
                    T.gemm(A_s, B_s, C, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.sync_threads()
            with T.ws(0):
                T.copy(C, Out)
    return main

torch.manual_seed(0)
a=torch.randn(1,16,512,device="cuda",dtype=torch.float16)
b=torch.randn(1,64,2,512,device="cuda",dtype=torch.float16)
o=torch.zeros(16,32,device="cuda",dtype=torch.float16)
try:
    k=build(); k(a,b,o); torch.cuda.synchronize()
    ref=sum((a[0].float()@b[0,i*32:(i+1)*32,0,:].float().T) for i in range(2)).half()
    print("max_abs", (o.float()-ref.float()).abs().max().item())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
