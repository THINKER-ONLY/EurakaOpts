import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build():
    @T.prim_func
    def main(
        Q: T.Tensor([16, 512], "float16"),
        QP: T.Tensor([16, 64], "float16"),
        K: T.Tensor([32, 512], "float16"),
        KP: T.Tensor([32, 64], "float16"),
        O: T.Tensor([1, 16, 1, 512], "float16"),
    ):
        with T.Kernel(1, threads=128) as _:
            Q_lo = T.alloc_shared([16, 256], "float16")
            Q_hi = T.alloc_shared([16, 256], "float16")
            QP_lo = T.alloc_shared([16, 32], "float16")
            QP_hi = T.alloc_shared([16, 32], "float16")
            K_lo = T.alloc_shared([32, 256], "float16")
            K_hi = T.alloc_shared([32, 256], "float16")
            KP_lo = T.alloc_shared([32, 32], "float16")
            KP_hi = T.alloc_shared([32, 32], "float16")
            S_s = T.alloc_shared([16, 32], "float16")
            acc_s = T.alloc_fragment([16, 32], "float32")
            o_lo = T.alloc_fragment([16, 256], "float32")
            o_hi = T.alloc_fragment([16, 256], "float32")
            m = T.alloc_fragment([16], "float32")
            sm = T.alloc_fragment([16], "float32")
            T.copy(Q[:, 0:256], Q_lo); T.copy(Q[:, 256:512], Q_hi)
            T.copy(QP[:, 0:32], QP_lo); T.copy(QP[:, 32:64], QP_hi)
            T.copy(K[:, 0:256], K_lo); T.copy(K[:, 256:512], K_hi)
            T.copy(KP[:, 0:32], KP_lo); T.copy(KP[:, 32:64], KP_hi)
            T.clear(acc_s)
            T.gemm(Q_lo, K_lo, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(Q_hi, K_hi, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(QP_lo, KP_lo, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(QP_hi, KP_hi, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.reduce_max(acc_s, m, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                acc_s[i, j] = T.exp2(acc_s[i, j] * 0.0601132 - m[i] * 0.0601132)
            T.reduce_sum(acc_s, sm, dim=1, clear=True)
            T.copy(acc_s, S_s)
            T.fill(o_lo, 0); T.fill(o_hi, 0)
            T.gemm(S_s, K_lo, o_lo, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(S_s, K_hi, o_hi, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(16, 256):
                o_lo[i, j] /= sm[i]
                o_hi[i, j] /= sm[i]
            T.copy(o_lo, O[0, :, 0, 0:256]); T.copy(o_hi, O[0, :, 0, 256:512])
    return main

torch.manual_seed(0)
q=torch.randn(16,512,device="cuda",dtype=torch.float16)
qp=torch.randn(16,64,device="cuda",dtype=torch.float16)
k=torch.randn(32,512,device="cuda",dtype=torch.float16)
kp=torch.randn(32,64,device="cuda",dtype=torch.float16)
o=torch.zeros(1,16,1,512,device="cuda",dtype=torch.float16)
try:
    ker=build(); ker(q,qp,k,kp,o); torch.cuda.synchronize()
    s=(q.float()@k.float().transpose(0,1)+qp.float()@kp.float().transpose(0,1))/(576**0.5)
    p=torch.softmax(s,dim=-1)
    ref=p@k.float()
    print("max_abs", (o[0,:,0,:].float()-ref).abs().max().item(), "ref_scale", ref.abs().max().item())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
