import torch, tilelang
import tilelang.language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def build(scale):
    @T.prim_func
    def main(
        Q: T.Tensor([16, 512], "float16"),
        Qp: T.Tensor([16, 64], "float16"),
        K: T.Tensor([16, 512], "float16"),
        Kp: T.Tensor([16, 64], "float16"),
        O: T.Tensor([16, 512], "float16"),
    ):
        with T.Kernel(1, threads=64) as _:
            Ql = T.alloc_fragment([16, 512], "float16")
            Qpl = T.alloc_fragment([16, 64], "float16")
            KVs = T.alloc_shared([16, 512], "float16")
            Kps = T.alloc_shared([16, 64], "float16")
            Ss = T.alloc_shared([16, 16], "float16")
            acc_s = T.alloc_fragment([16, 16], "float32")
            acc_o = T.alloc_fragment([16, 512], "float32")
            a = T.alloc_local([4], "float16")
            b = T.alloc_local([4], "float16")
            T.annotate_layout({
                acc_o: T.Fragment(
                    [16, 512],
                    forward_thread_fn=lambda i, j: ((j % 16) // 4) * 16 + i,
                    forward_index_fn=lambda i, j: (j // 16) * 4 + j % 4,
                )
            })
            tx = T.get_thread_binding(0)
            lane = tx % 64
            T.copy(Q, Ql)
            T.copy(Qp, Qpl)
            T.copy(K, KVs)
            T.copy(Kp, Kps)
            T.clear(acc_s)
            T.gemm(Ql, KVs, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.gemm(Qpl, Kps, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(16, 16):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale)
            T.copy(acc_s, Ss)
            T.fill(acc_o, 0)
            T.sync_threads()
            a[0] = Ss[lane % 16, (lane // 16) * 4 + 0]
            a[1] = Ss[lane % 16, (lane // 16) * 4 + 1]
            a[2] = Ss[lane % 16, (lane // 16) * 4 + 2]
            a[3] = Ss[lane % 16, (lane // 16) * 4 + 3]
            for n in T.serial(32):
                b[0] = KVs[(lane // 16) * 4 + 0, n * 16 + lane % 16]
                b[1] = KVs[(lane // 16) * 4 + 1, n * 16 + lane % 16]
                b[2] = KVs[(lane // 16) * 4 + 2, n * 16 + lane % 16]
                b[3] = KVs[(lane // 16) * 4 + 3, n * 16 + lane % 16]
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4", b.data, 0, a.data, 0, acc_o.data, n, dtype="float32x4")
            T.copy(acc_o, O)
    return main

torch.manual_seed(0)
q=torch.randn(16,512,device="cuda",dtype=torch.float16)
qp=torch.randn(16,64,device="cuda",dtype=torch.float16)
k=torch.randn(16,512,device="cuda",dtype=torch.float16)
kp=torch.randn(16,64,device="cuda",dtype=torch.float16)
o=torch.zeros(16,512,device="cuda",dtype=torch.float16)
scale=(576**-0.5)*1.44269504
try:
    ker=build(scale)
    ker(q,qp,k,kp,o)
    torch.cuda.synchronize()
    s=(q.float()@k.float().T + qp.float()@kp.float().T)/ (576**0.5)
    ref=(torch.exp(s)@k.float())
    print("max_abs", (o.float()-ref).abs().max().item())
    print("out sample", o[0,:4].tolist())
    print("ref sample", ref[0,:4].tolist())
    print("out abs max", o.abs().max().item(), "ref abs max", ref.abs().max().item())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:400])
