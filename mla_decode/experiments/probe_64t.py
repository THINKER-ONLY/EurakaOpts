"""Probe: is M=16,N=16 QK legal with 64 threads (1 warp), and does
T.Pipelined(num_stages=2) double-buffer + stay numerically right on MACA?"""
import torch, tilelang
import tilelang.language as T


@tilelang.jit
def build(ctx, stages):
    @T.prim_func
    def main(
        Q: T.Tensor([16, 576], "float16"),
        KV: T.Tensor([ctx, 576], "float16"),
        Out: T.Tensor([16, 16], "float16"),
    ):
        with T.Kernel(1, threads=64) as _:
            Q_s = T.alloc_fragment([16, 512], "float16")
            Qp_s = T.alloc_fragment([16, 64], "float16")
            KV_s = T.alloc_shared([16, 512], "float16")
            KP_s = T.alloc_shared([16, 64], "float16")
            acc = T.alloc_fragment([16, 16], "float32")
            a = T.alloc_fragment([16, 16], "float32")
            T.copy(Q[:, :512], Q_s)
            T.copy(Q[:, 512:], Qp_s)
            T.fill(a, 0)
            for k in T.Pipelined(ctx // 16, num_stages=stages):
                T.copy(KV[k * 16:(k + 1) * 16, :512], KV_s)
                T.copy(KV[k * 16:(k + 1) * 16, 512:], KP_s)
                T.gemm(Q_s, KV_s, acc, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(Qp_s, KP_s, acc, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(16, 16):
                    a[i, j] = a[i, j] + acc[i, j]
            T.copy(a, Out)
    return main


torch.manual_seed(0)
ctx = 4096
q = torch.randn(16, 576, device="cuda", dtype=torch.float16)
kv = torch.randn(ctx, 576, device="cuda", dtype=torch.float16)
for st in (0, 2):
    try:
        o = torch.zeros(16, 16, device="cuda", dtype=torch.float16)
        k = build(ctx, st)
        k(q, kv, o)
        torch.cuda.synchronize()
        ksum = kv.float().reshape(ctx // 16, 16, 576).sum(0)
        ref = (q.float() @ ksum.T).half()
        print(f"stages={st} max_abs={(o.float()-ref.float()).abs().max().item():.5f}")
    except Exception as e:
        print(f"stages={st} FAIL {type(e).__name__}: {str(e)[:300]}")
