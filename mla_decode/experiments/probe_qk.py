import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build(Kdim=512):
    @T.prim_func
    def main(Q: T.Tensor([16, Kdim], "float16"),
             KV: T.Tensor([32, Kdim], "float16"),
             Out: T.Tensor([16, 32], "float32")):
        with T.Kernel(1, threads=128) as _:
            Qs = T.alloc_shared([16, Kdim], "float16")
            KVs = T.alloc_shared([32, Kdim], "float16")
            acc = T.alloc_local([4], "float32")
            a = T.alloc_local([4], "float16")
            b = T.alloc_local([4], "float16")
            T.copy(Q, Qs); T.copy(KV, KVs)
            tx = T.get_thread_binding(0)
            lane = tx % 64
            warp = tx // 64
            for c in T.serial(4):
                acc[c] = 0.0
            T.sync_threads()
            for kk in T.serial(Kdim // 16):
                a[0] = Qs[lane % 16, kk * 16 + (lane // 16) * 4 + 0]
                a[1] = Qs[lane % 16, kk * 16 + (lane // 16) * 4 + 1]
                a[2] = Qs[lane % 16, kk * 16 + (lane // 16) * 4 + 2]
                a[3] = Qs[lane % 16, kk * 16 + (lane // 16) * 4 + 3]
                b[0] = KVs[warp * 16 + lane % 16, kk * 16 + (lane // 16) * 4 + 0]
                b[1] = KVs[warp * 16 + lane % 16, kk * 16 + (lane // 16) * 4 + 1]
                b[2] = KVs[warp * 16 + lane % 16, kk * 16 + (lane // 16) * 4 + 2]
                b[3] = KVs[warp * 16 + lane % 16, kk * 16 + (lane // 16) * 4 + 3]
                T.tvm_mfma("16x16x16f16", "row", "row", "float16x4", "float16x4", "float32x4",
                           b.data, 0, a.data, 0, acc.data, 0, dtype="float32x4")
            for c in T.serial(4):
                Out[lane % 16, warp * 16 + (lane // 16) * 4 + c] = acc[c]
    return main

torch.manual_seed(0)
q=torch.randn(16,512,device="cuda",dtype=torch.float16)
kv=torch.randn(32,512,device="cuda",dtype=torch.float16)
o=torch.zeros(16,32,device="cuda",dtype=torch.float32)
try:
    k=build(); k(q,kv,o); torch.cuda.synchronize()
    ref=q.float()@kv.float().T
    print("max_abs", (o-ref).abs().max().item(), "ref_scale", ref.abs().max().item())
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
