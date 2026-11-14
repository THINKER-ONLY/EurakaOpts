import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build():
    @T.prim_func
    def main(X: T.Tensor([128], "int32"), Y: T.Tensor([128], "int32")):
        with T.Kernel(1, threads=256) as _:
            bar = T.alloc_shared([1], "int32")
            S = T.alloc_shared([128], "int32")
            tx = T.get_thread_binding(0)
            if tx == 0:
                bar[0] = 0
            T.sync_threads()
            # phase 1: group0 writes S
            if tx < 128:
                S[tx] = X[tx] * 2
            # partial barrier: all 128 compute threads arrive
            if tx < 128:
                prev = T.atomic_add(bar[0], 1, memory_order="acq_rel", return_prev=True)
                while T.atomic_add(bar[0], 0, memory_order="acq_rel", return_prev=True) < 128:
                    T.evaluate(0)
            # group1 must not touch S; but it must not pass the next block-wide sync early
            T.sync_threads()
            # phase 2: group0 reads S
            if tx < 128:
                Y[tx] = S[127 - tx] + 1
    return main

x=torch.arange(128,dtype=torch.int32,device="cuda")
y=torch.zeros(128,dtype=torch.int32,device="cuda")
try:
    k=build(); k(x,y); torch.cuda.synchronize()
    ref=(x*2).flip(0)+1
    print("ok", torch.equal(y,ref))
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
