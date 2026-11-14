import torch, tilelang
import tilelang.language as T

@tilelang.jit
def build(nblocks):
    @T.prim_func
    def main(state: T.Tensor([1], "int32"), out: T.Tensor([64], "int32")):
        with T.Kernel(nblocks, threads=64) as bx:
            tx = T.get_thread_binding(0)
            if tx == 0:
                old = T.atomic_add(state[0], 1, memory_order="acq_rel", return_prev=True)
                target = (old // nblocks + 1) * nblocks
                while T.atomic_add(state[0], 0, memory_order="acq_rel", return_prev=True) < target:
                    T.evaluate(0)
            T.sync_threads()
            out[bx] = bx + 100
    return main

state = torch.zeros(1, dtype=torch.int32, device="cuda")
out = torch.zeros(64, dtype=torch.int32, device="cuda")
k = build(64)
for i in range(3):
    k(state, out)
    torch.cuda.synchronize()
    print("call", i, "state", state.item(), "out_ok", out.cpu().tolist()[:5], out.cpu().tolist()[-3:])
