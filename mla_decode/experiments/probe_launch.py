import tilelang, tilelang.language as T
@tilelang.jit
def build():
    @T.prim_func
    def main(A: T.Tensor([1], "float16")):
        with T.Kernel(64, threads=256) as (bx,):
            A[0] = T.cast(1.0, "float16")
    return main
import torch, statistics
k = build(); a = torch.zeros(1, device="cuda", dtype=torch.float16)
for _ in range(20): k(a)
torch.cuda.synchronize()
vals=[]
for _ in range(50):
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(50): k(a)
    e.record(); torch.cuda.synchronize()
    vals.append(s.elapsed_time(e)/50*1000)
print("empty-kernel launch: median %.2fus min %.2fus" % (statistics.median(vals), min(vals)))
