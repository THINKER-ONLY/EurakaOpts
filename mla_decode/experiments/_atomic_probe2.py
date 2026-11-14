"""Probe 2: isolate atomic contention vs atomic throughput; calibrate stores.

All variants move 64 CTAs x 8192 fp32 = 2MB.
  store_c   : coalesced plain stores to [64,8192]        (bandwidth reference)
  at_nc     : atomics to [64,8192]  (0-way contention, same 2MB)
  at_8      : atomics to [8,8192]   (8-way contention/address)
  at_64     : atomics to [1,8192]   (64-way contention)  <- the naive fused-merge design
"""
import sys
sys.path.insert(0, "/data/tilelang-metax")
import torch
import tilelang
import tilelang.language as T

NT, NCTA = 256, 64
NF = NCTA * 8192  # floats per CTA


def _mk_slot(nslot, coalesced):
    @tilelang.jit
    def build(ncta):
        @T.prim_func
        def main(Out: T.Tensor([nslot * 8192], T.float32), Src: T.Tensor([ncta, 8192], T.float32)):
            with T.Kernel(ncta, threads=NT) as bx:
                tx = T.get_thread_binding(0)
                if coalesced:
                    for k in T.serial(32):
                        Out[bx * 8192 + k * 256 + tx] = Src[bx, k * 256 + tx]
                else:
                    for k in T.serial(32):
                        T.atomic_add(Out[(bx % nslot) * 8192 + k * 256 + tx], Src[bx, k * 256 + tx])
        return main
    return build


@tilelang.jit
def mk_empty(ncta):
    @T.prim_func
    def main(Out: T.Tensor([8192], T.float32), Src: T.Tensor([ncta, 8192], T.float32)):
        with T.Kernel(ncta, threads=NT) as bx:
            tx = T.get_thread_binding(0)
            if tx == 100000:
                Out[0] = Src[bx, 0]
    return main


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000.0


def main():
    src = torch.ones(NCTA, 8192, dtype=torch.float32, device="cuda")
    bufs = {n: torch.zeros(n * 8192, dtype=torch.float32, device="cuda") for n in (1, 8, 64)}
    k_em = mk_empty(NCTA)
    t_em = bench(lambda: k_em(bufs[1], src))
    print(f"empty                        {t_em:7.2f} us")
    k_sc = _mk_slot(64, True)(NCTA)
    print(f"store coalesced  (2MB)       {bench(lambda: k_sc(bufs[64], src)):7.2f} us  net {bench(lambda: k_sc(bufs[64], src))-t_em:6.2f}")
    for ns in (64, 8, 1):
        k = _mk_slot(ns, False)(NCTA)
        t = bench(lambda: k(bufs[ns], src))
        print(f"atomic {ns:2d}-way      (2MB)  {t:7.2f} us  net {t-t_em:6.2f}")
    torch.cuda.synchronize()
    print("sanity sums:", bufs[1][0].item(), bufs[8][0].item(), bufs[64][0].item())


if __name__ == "__main__":
    main()
