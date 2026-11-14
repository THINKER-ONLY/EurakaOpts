"""Probe: cost of the fp32 global-atomic epilogue vs the current partial-store epilogue.

Pattern mirrors v540's acc_o fragment layout exactly:
  thread tx -> head (tx&15), 8 groups g of 4 consecutive columns,
  col = (tx//64)*128 + g*16 + ((tx//16)&3)*4, plus r in 0..3
=> 8192 floats per CTA (16 heads x 512 dims).

  atomic : 64 CTAs add into one [8192] fp32 buffer  (64-way contention/address)
  astore : 64 CTAs store into [64, 8192]            (no contention, 2MB written)
"""
import sys
sys.path.insert(0, "/data/tilelang-metax")
import torch
import tilelang
import tilelang.language as T

NT = 256
NCTA = 64


def _idx(tx, g):
    return (tx % 16) * 512 + (tx // 64) * 128 + g * 16 + ((tx // 16) % 4) * 4


@tilelang.jit
def mk_atomic(ncta):
    @T.prim_func
    def main(Out: T.Tensor([8192], T.float32), Src: T.Tensor([ncta, 8192], T.float32)):
        with T.Kernel(ncta, threads=NT) as bx:
            tx = T.get_thread_binding(0)
            for g in T.serial(8):
                for r in T.serial(4):
                    T.atomic_add(Out[_idx(tx, g) + r], Src[bx, _idx(tx, g) + r])
    return main


@tilelang.jit
def mk_store(ncta):
    @T.prim_func
    def main(Out: T.Tensor([ncta, 8192], T.float32), Src: T.Tensor([ncta, 8192], T.float32)):
        with T.Kernel(ncta, threads=NT) as bx:
            tx = T.get_thread_binding(0)
            for g in T.serial(8):
                for r in T.serial(4):
                    Out[bx, _idx(tx, g) + r] = Src[bx, _idx(tx, g) + r]
    return main


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
    return s.elapsed_time(e) / n * 1000.0  # us


def main():
    dev = "cuda"
    acc = torch.zeros(8192, dtype=torch.float32, device=dev)
    src = torch.ones(NCTA, 8192, dtype=torch.float32, device=dev)
    part = torch.zeros(NCTA, 8192, dtype=torch.float32, device=dev)

    k_at = mk_atomic(NCTA)
    k_st = mk_store(NCTA)
    k_em = mk_empty(NCTA)

    t_em = bench(lambda: k_em(acc, src))
    t_at = bench(lambda: k_at(acc, src))
    t_st = bench(lambda: k_st(part, src))
    acc.zero_()
    k_at(acc, src)
    torch.cuda.synchronize()
    print(f"empty launch      {t_em:7.2f} us")
    print(f"fp32 atomic x 8192/CTA  {t_at:7.2f} us  (net {t_at-t_em:6.2f}, sum={acc[0].item():.0f})")
    print(f"fp32 store x 8192/CTA   {t_st:7.2f} us  (net {t_st-t_em:6.2f})")


if __name__ == "__main__":
    main()
