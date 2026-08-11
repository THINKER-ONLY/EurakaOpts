import time

import torch
import tilelang
import tilelang.language as T


E = 64
M = 128
K = 2048
N = 2048


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def build(block_n, block_k, policy):
    @T.prim_func
    def kernel(
        A: T.Tensor((E, M, K), T.float16),
        B: T.Tensor((E, K, N), T.float16),
        C: T.Tensor((E, M, N), T.float16),
    ):
        with T.Kernel(E, T.ceildiv(N, block_n), threads=256) as (e, bn):
            A_shared = T.alloc_shared((M, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            accum = T.alloc_fragment((M, block_n), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(A[e, 0, ko * block_k], A_shared)
                T.copy(B[e, ko * block_k, bn * block_n], B_shared)
                T.gemm(A_shared, B_shared, accum, policy=policy)
            T.copy(accum, C[e, 0, bn * block_n])

    return kernel


def main():
    torch.manual_seed(29)
    device = "cuda"
    A = torch.randn((E, M, K), device=device, dtype=torch.float16)
    B = torch.randn((E, K, N), device=device, dtype=torch.float16)
    C = torch.empty((E, M, N), device=device, dtype=torch.float16)

    for _ in range(2):
        reference = A @ B
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        reference = A @ B
    torch.cuda.synchronize()
    bmm_ms = (time.perf_counter() - start) * 1000 / 5
    print("BMM_MS", bmm_ms, flush=True)

    variants = [
        ("sq256k32", 256, 32, T.GemmWarpPolicy.Square),
        ("row256k32", 256, 32, T.GemmWarpPolicy.FullRow),
        ("col256k32", 256, 32, T.GemmWarpPolicy.FullCol),
        ("sq256k64", 256, 64, T.GemmWarpPolicy.Square),
        ("row256k64", 256, 64, T.GemmWarpPolicy.FullRow),
        ("row128k64", 128, 64, T.GemmWarpPolicy.FullRow),
    ]
    for name, block_n, block_k, policy in variants:
        try:
            kernel = build(block_n, block_k, policy)
            for _ in range(2):
                kernel(A, B, C)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                kernel(A, B, C)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000 / 5
            error = (C.float() - reference.float()).abs()
            print(
                "VARIANT",
                name,
                "MS",
                elapsed_ms,
                "RATIO",
                elapsed_ms / bmm_ms,
                "MAX",
                float(error.max().item()),
                "MEAN",
                float(error.mean().item()),
                flush=True,
            )
        except Exception as exc:
            print("VARIANT", name, "FAIL", type(exc).__name__, str(exc)[:300], flush=True)


if __name__ == "__main__":
    main()
