import math
import time

import torch
import tilelang
import tilelang.language as T


E = 64
K = 2048
N = 2048
RAW_M = 9088
DENSE_M = 192


def build_counts():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(81394)
    assignments = torch.randint(0, E, (RAW_M,), generator=generator, dtype=torch.int64)
    return tuple(torch.bincount(assignments, minlength=E).tolist())


COUNTS = build_counts()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def build_grouped(block_m, block_n, block_k):
    batch_sum = sum(COUNTS)
    total_m_blocks = sum((size + block_m - 1) // block_m for size in COUNTS)

    @T.prim_func
    def kernel(
        A: T.Tensor((batch_sum, K), T.float16),
        B: T.Tensor((E, K, N), T.float16),
        C: T.Tensor((batch_sum, N), T.float16),
        batch_sizes: T.Tensor((E,), T.int32),
        batch_offsets: T.Tensor((E + 1,), T.int32),
        batch_padded_offsets: T.Tensor((E + 1,), T.int32),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_n), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)
            current_expert = T.alloc_var(T.int32)

            padded_start = bx * block_m
            current_expert = 0
            for expert_id in range(E):
                if padded_start >= batch_padded_offsets[expert_id]:
                    current_expert = expert_id

            group_size = batch_sizes[current_expert]
            row_start = (
                padded_start
                - batch_padded_offsets[current_expert]
                + batch_offsets[current_expert]
            )
            actual_rows = T.max(
                0,
                T.min(
                    block_m,
                    group_size
                    + batch_padded_offsets[current_expert]
                    - padded_start,
                ),
            )

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(A[row_start, ko * block_k], A_shared)
                T.copy(B[current_expert, ko * block_k, by * block_n], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for i, j in T.Parallel(block_m, block_n):
                if i < actual_rows:
                    C[row_start + i, by * block_n + j] = C_local[i, j]

    return kernel


def build_metadata(block_m, device):
    offsets = [0]
    padded_offsets = [0]
    for count in COUNTS:
        offsets.append(offsets[-1] + count)
        padded_offsets.append(padded_offsets[-1] + math.ceil(count / block_m) * block_m)
    return (
        torch.tensor(COUNTS, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(padded_offsets, dtype=torch.int32, device=device),
    )


def time_call(callable_obj, iterations=5):
    for _ in range(2):
        callable_obj()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        callable_obj()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    torch.manual_seed(31)
    device = "cuda"
    compact = torch.randn((RAW_M, K), device=device, dtype=torch.float16)
    dense = torch.zeros((E, DENSE_M, K), device=device, dtype=torch.float16)
    weights = torch.randn((E, K, N), device=device, dtype=torch.float16)
    output = torch.empty((RAW_M, N), device=device, dtype=torch.float16)

    offset = 0
    for expert_id, count in enumerate(COUNTS):
        dense[expert_id, :count].copy_(compact[offset : offset + count])
        offset += count

    dense_ms = time_call(lambda: dense @ weights)
    print("COUNTS", min(COUNTS), max(COUNTS), sum(COUNTS), flush=True)
    print("DENSE_M192_MS", dense_ms, flush=True)

    for block_m in (64, 128):
        sizes, offsets, padded_offsets = build_metadata(block_m, device)
        try:
            kernel = build_grouped(block_m, 256, 32)
            grouped_ms = time_call(
                lambda: kernel(
                    compact,
                    weights,
                    output,
                    sizes,
                    offsets,
                    padded_offsets,
                )
            )
            print(
                "GROUPED",
                block_m,
                "MS",
                grouped_ms,
                "RATIO",
                grouped_ms / dense_ms,
                "PADDED_ROWS",
                int(padded_offsets[-1].item()),
                flush=True,
            )
        except Exception as exc:
            print("GROUPED", block_m, "FAIL", type(exc).__name__, str(exc)[:500], flush=True)


if __name__ == "__main__":
    main()
