# XPUOJ v001: 64-Row Tail Specialization

## Objective

Reduce padding-only GEMM work without changing the official 128-row routing
metadata, tensor layouts, numerical path, or public `run_kernel` contract.

## Design

- Full routed blocks (`actual_rows == 128`) use the official `128 x 128`
  schedule unchanged.
- Partial routed blocks are split into two 64-row sub-blocks.
- A sub-block only enters the K-loop when it contains at least one valid row.
- FC1 and FC2 use the same full/tail mapping.
- Padded tensors use padded row indices; route weights continue to use
  `group_offsets[expert] + token_offset`.

For an expert with 142 valid rows, the baseline computes 256 M rows per GEMM.
This candidate computes one full 128-row block and one 64-row tail block, or
192 M rows per GEMM. This is a 25% reduction in scheduled M-row work for that
expert before accounting for branch and launch overhead.

## Files

- `submission.py`: paste this complete file into an XPUOJ TileLang submission.
- `../check_submission.py`: local CUDA correctness and launch-stability check.
- `../../tests/test_xpuoj_v001_tail_bm64.py`: static contract and row-coverage
  regression tests.

The repository-root `Euraka_fusedmoe.py` and all files under `tile/` remain
unchanged.

## Verification

- Static tests cover group sizes around 64-row and 128-row boundaries.
- Local TileLang 0.1.9/NVRTC compilation succeeds on an RTX 4060 Laptop GPU.
- CUDA correctness passes for counts `[142, 65, 128]`, including padded output
  rows, with maximum absolute error `0.001953125` against an FP32 PyTorch
  reference.
- Repeated local launches complete successfully.

Local CUDA timing is not C500 evidence. XPUOJ correctness and per-case timing
were measured after the local checks:

```text
Official baseline: 65 ms, 50 points
v001 tail BM64:    97 ms, 39.67 points
Latency change:    +49.2%
```

Decision: **rejected**. The extra full/tail kernel regions and expanded CTA
grids cost substantially more on C500 than the reduced tail-row MMA work saves.
This version must not be used as the parent of later candidates.
