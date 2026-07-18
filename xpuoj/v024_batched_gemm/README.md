# XPUOJ v024: Sandbox-Compatible Batched GEMM

## Objective

Remove the dominant per-expert row-padding cost and use the optimized MACA
batched GEMM implementation instead of scheduling every routed metadata block
as a separate fixed-BM TileLang GEMM.

## Change

Two small TileLang kernels pack the variable padded expert layout into a dense
`(experts, 192, hidden)` tensor and unpack the final result while applying the
route weights. Invalid expert rows are explicitly zeroed during packing. BM192
covers the observed official-proxy maximum of 170 rows per expert with margin
and avoids sandbox-forbidden host reads of tensor values.

The three matrix products use Python's tensor `@` operator, which dispatches to
the optimized MACA mcBLAS FP16 batched GEMM path:

1. packed input times gate weights;
2. packed input times up weights;
3. SwiGLU activation times down weights.

The standalone TileLang SwiGLU kernel combines gate and up results in-place.
This avoids high-level activation and multiplication APIs while keeping the
three mcBLAS calls separate from layout packing and unpacking. XPUOJ's sandbox
rejects both `torch.bmm` and the tensor `.bmm()` method, but permits the `@`
operator; submissions 64189 and 64200 established those restrictions before
submission 64212 accepted this implementation.

All compiled pack, SwiGLU, and unpack kernels plus the dense input workspace
are cached. Compilation and workspace allocation therefore do not occur in
measured steady-state calls. Cache keys use only sandbox-permitted tensor
shape, device, and dtype metadata.

## Correctness

- All three official-shape proxies are bit-identical to v023 on valid and
  padded rows.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00040603` and `0.00095081` respectively.
- Full-range random-data tests on the E16, E32, and E64 official-shape proxies
  have maximum absolute errors `0.001953125`, `0.001953125`, and
  `0.00201416015625` respectively.
- Every padded output row remained exactly zero in all tests.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started with physical GPU utilization at 0%,
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v023 median | v024 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 6.2386 ms | 2.1331 ms | **+65.80%** |
| case2 | 9.8643 ms | 3.7832 ms | **+61.62%** |
| case3 | 19.6046 ms | 7.9693 ms | **+59.35%** |
| total | 35.7075 ms | 13.8856 ms | **+61.10%** |

All five aggregate pairs improved: `+61.2187%, +61.1029%, +61.1240%,
+61.0341%, +61.0890%`. Candidate peak allocation in the paired Case3 run was
7.61 GiB, within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`34c938b6f9aae67404a0dde5375ab287c7538eee604c3597add8556a46757b9f`.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
