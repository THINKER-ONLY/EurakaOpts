# MLA Decode v014: Fused 576-D QK GEMM

## Objective

Reduce fixed GEMM setup overhead in v013's materialized QK kernel, especially
for short per-split contexts.

## Change

- allocate contiguous `dim + pe_dim = 576` shared-memory Q and K tiles;
- copy the 512-dimensional latent components and 64-dimensional RoPE
  components into adjacent buffer regions;
- replace the separate 512-D and 64-D QK GEMMs with one 576-D QK GEMM;
- retain v013's direct use of the first 128 KV columns for output part 0;
- retain prefix-scaled weights, remaining PV partitions, split policy, direct
  path, and final reduction.

The shared-memory byte count and score arithmetic are unchanged. The benefit
comes from removing the setup and synchronization cost of the small 64-D GEMM.

## Correctness

- All 31 official outputs are bitwise identical to v013.
- The standard two-input PyTorch-reference checks retain maximum absolute
  errors `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape/group | v013 | v014 | Improvement |
| --- | ---: | ---: | ---: |
| B2, context 8K | 0.10171 ms | 0.09876 ms | **+2.87%** |
| B4, context 2K | 0.08218 ms | 0.07772 ms | **+5.51%** |
| B8, context 2K | 0.10304 ms | 0.09866 ms | **+4.30%** |
| B16, context 32K | 1.56849 ms | 1.56180 ms | **+0.36%** |
| B32, context 64K | 6.16456 ms | 6.16123 ms | **+0.03%** |
| 24 materialized cases | 24.08192 ms | 23.95604 ms | **+0.52%** |
| 31-case total | 25.50932 ms | 25.36387 ms | **+0.63%** |

All five aggregate samples improved: `+0.3582%, +0.6365%, +0.7986%,
+0.5571%, +0.6326%`. No per-shape median regressed. The improvement decreases
as each split grows because the removed fixed GEMM overhead is amortized over
more QK work.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`f149143f8a1a58e00734cc9feadbe09a5b76b509e28aeb67839d69e3ec5e50f3`.
