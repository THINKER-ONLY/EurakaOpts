# MLA Decode v013: Shared KV for Output Part 0

## Objective

Remove a redundant KV-cache read from v012's dominant QK plus output-part-0
kernel.

## Change

- retain the full 512-column KV tile already loaded into `KV_shared` for the
  QK GEMM;
- pass the `KV_shared[:, 0:128]` buffer region directly to the output-part-0
  PV GEMM;
- remove the second global-memory copy of those 128 columns and the separate
  `V_shared` allocation;
- retain v012's prefix-scaled weight materialization, split policy, remaining
  PV partitions, direct path, and final reduction.

For the fixed 16x32 tile, QK-kernel dynamic shared memory falls from about 63
KiB to 55 KiB. More importantly, each loop iteration no longer reloads the
first 128 KV columns after the same 512-column tile was already read for QK.

## Correctness

- All 31 official outputs are bitwise identical to v012.
- The standard two-input PyTorch-reference checks retain maximum absolute
  errors `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Profiler Evidence

PyTorch's MACA activity profiler measured five calls of the B32/64K shape:

| Kernel | v012 average | v013 average | Change |
| --- | ---: | ---: | ---: |
| QK + output part 0 | 5.177 ms | 4.518 ms | **-12.7%** |
| output parts 1-3 | 1.617 ms | 1.616 ms | unchanged |
| split-K reduction | 0.0053 ms | 0.0054 ms | unchanged |

The QK plus part-0 kernel was 76.1% of v012's B32/64K device time, confirming
that the removed load was on the dominant path.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape/group | v012 | v013 | Improvement |
| --- | ---: | ---: | ---: |
| B2, context 8K | 0.10813 ms | 0.10294 ms | **+4.67%** |
| B4, context 2K | 0.08085 ms | 0.07470 ms | **+7.28%** |
| B8, context 2K | 0.11149 ms | 0.10240 ms | **+8.06%** |
| B16, context 32K | 1.71743 ms | 1.56142 ms | **+9.13%** |
| B32, context 64K | 6.84872 ms | 6.17969 ms | **+9.77%** |
| All B2 shapes | 1.10613 ms | 1.02085 ms | **+7.71%** |
| All B4 shapes | 1.87625 ms | 1.71187 ms | **+8.76%** |
| All B8 shapes | 3.44829 ms | 3.13697 ms | **+9.03%** |
| All B16 shapes | 6.61862 ms | 6.00932 ms | **+9.21%** |
| All B32 shapes | 13.41983 ms | 12.17293 ms | **+9.29%** |
| 31-case total | 27.83301 ms | 25.40165 ms | **+8.78%** |

All five aggregate samples improved: `+8.4845%, +8.7789%, +8.7798%,
+8.7210%, +9.0093%`. No per-shape median regressed.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`c0cd7cbe61e5ec0d2ad1e6fe5d7def5a013c192d3edd97a54b80d62b95383d15`.
