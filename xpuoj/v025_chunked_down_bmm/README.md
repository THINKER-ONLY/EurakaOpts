# XPUOJ v025: Chunked Down BMM

## Objective

Reduce the dominant E32/E64 down-projection time while preserving v024's
sandbox-compatible `@` batched GEMM path.

## Change

Split the down projection into batches of 16 experts. Each chunk is computed
with the tensor `@` operator and passed directly to a chunk-specialized
TileLang unpack kernel, which applies route weights and writes the corresponding
experts to the padded output layout. This avoids `torch.bmm(out=...)`, tensor
`.bmm()`, concatenation, and a full packed-output workspace.

Case1 has exactly 16 experts and therefore remains one down BMM. Case2 uses two
chunks and Case3 uses four. Pack, SwiGLU, and FC1 remain identical to v024.

## Correctness

- All three official-shape proxies are bit-identical to v024.
- The `(2048,8192)` and `(7168,2048)` FP32 oracles retain maximum absolute
  errors `0.00057054` and `0.00095081` respectively.
- Full-range random E16/E32/E64 tests have maximum absolute errors
  `0.001953125`, `0.001953125`, and `0.00201416015625` respectively.
- Every padded output row remained exactly zero.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. The accepted run started at 0% physical GPU utilization and
used five warmups, ten measured calls per sample, and five alternating paired
samples:

| Case | v024 median | v025 median | Median paired improvement |
| --- | ---: | ---: | ---: |
| case1 | 2.1324 ms | 2.1202 ms | **+0.60%** |
| case2 | 3.7896 ms | 3.6920 ms | **+2.58%** |
| case3 | 7.9795 ms | 6.7405 ms | **+15.54%** |
| total | 13.9014 ms | 12.5527 ms | **+9.65%** |

All five aggregate pairs improved: `+9.6458%, +9.5519%, +9.6373%,
+9.7582%, +9.7815%`. Candidate peak allocation in Case3 was 7.71 GiB,
within the 32 GiB sGPU quota.

Decision: **accepted as the local baseline and selected for XPUOJ testing**.

The archived submission SHA-256 is
`33b1649a40891b1e987181f6cb819aa7e3cc88eacf72d52ecb83713e66e518ad`.

## XPUOJ Result

Submission `#64237` was accepted with **89.67 points**:

| Case | Time | Display score |
| --- | ---: | ---: |
| case1 | 2.079 ms | 90 |
| case2 | 3.655 ms | 89 |
| case3 | 6.693 ms | 90 |
| total | 12.427 ms | **89.67** |

This improves the accepted v024 total from 13.778 ms by **9.81%** and raises
the score from 88.67 to 89.67.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
