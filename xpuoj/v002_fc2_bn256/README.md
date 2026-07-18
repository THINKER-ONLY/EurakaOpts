# XPUOJ v002: FC2 BN256 for Wide-Hidden Cases

## Objective

Reduce FC2 CTA count and repeated `up_logits` loads without changing the
official two-kernel structure or the 128-row routing grid.

## Design

- `(hidden=7168, intermediate=2048)` selects `fc2_block_n=256`.
- All other shapes, including `(2048, 8192)`, retain `fc2_block_n=128`.
- FC1 remains at `128 x 128` for every testcase.
- BK, threads, pipeline stages, numerical operations, caches, and tensor
  indexing are unchanged from the official baseline.

For testcases 2 and 3, the FC2 N grid changes from:

```text
7168 / 128 = 56 tiles
7168 / 256 = 28 tiles
```

Each FC2 CTA therefore reuses one `up_logits` tile across twice as many output
columns. The trade-off is a 2x larger FC2 output accumulator and down-weight
shared tile, which can reduce occupancy or cause spilling on C500.

## Verification

- Static tests require exactly two TileLang kernel regions and reject any tail
  schedule inherited from v001.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- Maximum absolute error is `0.001953125` against an FP32 PyTorch reference for
  both dimension pairs.
- Local CUDA timing is not used as C500 performance evidence.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.

## XPUOJ Result

```text
Status:          Accepted
Total score:     50.67
Displayed time:  63 ms
Memory:          22.2 G
Case scores:     50 / 51 / 51
Case times:      11 / 18 / 34 ms (display-rounded)
```

The official baseline scored 50 with a displayed total time of 65 ms. Case 1
retained BN128 and remained at 50 points; cases 2 and 3 selected BN256 and both
reached 51 points. Decision: **accepted as the next optimization baseline**.
