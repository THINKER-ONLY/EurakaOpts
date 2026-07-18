# XPUOJ v003: FC2 BN256 for All Official Shapes

## Objective

Extend the C500-confirmed FC2 BN256 schedule from testcases 2 and 3 to
testcase 1 without changing any other kernel behavior.

## Parent Evidence

`v002_fc2_bn256` scored 50.67 on XPUOJ. Its BN128 testcase scored 50, while
both BN256 testcases scored 51. This version uses v002 as its direct parent.

## Single Change

The FC2 selector now returns 256 for both official dimension pairs:

```text
(hidden=7168, intermediate=2048): BN256, unchanged from v002
(hidden=2048, intermediate=8192): BN256, changed from BN128
```

For testcase 1, the FC2 N grid decreases from 16 tiles to 8 tiles. FC1, BK,
threads, pipeline stages, caches, tensor indexing, numerical operations, the
128-row routing grid, and the two-kernel launch structure are byte-equivalent
to v002 after normalizing the selector expression.

## Verification

- An AST regression test proves the selector is the only source difference
  from v002.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- Maximum absolute error is `0.001953125` against the FP32 reference for both
  dimension pairs.

The new risk is isolated to testcase 1: its FC2 has `K=8192`, so the wider
accumulator and shared weight tile may reduce occupancy enough to offset the
lower CTA count.

## XPUOJ Result

```text
Status:      Accepted
Total score: 51
```

This improves on v002's 50.67 points. Decision: **accepted as the next
optimization baseline**. Exact per-testcase timing was not available when this
result was recorded.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
