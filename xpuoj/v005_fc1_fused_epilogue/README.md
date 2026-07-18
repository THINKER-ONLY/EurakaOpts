# XPUOJ v005: Fused Valid-Row FC1 Epilogue

## Objective

Remove one FC1 fragment traversal and skip activation work for padded rows
without changing the 56.33-point v004 GEMM schedules.

## Parent Evidence

`v004_fc1_shared_weight` scored 56.33 on XPUOJ with case scores 57/56/56 and
a displayed total time of 52 ms. This version uses v004 as its direct parent.

## Single Change

v004 performs two FC1 epilogue loops:

```text
all 128 rows: SiLU(gate) and gate * up
valid rows:   store up_logits
```

v005 performs one loop:

```text
valid rows: SiLU(gate), gate * up, and store up_logits
```

For an average 142-token expert occupying two 128-row blocks, this reduces the
epilogue element work from 256 rows to about 142 rows. FC1/FC2 GEMMs, the single
FC1 shared weight buffer, serial K-loop, barriers, FC2 BN256, CTA grids, kernel
count, dtypes, metadata indexing, caches, and public API are unchanged.

## Verification

- AST tests replace only the FC1 epilogue and prove all other v004 source
  structure is unchanged.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- Maximum absolute error is `0.001953125` against the FP32 reference for both
  dimension pairs, and padded output rows remain zero.

The three GEMMs dominate runtime, so this candidate is expected to have a much
smaller effect than v004.

## XPUOJ Result

```text
Status:          Accepted
Total score:     56.33
Displayed time:  52 ms
```

The score and displayed time are unchanged from v004, so the reduced epilogue
work has no visible benefit at the judge's measurement resolution. Decision:
**keep as a correct experiment, but do not use it as the parent of the next
optimization**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
