# XPUOJ v004: FC1 Shared Weight Tile Reuse

## Objective

Reduce explicit FC1 shared weight storage from two 128 x 128 FP16 tiles to one
reused tile while preserving the 51-point v003 FC2 BN256 schedule.

## Design

- Gate and Up keep separate FP32 accumulators.
- One 32 KiB `weight_shared` tile replaces the two 32 KiB Gate/Up tiles.
- Each K iteration loads Gate weights, executes Gate GEMM, synchronizes,
  overwrites the tile with Up weights, executes Up GEMM, and synchronizes.
- The FC1 K-loop uses `T.serial`. TileLang rejects two overlapping writes to
  one shared buffer inside `T.Pipelined`, even with `num_stages=1`.
- FC2, BN256 shape selection, CTA grids, kernel count, arithmetic, dtypes,
  metadata indexing, caches, and public API remain unchanged from v003.

## Compile Investigation

The first implementation retained `T.Pipelined` and failed during TileLang
PipelinePlanning:

```text
Multiple writes to overlapping buffer regions detected.
Stage 1 and stage 3 are both writing to buffer 'weight_shared'.
```

The final serial loop with two explicit barriers compiles and matches the
lifecycle previously validated by the repository's archived shared-weight
experiment.

## Verification

- AST tests normalize the shared-buffer lifecycle and prove that it is the
  only difference from v003.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- Maximum absolute error is `0.001953125` against the FP32 reference for both
  dimension pairs.

## Evidence And Risk

An older-interface RTX 4060 experiment of the same lifecycle produced paired
median speedups of about 1.002-1.003, but its distributions crossed 1.0, so it
did not establish an improvement. This XPUOJ version tests whether C500 crosses
an occupancy threshold when FC1 shared storage drops by 32 KiB.

The trade-off is explicit: lower shared-memory pressure versus loss of the
planned FC1 loop and two barriers per K iteration. No C500 speedup is claimed
before XPUOJ measurement.

## XPUOJ Result

```text
Status:          Accepted
Total score:     56.33
Displayed time:  52 ms
Memory:          22.2 G
Case scores:     57 / 56 / 56
Case times:      9 / 15 / 28 ms (display-rounded)
```

The result is substantially better than the 51-point v003 parent and the
50-point official baseline. All three testcases improved, which supports the
hypothesis that the 32 KiB FC1 shared-memory reduction crosses a C500 occupancy
threshold despite the serial loop and barriers. Decision: **accepted as the
next optimization baseline**.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
