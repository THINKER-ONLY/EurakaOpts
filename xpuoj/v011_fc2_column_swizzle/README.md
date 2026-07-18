# XPUOJ v011: FC2 Column Swizzle

## Objective

Test whether scheduling adjacent FC2 CTAs to reuse the 32 KiB Down weight tile
improves C500 cache behavior while preserving the accepted v008 compute and
resource configuration.

## Parent Evidence

`v008_fc2_bk64` is the current best result at 61.33 points and 43 ms, with
case scores 62/61/61 and displayed case times 7/12/24 ms.

## Single Change

Only the FC2 threadblock swizzle order changes:

```text
v008: row order,    panel size 10
v011: column order, panel size 10
```

FC1 remains on row order with panel size 10. FC1 and FC2 BM/BN/BK, threads,
pipeline stages, CTA grids, shared-memory allocations, accumulators, FLOPs,
workspace, arithmetic, metadata indexing, cache behavior, and public API are
otherwise unchanged.

## Hypothesis And Risk

At FC2 BK64, each CTA loads a 16 KiB Up activation tile and a 32 KiB Down
weight tile per K iteration. Row order gives immediate activation reuse but
separates same-expert weight reuse by 8 or 10 CTAs. Column order makes the
same-expert weight reuse adjacent while keeping activation reuse within a
10-CTA panel.

The hypothesis is falsified if C500 prefers the existing activation broadcast
pattern or does not preserve the expected launch-order locality. No benefit is
claimed without an XPUOJ result.

## Local Verification

- Full AST normalization proves the FC2 swizzle `order="column"` keyword is
  the only production-source difference from v008.
- Python syntax checks and all 31 `test_xpuoj_*.py` regressions pass.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- `(hidden=2048, intermediate=8192)`: maximum absolute error
  `0.001953125`, mean absolute error `5.124584276927635e-05`.
- `(hidden=7168, intermediate=2048)`: maximum absolute error
  `0.001953125`, mean absolute error `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.
- Generated CUDA contains one `rasterization2DRow<10>` for FC1 and one
  `rasterization2DColumn<10>` for FC2.
- Generated launch wrappers retain 32,768 bytes of dynamic shared memory and
  `__launch_bounds__(256, 1)` for both stages.
- Source SHA-256:
  `B6B3DD0C2334190ED3AF747EA5D140C0219EF538EA6CAFAC1A5144E4CD6841E0`.

The broader workspace suite ran 137 tests: 136 passed, and the one unrelated
legacy archive test still expects `Euraka_fusedmoe.py` at the repository root
instead of its current `EurakaOpts/` location. No unrelated source or test was
changed.

## XPUOJ Result

```text
Status:          Accepted
Total score:     61.33
Displayed time:  43 ms
Memory:          22.2 G
Case scores:     62 / 61 / 61
Case times:      7 / 12 / 24 ms (display-rounded)
```

The result is identical to v008 in total score, displayed time, and every
per-case score and time. Changing FC2 from row order to column order therefore
has no visible performance effect at the judge's measurement resolution.

## Decision

Decision: **rejected as a performance baseline**. Keep v008 as the best
baseline and retain FC2 row order with panel size 10. The experiment is still
useful negative evidence: changing only the FC2 L2-reuse preference does not
improve any official testcase on C500.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
