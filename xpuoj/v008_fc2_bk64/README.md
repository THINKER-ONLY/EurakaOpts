# XPUOJ v008: FC2 BK64

## Objective

Test whether reducing the official-shape FC2 shared-weight tile from 64 KiB to
32 KiB improves C500 residency without changing FC2's output tile or CTA grid.

## Parent Evidence

`v007_fc1_bk64_dual_buffer` is the current best result at 56.67 points and
51 ms, with case scores 58/56/56 and times 8/15/28 ms. This version uses v007
as its direct parent.

## Single Change

The `block_k` constant used only by FC2 changes from 128 to 64. On all
official shapes, FC2 retains BN256:

```text
v007: down_shared = 256 x 128 x FP16 = 64 KiB
v008: down_shared = 256 x  64 x FP16 = 32 KiB
```

FC2 threads, BM128, BN256, pipeline stages, CTA grid, FLOPs, total global
input/weight bytes, accumulator shape, epilogue, and route-weight indexing are
unchanged. The accepted v007 FC1 BK64 dual-buffer pipeline is unchanged.

## Hypothesis And Risk

The v004 FC1 result showed that reducing a 64 KiB shared-weight footprint to
32 KiB can cross a C500 resource threshold. This version tests the same
resource mechanism independently in FC2 without adding kernels or CTAs.

The trade-off is twice as many FC2 K-loop iterations: case 1 changes from 64
to 128 iterations because `intermediate=8192`, while cases 2 and 3 change
from 16 to 32 because `intermediate=2048`. Loop and pipeline-control overhead
can offset any residency benefit.

## Verification

- AST tests require FC2 BK64 and preserve the v007 FC1 schedule.
- Full AST normalization proves the BK constant is the only source difference
  from v007.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- `(hidden=2048, intermediate=8192)`: maximum absolute error
  `0.001953125`, mean absolute error `5.124584276927635e-05`.
- `(hidden=7168, intermediate=2048)`: maximum absolute error
  `0.001953125`, mean absolute error `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.

## XPUOJ Result

```text
Status:          Accepted
Total score:     61.33
Displayed time:  43 ms
Memory:          22.2 G
Case scores:     62 / 61 / 61
Case times:      7 / 12 / 24 ms (display-rounded)
```

The v007 parent scored 56.67 at 51 ms, with case scores 58/56/56 and times
8/15/28 ms. FC2 BK64 improves every testcase, reduces displayed total time by
8 ms (15.7%), and raises the total score by 4.66 points.

Because the production diff is exactly one constant, the result strongly
supports the hypothesis that v007's 64 KiB FC2 shared-weight tile constrained
C500 residency and/or throughput. Reducing it to 32 KiB produces a much larger
benefit than the cost of doubling the FC2 K-loop iteration count. Aggregate
judge timing does not distinguish residency from other resource effects, so
that finer attribution remains open.

Decision: **accepted as the new best baseline**. Preserve FC2 BK64 and the
v007 FC1 BK64 dual-buffer schedule in subsequent experiments.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
