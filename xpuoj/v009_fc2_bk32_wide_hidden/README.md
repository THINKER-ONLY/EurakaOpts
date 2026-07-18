# XPUOJ v009: FC2 BK32 for Wide-Hidden Shape

## Objective

Test the next FC2 shared-memory threshold only on the official wide-hidden
shape while preserving v008's case-1 BK64 path.

## Parent Evidence

`v008_fc2_bk64` is the current best result at 61.33 points and 43 ms, with
case scores 62/61/61 and times 7/12/24 ms. Its one-line FC2 BK128-to-BK64
change improved all testcases and reduced the official-shape FC2 shared-weight
tile from 64 KiB to 32 KiB.

## Single Change

FC2 block K becomes shape-selective:

```text
(hidden=7168, intermediate=2048): BK32
all other shapes:                    BK64
```

This keeps case 1 on the accepted BK64 schedule. Cases 2 and 3 retain BM128,
BN256, 256 threads, one pipeline stage, the same CTA grid, FLOPs, total global
bytes, accumulator, epilogue, and route-weight indexing, while reducing
`down_shared` from 32 KiB to 16 KiB.

The accepted FC1 BK64 dual-buffer pipeline is unchanged.

## Hypothesis And Risk

If 32 KiB still limits FC2 resident CTAs on the wide-hidden shape, BK32 can
expose another resource threshold. The trade-off is that FC2 K-loop iterations
increase from 32 to 64 for cases 2 and 3. If registers already cap residency,
the additional loop and pipeline-control work will be pure overhead.

BK32 is compatible with the MetaX FP16 micro-K=16 lowering and is used by
local MetaX TileLang examples.

## Verification

- Tests evaluate the dispatch expression for both official dimension pairs and
  a generic fallback shape.
- Full AST normalization proves the dispatch expression is the only source
  difference from v008.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- BK64 fallback `(hidden=2048, intermediate=8192)`: maximum absolute error
  `0.001953125`, mean absolute error `5.124584276927635e-05`.
- BK32 branch `(hidden=7168, intermediate=2048)`: maximum absolute error
  `0.001953125`, mean absolute error `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.

## XPUOJ Result

```text
Status:          Accepted (correctness)
Total score:     60.67
Displayed time:  45 ms
Memory:          22.2 G
Case scores:     62 / 60 / 60
Case times:      7 / 13 / 25 ms (display-rounded)
```

The v008 parent scored 61.33 at 43 ms, with case scores 62/61/61 and times
7/12/24 ms. Case 1 is exactly unchanged, confirming that the BK64 fallback
preserved the accepted path. The BK32 cases 2 and 3 each lose one point and
increase by one displayed millisecond.

This result rejects the hypothesis that reducing FC2 shared weight storage
from 32 KiB to 16 KiB exposes another useful residency threshold for the
wide-hidden shape. The evidence is consistent with another resource, such as
the large BN256 FP32 accumulator/register footprint, already limiting
residency; under that condition the doubled K-loop and pipeline-control work
becomes net overhead. Judge timing alone cannot identify the exact limiting
resource, so that finer attribution remains provisional.

Decision: **rejected**. Keep v008 as the best baseline, retain FC2 BK64 for
all official shapes, and do not explore BK16 without contrary target-profile
evidence.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
