# XPUOJ v010: FC2 BN128 for Wide-Hidden Shape

## Objective

Test whether halving the wide-hidden FC2 output tile relieves the accumulator
and register-pressure limit suggested by the rejected v009 BK32 experiment.

## Parent Evidence

`v008_fc2_bk64` remains the best result at 61.33 points and 43 ms, with case
scores 62/61/61 and times 7/12/24 ms. The v009 BK32 experiment preserved case
1 but regressed cases 2 and 3, showing that reducing FC2 shared weight storage
below 32 KiB without reducing the BN256 accumulator does not help.

## Single Change

FC2 block N becomes:

```text
(hidden=2048, intermediate=8192): BN256
(hidden=7168, intermediate=2048): BN128
generic fallback:                    BN128
```

For cases 2 and 3 at BK64:

```text
v008: accumulator 128 x 256 FP32, down_shared 256 x 64 FP16 = 32 KiB
v010: accumulator 128 x 128 FP32, down_shared 128 x 64 FP16 = 16 KiB
```

Case 1 keeps the accepted BN256/BK64 path. FC1, FC2 BK64, threads, pipeline
stages, FLOPs, down-weight bytes, epilogue, route-weight indexing, caches, and
the public API are unchanged.

## Hypothesis And Risk

BN128 halves the total FC2 accumulator and shared-weight footprint per CTA,
which can improve residency if registers now limit BN256 after v008's shared
memory reduction. The trade-off is twice as many N-direction CTAs and roughly
twice as many `up_logits` tile reads for cases 2 and 3.

Earlier BK128 evidence favored BN256 by one case point. This experiment is
still justified because BK64 moved FC2 into a different shared-memory resource
regime, but the historical result makes regression a material risk.

## Verification

- Tests evaluate the BN dispatch for both official shapes and a generic
  fallback.
- Full AST normalization proves the BN dispatch is the only source difference
  from v008.
- Local TileLang 0.1.9/NVRTC compilation and numerical checks pass for both
  official dimension pairs using one expert and 142 valid rows.
- BN256 case-1 branch `(hidden=2048, intermediate=8192)`: maximum absolute
  error `0.001953125`, mean absolute error `5.124584276927635e-05`.
- BN128 wide-hidden branch `(hidden=7168, intermediate=2048)`: maximum
  absolute error `0.001953125`, mean absolute error
  `4.858117245021276e-05`.
- Padded output rows remain exactly zero in both checks.

## XPUOJ Result

```text
Status:          Accepted (correctness)
Total score:     57.33
Displayed time:  51 ms
Memory:          22.2 G
Case scores:     62 / 55 / 55
Case times:      7 / 15 / 29 ms (display-rounded)
```

The v008 parent scored 61.33 at 43 ms, with case scores 62/61/61 and times
7/12/24 ms. Case 1 is exactly unchanged, confirming that its BN256 branch
preserved the accepted path. The BN128 cases 2 and 3 lose six points each and
increase by three and five displayed milliseconds, respectively.

This strongly rejects the hypothesis that reducing the BN256 accumulator is a
profitable way to expose more FC2 residency for the wide-hidden shape. Any
resource benefit from halving the accumulator and shared tile is overwhelmed
by doubling the N-direction CTA grid and repeated `up_logits` tile reads. The
result also confirms that v008's BN256/BK64 combination must be treated as a
coupled schedule rather than optimizing either dimension independently.

Decision: **rejected**. Keep v008 as the best baseline, retain FC2 BN256/BK64
for all official shapes, and do not revisit BN128 without contrary target
profiling evidence.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
