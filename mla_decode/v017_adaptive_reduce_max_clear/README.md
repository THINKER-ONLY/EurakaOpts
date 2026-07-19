# MLA Decode v017: Adaptive Reduce-Max Clear

## Objective

Reduce score-maximum initialization overhead for short split-K chunks without
regressing the throughput-sensitive long-chunk path.

## Change

- when `seqlen_kv / num_split <= 512`, remove the standalone per-tile
  `scores_max = -inf` fill and let `T.reduce_max(..., clear=True)` initialize
  its destination;
- retain v016's explicit fill plus `clear=False` reduction for longer chunks,
  where the alternative reduction schedule regressed B32/64K;
- apply the adaptive policy to the materialized QK kernel, direct split-K
  kernel, and no-split kernel;
- retain v016's cached split LSE, QK clear-accum scheduling, fused 576-D QK,
  prefix-scaled weights, split policy, and final reduction.

The initial pre-loop `scores_max = -inf` remains in place because it represents
the previous online-softmax maximum for the first tile. Only the redundant
initialization of the current tile's reduction output is folded into the
reduction.

## Correctness

- All 31 official outputs are bitwise identical to v016.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00038418` and `0.00028522`.
- Only compiled kernels are cached; every invocation reads current inputs,
  recomputes attention, and writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, ten alternating
measured calls per sample, and five samples over all 31 official shapes:

| Shape group | Cases | v016 total | v017 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Split chunk <= 512 | 14 | 2.10061 ms | 2.04388 ms | **+2.70%** |
| 31-case total | 31 | 25.60980 ms | 25.46895 ms | **+0.53%** |

All five aggregate samples improved: `+0.3510%, +0.5813%, +0.6102%,
+0.4637%, +0.5268%`. Every per-shape median improved. Selected short-split
shapes improved by `+1.62%` to `+5.61%`; long-split shapes retain v016's
original reduction path.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`e232decc37a70285dfa8027f32471a4580e1d4f03ce9c48d17e424d645ab124a`.
