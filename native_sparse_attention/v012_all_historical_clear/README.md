# NSA v012: All Historical Block Clear

## Objective

Extend v011's full-historical-block causal fast path to D64 after validating
the complete D64/S1 official family instead of relying on one long-sequence
probe.

## Change

- for every dimension, test `block_start + block_size <= token + 1` at runtime;
- clear the score fragment directly when the full block is historical;
- retain the original causal-mask loop for a current or partially visible
  block;
- retain v011's GEMM policies, thread policy, softmax paths, block loop, and
  shared-memory reuse.

The fast path is semantic rather than input-specific: arbitrary recent,
future, invalid, and partial blocks retain their existing handling.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All 36 official D64/S1 outputs are bitwise identical to v011.
- The official regression includes each sequence's initial partial block as
  well as deterministic historical blocks.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples:

| Official family | Cases | v011 total | v012 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D64 / S1 / BS16 | 29 | 1.6713 ms | 1.6396 ms | **+1.81%** |
| D64 / S1 / BS32 | 3 | 0.1094 ms | 0.1068 ms | **+2.27%** |
| D64 / S1 / BS64 | 4 | 0.1608 ms | 0.1560 ms | **+3.13%** |
| Combined | 36 | 1.9415 ms | 1.9024 ms | **+1.969%** |

All five combined samples improved: `+2.2040%, +2.0085%, +1.9692%,
+1.6997%, +1.7785%`. Thirty-five of 36 per-shape medians improved; the lone
`-0.52%` B1/L4096 point is outweighed consistently in every family aggregate
sample.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`7d7634e657d84097cc841d25b9d16a2eb8c90a11465181d653f79750101e24ff`.
