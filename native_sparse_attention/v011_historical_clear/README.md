# NSA v011: Historical Block Clear

## Objective

Avoid constructing a per-element causal mask when an entire selected block is
already earlier than the query token.

## Change

- for D32 and D128, test `block_start + block_size <= token + 1` at runtime;
- clear the score fragment directly when the full block is historical;
- retain the original causal-mask loop for a current or partially visible
  block;
- retain v010's GEMM policies, thread policy, softmax paths, block loop, and
  shared-memory reuse.

D64 remains on the original mask path because its long-sequence probe did not
benefit. The fast path is semantic rather than input-specific: arbitrary
recent, future, invalid, and partial blocks retain their existing handling.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All 64 official D32/D128 S1 outputs are bitwise identical to v010.
- The official regression includes each sequence's initial partial block as
  well as deterministic historical blocks.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted repeat used five warmups, twenty
alternating measured calls per sample, and five samples:

| Official family | Cases | v010 total | v011 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D32 / S1 | 33 | 1.3430 ms | 1.3022 ms | **+2.97%** |
| D128 / S1 | 31 | 1.9816 ms | 1.9439 ms | **+1.78%** |
| Combined | 64 | 3.3246 ms | 3.2461 ms | **+2.263%** |

All five combined samples improved: `+2.2142%, +2.2330%, +2.3734%,
+2.2628%, +2.2719%`. Every per-shape median improved; the smallest measured
gain was `+0.22%`.

An initial full run measured `+0.754%` median but included one negative
aggregate sample amid frequency outliers. Repeating after all kernels were
compiled produced the stable result above.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`3452589778bf37159e6ad57fd4a5109fe04d46d6a3a3c4540eaf0115d5e1d700`.
