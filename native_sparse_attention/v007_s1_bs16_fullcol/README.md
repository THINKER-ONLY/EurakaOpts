# NSA v007: S1 BS16 FullCol

## Objective

Extend v006's successful GEMM layout policy to the three largest remaining
official NSA shape families: D32, D64, and D128 with one selected 16-token
block.

## Change

- select `T.GemmWarpPolicy.FullCol` at compile time for all S1/BS16 shapes;
- retain v006's `FullCol` policy for D128/S1/BS32;
- retain `FullRow` for every other shape;
- keep the thread policy, single-block softmax specialization, shared-memory
  reuse, and gathered S2 kernel unchanged.

The policy applies to both QK and PV GEMMs. Every call recomputes from its
current tensors and block indices; only compiled kernels are cached.

## Correctness

- The standard two-input FP32 checks retain maximum absolute errors
  `0.00150776` and `0.00143313`.
- All 85 official S1/BS16 outputs are bitwise identical to v006.
- The official cases use deterministic historical block indices to exercise
  non-recent K/V accesses.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples over all 85 official
S1/BS16 shapes from the race branch's 109-case set:

| Official family | Cases | v006 total | v007 total | Median paired improvement |
| --- | ---: | ---: | ---: | ---: |
| D32 / S1 / BS16 | 33 | 1.3428 ms | 1.3008 ms | **+3.10%** |
| D64 / S1 / BS16 | 29 | 1.6023 ms | 1.5639 ms | **+2.37%** |
| D128 / S1 / BS16 | 23 | 1.4257 ms | 1.3935 ms | **+2.22%** |
| All S1 / BS16 | 85 | 4.3708 ms | 4.2583 ms | **+2.647%** |

All five aggregate samples improved: `+2.5545%, +2.6618%, +2.4286%,
+2.6465%, +3.5065%`. Every per-shape median improved; the smallest measured
gain was `+0.38%`.

The local benchmark now accepts the archived official JSON through
`--case-json native_sparse_attention/test_cases_official.json` and can select
this family with `--official-family s1_bs16`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`c28645e84f773b4831d3253712229bf40072224bd3085aad29ddabb70ffe8542`.
