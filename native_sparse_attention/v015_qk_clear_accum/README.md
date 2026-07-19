# NSA v015: QK Clear-Accum Scheduling

## Objective

Remove an unnecessary synchronization boundary between score-fragment
initialization and the QK GEMM on the dominant main-kernel path.

## Change

- for complete historical blocks, initialize the QK accumulator through the
  GEMM's `clear_accum=True` path;
- keep the causal-mask initialization and accumulating GEMM for partial
  blocks;
- keep D128/S1/BS32 on v014's original single-GEMM control-flow layout because
  duplicating that larger GEMM body regressed its B4/L1024 shape;
- retain v014's prenormalized S1/S2 weights, gathered S2 kernel, thread and
  GEMM policies, shared-memory reuse, and S4/S8 online softmax path.

On MACA, placing the historical-block GEMM inside the runtime branch schedules
the fragment clear after the shared-memory synchronization for the K copy.
The D128/BS32 compile-time exclusion was verified by comparing generated
kernel sources: its v014 and v015 SHA-256 values are identical.

## Correctness

- All 109 official outputs are bitwise identical to v014.
- The standard two-input FP32-reference checks retain maximum absolute errors
  `0.00085354` and `0.00094223`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.3, TileLang
`0.1.11+cuda.git56b76a2b`. The accepted run used five warmups, twenty
alternating measured calls per sample, and five samples over all 109 official
cases:

| Official family | Cases | v014 total | v015 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| S1 | 100 | 5.08361 ms | 4.98943 ms | **+1.85%** |
| S2 | 3 | 0.13924 ms | 0.13533 ms | **+2.80%** |
| S4 | 3 | 0.18049 ms | 0.17864 ms | **+1.03%** |
| S8 | 3 | 0.23389 ms | 0.23218 ms | **+0.73%** |
| D32 | 33 | 1.28696 ms | 1.25476 ms | **+2.50%** |
| D64 | 45 | 2.37651 ms | 2.33539 ms | **+1.73%** |
| D128 | 31 | 1.97376 ms | 1.94543 ms | **+1.44%** |
| 109-case total | 109 | 5.63724 ms | 5.53558 ms | **+1.78%** |

All five aggregate samples improved: `+1.6485%, +1.8860%, +1.7775%,
+1.8277%, +1.6861%`. The only negative per-shape median was `-0.41%` on
B4/L1024/D128/S1/BS32, whose generated candidate kernel is byte-for-byte the
same source as v014 and therefore represents paired measurement noise rather
than a changed path.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`4f181ffbb9480adb71e0060a72cb7dc339682e567f775007d3b699b7ce1b3b21`.
