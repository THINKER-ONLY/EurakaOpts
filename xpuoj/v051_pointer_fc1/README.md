# XPUOJ v051: Pointer FC1

## Objective

Remove the E32/E64 full-input pack from v050 while preserving complete FC1,
SwiGLU, down-projection, routing, and output recomputation on every invocation.

## Change

E32 and E64 no longer copy every expert into a dense `M176` input before the
combined FC1. A small TileLang kernel now builds three device-resident pointer
arrays from the current tensors and `group_padded_offsets`:

- one input address per expert in the stacked input;
- one combined gate/up weight address per expert;
- one preallocated gate/up output address per expert.

The two existing worker streams call `mcblasHgemmBatched` through its C ABI,
using those device pointer arrays. Each invocation rebuilds the pointer table
inside the CUDA graph and executes the same two half-batch FC1 launches as
v050. E16 retains v050's dense path.

An `M176` FC1 can read beyond an expert's 128-row padded segment. For every
non-final expert this remains inside the allocated stacked tensor and affects
only invalid rows that are never unpacked. The final expert has no following
segment, so it is copied on every invocation into its own safe 176-row buffer
and its input pointer targets that buffer. This avoids allocation-boundary
overreads without restoring the full pack.

The implementation does not inspect input values or identities, cache a
completed activation/output, or omit any valid-row computation. The combined
weight preprocessing inherited from v050 remains a warmup-time model-weight
layout conversion. The runtime activation, metadata, route weights, and output
are all consumed on every graph replay.

## Correctness

The repository's standard three-case comparison is bit-identical to v050 on
all valid and padded rows. Both one-expert FP32 oracle checks also retain
v050's errors:

| Shape | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| `(2048, 8192)` | 0.000570536 | 0.000570536 |
| `(7168, 2048)` | 0.000950813 | 0.000950813 |

Two additional graph-sensitivity checks passed:

- After capture, expert counts and all three offset/map tensors were changed
  in place while preserving total valid and padded rows. The smallest expert
  was moved to the final position (`119` rows), exercising the safe final
  buffer. E32 and E64 matched the FP32 oracle with zero tolerance violations.
- Two graph keys with different activation, route, metadata, and output
  pointers but a shared workspace were captured and replayed in alternating
  order. Both outputs matched independent FP32 oracles for E32 and E64, with
  padded rows remaining exactly zero.

The metadata-change error statistics were identical to v050: E32 maximum
`0.125`, mean `0.00210159`; E64 maximum `0.25`, mean `0.00243222`. The larger
absolute values come from the deliberately higher-amplitude random input and
remain below 22% of the allowed `atol=1e-2, rtol=1e-2` error envelope.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`. Each entry below is a fresh process that loaded only one
version. E16 used 100 calls per timing slot, E32 used 50, and E64 used 30;
every process recorded five timing samples after ten warmups.

| Case | v050 process medians | v051 process medians | Median improvement |
| --- | --- | --- | ---: |
| E16 | 1.90886, 1.90998, 1.90803 ms | 1.91026, 1.90804, 1.91050 ms | -0.07% (neutral) |
| E32 | 3.11492, 3.11165, 3.11143 ms | 3.02921, 3.02983, 3.02800 ms | **+2.65%** |
| E64 | 6.28844, 6.27791, 6.27744 ms | 6.09855, 6.09329, 6.08498 ms | **+2.94%** |

Summing the three process medians reduces the local proxy total from
`11.29843 ms` to `11.03276 ms`, an aggregate improvement of **+2.35%**.
Standalone peak allocation changes from 4.901 to 4.923 GiB for E32 and from
9.672 to 9.715 GiB for E64, remaining well below the device quota.

An E64 profiler run over three graph replays retained six FC1 and twelve down
GEMM launches, proving that the GEMM work was not skipped. Aggregate TileLang
kernel time fell from `2.723 ms` to `1.671 ms` as the two full half-input packs
were replaced by one final-expert pack and one pointer-table build per replay.

Decision: **accepted as the local full-compute baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`842f45d4de4e835c92b72452d2e0ec8e92c35f2bb9d17ca0d143dc90e1c9085c`.

## Submission

This version is retained for local C500 analysis and has not been submitted to
XPUOJ.
