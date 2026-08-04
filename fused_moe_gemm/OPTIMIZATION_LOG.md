# Fused MoE GEMM Optimization Log

## Workspace initialization

- Local workspace: `D:\\C_data\\EurakaOpts\\fused_moe_gemm`
- Server workspace: `/data/tilelang-metax/fused_moe_gemm`
- Git branch: `work/moe-baseline-bringup`
- Upstream reference: `race` branch, `race_tests/moe/custom_fusedmoe.py`
- Original upstream SHA-256: `11102a45774b9e60263c6defebd4ac777c64ba26181ee448a9852e0252872901`

## Previous non-baseline OJ attempts (not valid performance data)

- `102242`: SafeExecutor rejected a class definition.
- `102253` and `102254`: SafeExecutor rejected restricted imports/type annotations.
- `102260` and `102261`: duplicated by an orchestration mistake; also invalid because the implementation used `torch.empty` as GPU workspace allocation. Both are excluded from baseline comparisons.

## Active version

Pending creation after the original server baseline is copied and locally reproduced under this workflow.

## v008_tilelang_only_baseline

- Intent: first direct OJ-style, class-free, TileLang-only fused kernel. Compact tensor indexing uses `group_offsets`; `group_padded_offsets` maps virtual 128-row blocks; only valid rows write `out`.
- Static gate: passed (`torch` absent, no class definition, no `from` import, unannotated `run_kernel`).
- C500 result: failed at TileLang lowering before launch. `T.Pipelined` saw repeated overlapping writes to one `weight_shared` tile while loading gate, up, and down weights: `Pipeline planning error: Multiple writes to overlapping buffer regions`.
- Decision: not submitted. This is a compiler-legalization issue, not a numerical result.

## v009_serial_k_baseline

- Change from v008: replace only the K loop `T.Pipelined(...)` with `T.serial(...)`; preserve TileLang-only fused computation, dimensions, compact indexing, and output semantics.
- Source SHA-256: `9f048a8f5fef01e65833c98953f28a0fc72ff915c65f62c0cc5573295910a38f`.
- Static gate: passed (`torch` absent, no class definition, no `from` import, top-level unannotated `run_kernel`).
- C500 local smoke validation: passed.
  - `[128]`: max absolute error `1.220703125e-04`.
  - compact uneven groups `[100, 150]`, virtual blocks `[0,1,1]`: max absolute error `2.44140625e-04`.
- Reviewer gate: two external reviewer calls were attempted; both failed at transport/response parsing, so no reviewer verdict is recorded yet. Retry is required before OJ submission.
- Submission: none. Next: large realistic local correctness test, then reviewer retry.

## v010_guarded_input_baseline

- Local race benchmark functional suite passed, but this was later recognized as a compact-layout harness and is not sufficient proof for the OJ padded-layout contract.
- OJ submission `102568`: RuntimeError, zero score. Detail only: `worker is unhealthy after 2 seconds (retrying)`; no compiler/runtime stack was provided.
- Diagnostic hypothesis: 128x128 float16 `weight_shared` (32 KiB) plus 128x128 float16 `hidden_shared` (32 KiB) consumes 64 KiB shared memory before compiler/internal allocations, likely exceeding or exhausting the C500 worker configuration.

## v011_split_m64_baseline

- Change: split each logical 128-token block into two 64-token physical blocks; expected shared use is 48 KiB.
- Local compact-layout functional suite passed but was not the authoritative OJ padded-layout test.
- OJ submission `102588`: RuntimeError, zero score. Same opaque `worker is unhealthy after 2 seconds (retrying)` detail.
- Retrospective: v011 still used compact input/output indexes; this conflicts with the OJ reference code, which explicitly uses `stacked_expert_tokens[padded_start:...]` and `out[padded_start:...]`. No further compact-layout candidate will be submitted.

## v012_padded_m64_baseline

- Intent: first baseline that follows the OJ reference layout literally. `group_padded_offsets` addresses padded input/output. `group_offsets` addresses compact routing weights. Each logical 128-row group block is split into two 64-row physical blocks, retaining TileLang-only computation and reducing explicit shared-memory allocation to 48 KiB.
- Static gate: passed (`torch` absent, no class definition, no `from` import, direct top-level unannotated `run_kernel`).
- C500 padded-layout test: passed using two experts, sizes `[100, 100]`, raw offsets `[0, 100]`, padded offsets `[0, 128]`, physical input/output rows `256`. Max absolute error `1.220703125e-04`; both padding ranges were exactly zero.
- Reviewer: repeated external reviewer requests returned response parse failures; no reviewer verdict is available. This transport failure is logged and does not alter the code evidence.
- Next: submit only after the above static and C500 padded-layout gates; query detail; synchronize source/log.

## v010_guarded_input_baseline

- Local race benchmark functional suite passed, but this was later recognized as a compact-layout harness and is not sufficient proof for the OJ padded-layout contract.
- OJ submission `102568`: RuntimeError, zero score. Detail only: `worker is unhealthy after 2 seconds (retrying)`; no compiler/runtime stack was provided.
- Diagnostic hypothesis: 128x128 float16 `weight_shared` (32 KiB) plus 128x128 float16 `hidden_shared` (32 KiB) consumes 64 KiB shared memory before compiler/internal allocations, likely exceeding or exhausting the C500 worker configuration.

## v011_split_m64_baseline

- Change: split each logical 128-token block into two 64-token physical blocks; expected shared use is 48 KiB.
- Local compact-layout functional suite passed but was not the authoritative OJ padded-layout test.
- OJ submission `102588`: RuntimeError, zero score. Same opaque `worker is unhealthy after 2 seconds (retrying)` detail.
- Retrospective: v011 still used compact input/output indexes; this conflicts with the OJ reference code, which explicitly uses `stacked_expert_tokens[padded_start:...]` and `out[padded_start:...]`. No further compact-layout candidate will be submitted.

## v012_padded_m64_baseline

- Intent: first baseline that follows the OJ reference layout literally. `group_padded_offsets` addresses padded input/output. `group_offsets` addresses compact routing weights. Each logical 128-row group block is split into two 64-row physical blocks, retaining TileLang-only computation and reducing explicit shared-memory allocation to 48 KiB.
- Static gate: passed (`torch` absent, no class definition, no `from` import, direct top-level unannotated `run_kernel`).
- C500 padded-layout test: passed using two experts, sizes `[100, 100]`, raw offsets `[0, 100]`, padded offsets `[0, 128]`, physical input/output rows `256`. Max absolute error `1.220703125e-04`; both padding ranges were exactly zero.
- Reviewer: repeated external reviewer requests returned response parse failures; no reviewer verdict is available. This transport failure is logged and does not alter the code evidence.
- Next: submit only after the above static and C500 padded-layout gates; query detail; synchronize source/log.

## v013_padded_m64_n64

- Intent: reduce explicit shared memory by changing the expert tile from 128 to 64.
- C500 result: not executable. The prior single `weight_shared` had shape `(64, 128)`, then was reused for a down matrix whose required K dimension is 64. TileLang rejected this correctly: `T.gemm K shape check failed: K_A = 64, K_B = 128`.
- Decision: not submitted. The failure exposed a source-level layout error rather than an OJ issue.

## v014_separate_weight_tiles

- Change from v013: retain M=64/N=64 padded-layout baseline, but allocate `projection_weight_shared: (64,128)` for gate/up and `down_weight_shared: (128,64)` for down projection. This makes both `T.gemm` K dimensions explicit and valid. Explicit shared memory: 16 KiB + 16 KiB + 8 KiB = 40 KiB, below the previous 64 KiB design.
- Source SHA-256: `3f6872344a4c84d4ed1c70448e901fdb73324cca8bc69fbe7d04b55de356b196`.
- Static gate: passed (`torch` absent, no class definition, no `from` import, direct unannotated `run_kernel`).
- C500 padded-layout test: passed; 2 experts of 100 rows, raw offsets `[0,100]`, padded offsets `[0,128]`, physical rows=256. Max absolute error `1.220703125e-04`; padding output exactly zero.
- Submission readiness: first candidate with OJ-reference padded input/output indexing, compact raw routing-weight index, valid 64/128 GEMM layout, and <64 KiB explicit shared memory. Submit exactly once next.

## v013_padded_m64_n64

- Intent: reduce explicit shared memory by changing the expert tile from 128 to 64.
- C500 result: not executable. The prior single `weight_shared` had shape `(64, 128)`, then was reused for a down matrix whose required K dimension is 64. TileLang rejected this correctly: `T.gemm K shape check failed: K_A = 64, K_B = 128`.
- Decision: not submitted. The failure exposed a source-level layout error rather than an OJ issue.

## v014_separate_weight_tiles

- Change from v013: retain M=64/N=64 padded-layout baseline, but allocate `projection_weight_shared: (64,128)` for gate/up and `down_weight_shared: (128,64)` for down projection. This makes both `T.gemm` K dimensions explicit and valid. Explicit shared memory: 16 KiB + 16 KiB + 8 KiB = 40 KiB, below the previous 64 KiB design.
- Source SHA-256: `3f6872344a4c84d4ed1c70448e901fdb73324cca8bc69fbe7d04b55de356b196`.
- Static gate: passed (`torch` absent, no class definition, no `from` import, direct unannotated `run_kernel`).
- C500 padded-layout test: passed; 2 experts of 100 rows, raw offsets `[0,100]`, padded offsets `[0,128]`, physical rows=256. Max absolute error `1.220703125e-04`; padding output exactly zero.
- Submission readiness: first candidate with OJ-reference padded input/output indexing, compact raw routing-weight index, valid 64/128 GEMM layout, and <64 KiB explicit shared memory. Submit exactly once next.
