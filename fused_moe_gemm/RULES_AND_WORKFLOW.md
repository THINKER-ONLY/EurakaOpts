# Fused MoE GEMM: Rules and Autonomous Workflow

## Scope and success criteria

Target: XPUOJ Contest 5, Problem 1, `Tilelang ???? - Fused MoE GEMM`.

The required operation is pre-routed DeepSeek-style expert MLP:

```text
gate = x @ gate_w[e]^T
up = x @ up_w[e]^T
hidden = silu(gate) * up
out = hidden @ down_w[e]^T
out *= routed_expert_weights[raw token index]
```

Basic tier: function-test pass rate above 99%, with an improvement over the reference sample performance.

Performance tier: after satisfying the basic tier, ranking uses end-to-end test API latency. The contest performance procedure is warmup 10 times followed by 100 repetitions averaged; smaller latency ranks higher. Preliminary performance ranking contributes to 10 final-round points.

## OJ interface and tensor semantics

Submission language: `tilelang.maca-c500`.

Only this entry point is called:

```python
def run_kernel(
    stacked_expert_tokens, gate_w, up_w, down_w,
    routed_expert_weights, group_sizes, group_offsets,
    group_padded_offsets, group_idx_for_bx, out,
):
    ...
```

- Write only to the preallocated `out`; it is the sole INOUT argument. Do not return a tensor.
- `out.shape == (group_sum, d_hidden)` and `out.dtype == float16` are fixed by the evaluator.
- Inputs, weights, workspace, and output use float16; group metadata uses int32.
- `stacked_expert_tokens` and `out` use padded token positions.
- `routed_expert_weights` uses compact, real-token positions.
- For expert `e`: `raw_start = group_offsets[e]`, `padded_start = group_padded_offsets[e]`, and `valid_count = group_sizes[e]`.
- Padding rows must be zero and must not contribute to computation.
- `group_idx_for_bx` maps 128-token TileLang blocks to experts.
- Router, top-k, and scatter-reduce are out of scope.

## Hard constraints

- GPU compute operators must be implemented with TileLang only. Do not use PyTorch GPU tensor allocation or computation inside the submitted implementation: no `torch.empty`, `torch.zeros`, matmul, functional operators, indexing kernels, or synchronization as a computational workaround.
- The repository reference is commit `ee6db4376484f2f7270183c01fd0d90f794965cb`, `race_tests/moe/custom_fusedmoe.py`.
- SafeExecutor has rejected class definitions, `from ... import ...`, and qualified type annotations in earlier attempts. Submitted code must be top-level-function based, use direct imports only where needed, and keep `run_kernel` annotations absent unless independently proven accepted.
- Never modify the upstream baseline under `race_tests/moe/`; all experiments live in this workspace.
- Do not submit until the candidate completes a server-local correctness run.

## Test dimensions and OJ timing

| Case | Experts | d_hidden | d_expert | group_sum | OJ warmup | OJ iters |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 2048 | 8192 | 2272 | 5 | 30 |
| 2 | 32 | 7168 | 2048 | 4544 | 5 | 20 |
| 3 | 64 | 7168 | 2048 | 9088 | 5 | 20 |

## Version workflow

1. Create one version as `vNNN_<english_method>.py` in this folder.
2. Record hypothesis, exact differences, local command, source SHA-256, correctness result, and timing in `OPTIMIZATION_LOG.md` before OJ submission.
3. Run on the MetaX C500 server using the documented TileLang/MACA environment. First verify numerical behavior and padding behavior; then run the relevant local benchmark.
4. Submit exactly once after local success, login again only when required.
5. Query the final OJ verdict and detailed error when failed.
6. Synchronize the exact submitted source and record the submission ID, verdict, score, resource data, and diagnosis in both local and server logs.
7. Analyze a failed or regressive result before a new version. Never issue duplicate parallel submissions.
8. Commit local progress only after a meaningful checkpoint; do not push unless explicitly requested.

## Recovery

For transient transport, API overload, or tool interruption, reconnect/retry the interrupted read-only or local operation with bounded retries. Before any retry of submission, query recent submissions by source hash and time to prevent duplication.
