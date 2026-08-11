import gc
import importlib.util
import math
import traceback

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v025_unconditional_gemm_sync.py"
BLOCK_TOKEN = 128
CONFIGS = [
    {
        "name": "functional_1",
        "d_hidden": 7168,
        "d_expert": 2048,
        "n_experts": 8,
        "raw_rows": 32768,
        "seed": 81394,
    },
    {
        "name": "functional_2",
        "d_hidden": 3584,
        "d_expert": 1024,
        "n_experts": 4,
        "raw_rows": 16384,
        "seed": 81394,
    },
]


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_metadata(raw_rows, n_experts, device):
    base = raw_rows // n_experts
    counts = [base for _ in range(n_experts)]
    for index in range(raw_rows - base * n_experts):
        counts[index] += 1

    group_offsets = []
    group_padded_offsets = []
    raw_offset = 0
    padded_offset = 0
    for count in counts:
        group_offsets.append(raw_offset)
        group_padded_offsets.append(padded_offset)
        raw_offset += count
        padded_offset += math.ceil((count + 1) / BLOCK_TOKEN) * BLOCK_TOKEN

    num_blocks = padded_offset // BLOCK_TOKEN
    group_idx_for_bx = []
    expert_id = 0
    for bx in range(num_blocks):
        padded_start = bx * BLOCK_TOKEN
        while (
            expert_id + 1 < n_experts
            and padded_start >= group_padded_offsets[expert_id + 1]
        ):
            expert_id += 1
        group_idx_for_bx.append(expert_id)

    return (
        counts,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(group_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_padded_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_idx_for_bx, dtype=torch.int32, device=device),
        padded_offset,
    )


def checksum(tensor):
    values = tensor.float()
    return float(values.sum().item()), float(values.square().sum().item())


def run_case(module, config):
    name = config["name"]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    n_experts = config["n_experts"]
    raw_rows = config["raw_rows"]
    device = "cuda"

    torch.manual_seed(config["seed"])
    (
        counts,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
        padded_rows,
    ) = build_metadata(raw_rows, n_experts, device)

    stacked_expert_tokens = torch.randn(
        (padded_rows, d_hidden), dtype=torch.float16, device=device
    )
    gate_w = torch.randn(
        (n_experts, d_expert, d_hidden), dtype=torch.float16, device=device
    ) / math.sqrt(d_expert)
    up_w = torch.randn_like(gate_w) / math.sqrt(d_expert)
    down_w = torch.randn(
        (n_experts, d_hidden, d_expert), dtype=torch.float16, device=device
    ) / math.sqrt(d_hidden)
    routed_expert_weights = torch.rand(
        (raw_rows,), dtype=torch.float16, device=device
    )
    out = torch.full(
        (padded_rows, d_hidden), 7, dtype=torch.float16, device=device
    )

    float_inputs = [
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
    ]
    metadata_inputs = [
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
    ]
    before_checksums = [checksum(tensor) for tensor in float_inputs]
    metadata_copies = [tensor.clone() for tensor in metadata_inputs]

    print(
        "CASE_BEGIN",
        name,
        d_hidden,
        d_expert,
        n_experts,
        raw_rows,
        padded_rows,
        flush=True,
    )
    module.run_kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
        out,
    )
    torch.cuda.synchronize()
    print("KERNEL_EXECUTION_PASS", name, flush=True)

    max_abs_error = 0.0
    sum_abs_error = 0.0
    compared_values = 0
    max_reference = 0.0
    padding_max = 0.0

    for expert_id, count in enumerate(counts):
        padded_start = group_padded_offsets[expert_id]
        raw_start = group_offsets[expert_id]
        x = stacked_expert_tokens[padded_start : padded_start + count].float()
        gate = x @ gate_w[expert_id].float().t()
        up = x @ up_w[expert_id].float().t()
        activated = (torch.nn.functional.silu(gate) * up).half()
        reference = activated.float() @ down_w[expert_id].float().t()
        reference *= routed_expert_weights[
            raw_start : raw_start + count
        ].float().view(-1, 1)
        reference = reference.half().float()
        actual = out[padded_start : padded_start + count].float()
        error = (actual - reference).abs()

        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
        max_abs_error = max(max_abs_error, float(error.max().item()))
        sum_abs_error += float(error.sum().item())
        compared_values += error.numel()
        max_reference = max(max_reference, float(reference.abs().max().item()))

        padded_end = group_padded_offsets[expert_id + 1] if expert_id + 1 < n_experts else padded_rows
        if padded_start + count < padded_end:
            padding_max = max(
                padding_max,
                float(out[padded_start + count : padded_end].abs().max().item()),
            )

        del x, gate, up, activated, reference, actual, error
        torch.cuda.empty_cache()

    after_checksums = [checksum(tensor) for tensor in float_inputs]
    for before, after in zip(before_checksums, after_checksums):
        assert before == after
    for tensor, copy in zip(metadata_inputs, metadata_copies):
        assert torch.equal(tensor, copy)
    assert padding_max == 0.0

    mean_abs_error = sum_abs_error / compared_values
    print("MAX_ABS_ERROR", name, max_abs_error, flush=True)
    print("MEAN_ABS_ERROR", name, mean_abs_error, flush=True)
    print("MAX_REFERENCE", name, max_reference, flush=True)
    print("PADDING_MAX", name, padding_max, flush=True)
    print("INPUT_CHECKSUMS_PASS", name, flush=True)
    print("METADATA_IMMUTABILITY_PASS", name, flush=True)
    print("FUNCTIONAL_CASE_PASS", name, flush=True)

    del (
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        out,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
    )
    gc.collect()
    torch.cuda.empty_cache()


def main():
    module = load_candidate()
    for config in CONFIGS:
        run_case(module, config)
    print("ALL_FULL_CORRECTNESS_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
