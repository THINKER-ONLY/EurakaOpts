import importlib.util
import math
import traceback

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v034_internal_global_workspace.py"


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    counts,
    group_offsets,
    group_padded_offsets,
):
    result = torch.zeros_like(stacked_expert_tokens)
    for expert_id, count in enumerate(counts):
        if count == 0:
            continue
        padded_start = group_padded_offsets[expert_id]
        raw_start = group_offsets[expert_id]
        x = stacked_expert_tokens[padded_start : padded_start + count].float()
        gate = x @ gate_w[expert_id].float().t()
        up = x @ up_w[expert_id].float().t()
        activated = (torch.nn.functional.silu(gate) * up).half()
        expert_out = activated.float() @ down_w[expert_id].float().t()
        expert_out *= routed_expert_weights[raw_start : raw_start + count].float().view(-1, 1)
        result[padded_start : padded_start + count] = expert_out.half()
    return result


def main():
    module = load_candidate()
    torch.manual_seed(81394)

    experts = 2
    d_hidden = 128
    d_expert = 128
    block_token = 128
    counts = [129, 17]

    group_offsets = []
    group_padded_offsets = []
    raw_offset = 0
    padded_offset = 0
    for count in counts:
        group_offsets.append(raw_offset)
        group_padded_offsets.append(padded_offset)
        raw_offset += count
        padded_offset += math.ceil((count + 1) / block_token) * block_token

    raw_group_sum = raw_offset
    padded_group_sum = padded_offset
    num_blocks_m = padded_group_sum // block_token
    group_idx_for_bx = []
    for bx in range(num_blocks_m):
        padded_start = bx * block_token
        expert_id = 0
        for index, offset in enumerate(group_padded_offsets):
            if padded_start >= offset:
                expert_id = index
        group_idx_for_bx.append(expert_id)

    device = "cuda"
    stacked_expert_tokens = torch.randn(
        (padded_group_sum, d_hidden), dtype=torch.float16, device=device
    ) * 0.25
    gate_w = torch.randn(
        (experts, d_expert, d_hidden), dtype=torch.float16, device=device
    ) * 0.1
    up_w = torch.randn_like(gate_w) * 0.1
    down_w = torch.randn(
        (experts, d_hidden, d_expert), dtype=torch.float16, device=device
    ) * 0.1
    routed_expert_weights = (
        torch.rand((raw_group_sum,), dtype=torch.float16, device=device) * 0.75 + 0.25
    )
    group_sizes = torch.tensor(counts, dtype=torch.int32, device=device)
    group_offsets_tensor = torch.tensor(group_offsets, dtype=torch.int32, device=device)
    group_padded_offsets_tensor = torch.tensor(
        group_padded_offsets, dtype=torch.int32, device=device
    )
    group_idx_tensor = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=device)

    readonly_inputs = [
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
    ]
    readonly_copies = [tensor.clone() for tensor in readonly_inputs]

    expected = reference(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        counts,
        group_offsets,
        group_padded_offsets,
    )

    outputs = []
    for call_index in range(2):
        out = torch.full(
            (padded_group_sum, d_hidden),
            7,
            dtype=torch.float16,
            device=device,
        )
        module.run_kernel(*readonly_inputs, out)
        torch.cuda.synchronize()
        outputs.append(out.clone())
        print(f"CALL_{call_index + 1}_OK", flush=True)

    actual = outputs[0]
    valid_mask = torch.zeros((padded_group_sum,), dtype=torch.bool, device=device)
    for expert_id, count in enumerate(counts):
        start = group_padded_offsets[expert_id]
        valid_mask[start : start + count] = True

    valid_actual = actual[valid_mask].float()
    valid_expected = expected[valid_mask].float()
    abs_error = (valid_actual - valid_expected).abs()
    max_abs = float(abs_error.max().item())
    mean_abs = float(abs_error.mean().item())
    max_ref = float(valid_expected.abs().max().item())
    padding_max = float(actual[~valid_mask].abs().max().item())
    repeated_max = float((outputs[0].float() - outputs[1].float()).abs().max().item())

    print("TEST_SHAPES", experts, d_hidden, d_expert, raw_group_sum, padded_group_sum, flush=True)
    print("GROUP_OFFSETS", group_offsets, flush=True)
    print("GROUP_PADDED_OFFSETS", group_padded_offsets, flush=True)
    print("GROUP_IDX_FOR_BX", group_idx_for_bx, flush=True)
    print("MAX_ABS_ERROR", max_abs, flush=True)
    print("MEAN_ABS_ERROR", mean_abs, flush=True)
    print("MAX_REFERENCE", max_ref, flush=True)
    print("PADDING_MAX", padding_max, flush=True)
    print("REPEATED_MAX", repeated_max, flush=True)

    torch.testing.assert_close(valid_actual, valid_expected, atol=1e-2, rtol=1e-2)
    assert padding_max == 0.0
    assert repeated_max == 0.0
    for original, copy in zip(readonly_inputs, readonly_copies):
        assert torch.equal(original, copy)

    print("NONZERO_CORRECTNESS_PASS", flush=True)
    print("PADDING_ZERO_PASS", flush=True)
    print("READONLY_INPUTS_PASS", flush=True)
    print("REPEATED_CALL_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
