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


def main():
    module = load_candidate()
    experts = 16
    d_hidden = 2048
    d_expert = 8192
    raw_group_sum = 2272
    block_token = 128
    counts = [raw_group_sum // experts] * experts
    for index in range(raw_group_sum % experts):
        counts[index] += 1

    group_offsets = []
    group_padded_offsets = []
    raw_offset = 0
    padded_offset = 0
    for count in counts:
        group_offsets.append(raw_offset)
        group_padded_offsets.append(padded_offset)
        raw_offset += count
        padded_offset += math.ceil((count + 1) / block_token) * block_token

    num_blocks_m = math.ceil(raw_group_sum / block_token) + experts
    padded_group_sum = num_blocks_m * block_token
    group_idx_for_bx = []
    for bx in range(num_blocks_m):
        padded_start = bx * block_token
        expert = 0
        for index, offset in enumerate(group_padded_offsets):
            if padded_start >= offset:
                expert = index
        group_idx_for_bx.append(expert)

    device = "cuda"
    stacked_expert_tokens = torch.zeros((padded_group_sum, d_hidden), dtype=torch.float16, device=device)
    gate_w = torch.zeros((experts, d_expert, d_hidden), dtype=torch.float16, device=device)
    up_w = torch.zeros_like(gate_w)
    down_w = torch.zeros((experts, d_hidden, d_expert), dtype=torch.float16, device=device)
    routed_expert_weights = torch.ones((raw_group_sum,), dtype=torch.float16, device=device)
    group_sizes = torch.tensor(counts, dtype=torch.int32, device=device)
    group_offsets_tensor = torch.tensor(group_offsets, dtype=torch.int32, device=device)
    group_padded_offsets_tensor = torch.tensor(group_padded_offsets, dtype=torch.int32, device=device)
    group_idx_tensor = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=device)
    out = torch.full((padded_group_sum, d_hidden), 7, dtype=torch.float16, device=device)

    print(
        "SMOKE_INPUTS",
        experts,
        d_hidden,
        d_expert,
        raw_group_sum,
        padded_group_sum,
        num_blocks_m,
        flush=True,
    )
    try:
        module.run_kernel(
            stacked_expert_tokens,
            gate_w,
            up_w,
            down_w,
            routed_expert_weights,
            group_sizes,
            group_offsets_tensor,
            group_padded_offsets_tensor,
            group_idx_tensor,
            out,
        )
        torch.cuda.synchronize()
        print("SMOKE_EXECUTION_OK", flush=True)
        print("OUT_MAX", float(out.abs().max().item()), flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
