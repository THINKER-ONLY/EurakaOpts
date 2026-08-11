import importlib.util
import math
import sys

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v160_e16_fc1_sequential_free.py"
CONFIGS = {
    "case1": (16, 2048, 8192, 2272),
    "case2": (32, 7168, 2048, 4544),
    "case3": (64, 7168, 2048, 9088),
}
BLOCK_TOKEN = 128


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_metadata(group_sum, n_experts, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(81394)
    assignments = torch.randint(
        0, n_experts, (group_sum,), generator=generator, dtype=torch.int64
    )
    counts = torch.bincount(assignments, minlength=n_experts).tolist()
    offsets = [0]
    padded_offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
        padded_offsets.append(
            padded_offsets[-1] + math.ceil((count + 1) / BLOCK_TOKEN) * BLOCK_TOKEN
        )
    group_idx = []
    expert_id = 0
    for bx in range(padded_offsets[-1] // BLOCK_TOKEN):
        padded_start = bx * BLOCK_TOKEN
        while expert_id + 1 < n_experts and padded_start >= padded_offsets[expert_id + 1]:
            expert_id += 1
        group_idx.append(expert_id)
    return (
        padded_offsets[-1],
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(padded_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_idx, dtype=torch.int32, device=device),
    )


def run(case_name):
    n_experts, hidden, intermediate, group_sum = CONFIGS[case_name]
    device = "cuda"
    torch.manual_seed(81394)
    padded_sum, group_sizes, group_offsets, padded_offsets, group_idx = build_metadata(
        group_sum, n_experts, device
    )
    x = torch.randn((padded_sum, hidden), dtype=torch.float16, device=device)
    gate_w = torch.randn(
        (n_experts, intermediate, hidden), dtype=torch.float16, device=device
    ) / math.sqrt(intermediate)
    up_w = torch.randn_like(gate_w) / math.sqrt(intermediate)
    down_w = torch.randn(
        (n_experts, hidden, intermediate), dtype=torch.float16, device=device
    ) / math.sqrt(hidden)
    route_w = torch.rand((group_sum,), dtype=torch.float32, device=device)
    out = torch.zeros((padded_sum, hidden), dtype=torch.float16, device=device)
    module = load_candidate()
    args = (
        x,
        gate_w,
        up_w,
        down_w,
        route_w,
        group_sizes,
        group_offsets,
        padded_offsets,
        group_idx,
        out,
    )
    for _ in range(3):
        module.run_kernel(*args)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for _ in range(10):
            module.run_kernel(*args)
        torch.cuda.synchronize()
    print("PROFILE_CASE", case_name, n_experts, hidden, intermediate, padded_sum)
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
    )


if __name__ == "__main__":
    run(sys.argv[1])
