import importlib.util
import math
import sys
import time

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


def bench(name, function, iterations=10):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    print("STAGE", name, elapsed, flush=True)
    return elapsed


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

    expert_block_m = module._EXPERT_BLOCK_M
    pack, swiglu, _, _, unpack_chunks = module._get_kernels(
        hidden,
        intermediate,
        n_experts,
        padded_sum,
        group_sum,
        expert_block_m,
    )
    packed_input, _ = module._get_workspace(
        x, intermediate, n_experts, expert_block_m
    )
    pack(x, group_sizes, padded_offsets, packed_input)
    gate_output = packed_input @ gate_w.transpose(1, 2)
    up_output = packed_input @ up_w.transpose(1, 2)
    activation = torch.empty_like(gate_output)
    swiglu(gate_output, up_output, group_sizes, activation)
    torch.cuda.synchronize()

    sink = [None]
    bench("full", lambda: module.run_kernel(*args))
    bench("pack", lambda: pack(x, group_sizes, padded_offsets, packed_input))

    def gate_bmm():
        sink[0] = packed_input @ gate_w.transpose(1, 2)

    def up_bmm():
        sink[0] = packed_input @ up_w.transpose(1, 2)

    bench("gate_bmm", gate_bmm)
    bench("up_bmm", up_bmm)
    bench(
        "swiglu",
        lambda: swiglu(gate_output, up_output, group_sizes, activation),
    )

    expert_chunk = 32 if n_experts == 32 else 16

    def all_fc2():
        outputs = []
        for expert_start in range(0, n_experts, expert_chunk):
            expert_end = min(expert_start + expert_chunk, n_experts)
            outputs.append(
                activation[expert_start:expert_end]
                @ down_w[expert_start:expert_end].transpose(1, 2)
            )
        sink[0] = outputs

    all_fc2()
    torch.cuda.synchronize()
    packed_outputs = sink[0]
    bench("all_fc2", all_fc2)

    def all_unpack():
        for chunk_id, expert_start in enumerate(range(0, n_experts, expert_chunk)):
            unpack_chunks[chunk_id](
                packed_outputs[chunk_id],
                route_w,
                group_sizes,
                group_offsets,
                padded_offsets,
                out,
            )

    bench("all_unpack", all_unpack)
    print("PROFILE_CASE", case_name, n_experts, hidden, intermediate, padded_sum)


if __name__ == "__main__":
    run(sys.argv[1])
