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
    padded_offsets = [0]
    for count in counts:
        padded_offsets.append(
            padded_offsets[-1] + math.ceil((count + 1) / BLOCK_TOKEN) * BLOCK_TOKEN
        )
    return (
        padded_offsets[-1],
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(padded_offsets, dtype=torch.int32, device=device),
    )


def bench(name, function, iterations=20):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    print("VARIANT", name, elapsed, flush=True)


def run(case_name):
    n_experts, hidden, intermediate, group_sum = CONFIGS[case_name]
    device = "cuda"
    torch.manual_seed(81394)
    padded_sum, group_sizes, padded_offsets = build_metadata(
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

    module = load_candidate()
    pack, swiglu, _, _, _ = module._get_kernels(
        hidden,
        intermediate,
        n_experts,
        padded_sum,
        group_sum,
        module._EXPERT_BLOCK_M,
    )
    packed_input, _ = module._get_workspace(
        x, intermediate, n_experts, module._EXPERT_BLOCK_M
    )
    pack(x, group_sizes, padded_offsets, packed_input)
    gate_output = packed_input @ gate_w.transpose(1, 2)
    up_output = packed_input @ up_w.transpose(1, 2)
    activation = torch.empty_like(gate_output)
    swiglu(gate_output, up_output, group_sizes, activation)
    torch.cuda.synchronize()

    gate_preallocated = torch.empty_like(gate_output)
    bench(
        "fc1_alloc",
        lambda: torch.bmm(packed_input, gate_w.transpose(1, 2)),
    )
    bench(
        "fc1_out",
        lambda: torch.bmm(
            packed_input,
            gate_w.transpose(1, 2),
            out=gate_preallocated,
        ),
    )

    for chunk in (8, 16, 32, 64):
        if chunk > n_experts or n_experts % chunk:
            continue

        def fc1_chunks():
            result = []
            for start in range(0, n_experts, chunk):
                result.append(
                    torch.bmm(
                        packed_input[start : start + chunk],
                        gate_w[start : start + chunk].transpose(1, 2),
                    )
                )
            return result

        bench(f"fc1_alloc_chunk{chunk}", fc1_chunks)

    for chunk in (8, 16, 20, 24, 32, 64):
        if chunk > n_experts or n_experts % chunk:
            continue
        outputs = [
            torch.empty(
                (chunk, module._EXPERT_BLOCK_M, hidden),
                dtype=torch.float16,
                device=device,
            )
            for _ in range(n_experts // chunk)
        ]

        def fc2_alloc():
            result = []
            for start in range(0, n_experts, chunk):
                result.append(
                    torch.bmm(
                        activation[start : start + chunk],
                        down_w[start : start + chunk].transpose(1, 2),
                    )
                )
            return result

        def fc2_out():
            for chunk_id, start in enumerate(range(0, n_experts, chunk)):
                torch.bmm(
                    activation[start : start + chunk],
                    down_w[start : start + chunk].transpose(1, 2),
                    out=outputs[chunk_id],
                )

        def fc2_out_shared():
            for start in range(0, n_experts, chunk):
                torch.bmm(
                    activation[start : start + chunk],
                    down_w[start : start + chunk].transpose(1, 2),
                    out=outputs[0],
                )

        bench(f"fc2_alloc_chunk{chunk}", fc2_alloc)
        bench(f"fc2_out_chunk{chunk}", fc2_out)
        bench(f"fc2_out_shared_chunk{chunk}", fc2_out_shared)

    print("BMM_CASE", case_name, n_experts, hidden, intermediate, padded_sum)


if __name__ == "__main__":
    run(sys.argv[1])
