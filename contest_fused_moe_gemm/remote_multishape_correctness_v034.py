import gc
import importlib.util
import math
import traceback

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v034_internal_global_workspace.py"
BLOCK_TOKEN = 128
CONFIGS = (
    ("case1", 16, 2048, 8192, 2272),
    ("case2", 32, 7168, 2048, 4544),
    ("case3", 64, 7168, 2048, 9088),
)


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_metadata(group_sum, n_experts, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(81394)
    assignments = torch.randint(
        0,
        n_experts,
        (group_sum,),
        generator=generator,
        dtype=torch.int64,
    )
    counts = torch.bincount(assignments, minlength=n_experts).tolist()

    group_offsets = []
    group_padded_offsets = []
    raw_offset = 0
    padded_offset = 0
    for count in counts:
        group_offsets.append(raw_offset)
        group_padded_offsets.append(padded_offset)
        raw_offset += count
        padded_offset += math.ceil((count + 1) / BLOCK_TOKEN) * BLOCK_TOKEN

    group_idx = []
    expert_id = 0
    for bx in range(padded_offset // BLOCK_TOKEN):
        padded_start = bx * BLOCK_TOKEN
        while (
            expert_id + 1 < n_experts
            and padded_start >= group_padded_offsets[expert_id + 1]
        ):
            expert_id += 1
        group_idx.append(expert_id)

    return (
        counts,
        group_offsets,
        group_padded_offsets,
        padded_offset,
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(group_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_padded_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_idx, dtype=torch.int32, device=device),
    )


def checksum(tensor):
    values = tensor.float()
    return float(values.sum().item()), float(values.square().sum().item())


def run_case(module, case_name, n_experts, d_hidden, d_expert, group_sum):
    device = "cuda"
    torch.manual_seed(81394)
    (
        counts,
        group_offsets,
        group_padded_offsets,
        padded_sum,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
    ) = build_metadata(group_sum, n_experts, device)

    x = torch.randn((padded_sum, d_hidden), dtype=torch.float16, device=device)
    gate_w = torch.randn(
        (n_experts, d_expert, d_hidden), dtype=torch.float16, device=device
    ) / math.sqrt(d_expert)
    up_w = torch.randn_like(gate_w) / math.sqrt(d_expert)
    down_w = torch.randn(
        (n_experts, d_hidden, d_expert), dtype=torch.float16, device=device
    ) / math.sqrt(d_hidden)
    route_w = torch.rand((group_sum,), dtype=torch.float16, device=device)
    out = torch.full(
        (padded_sum, d_hidden),
        7,
        dtype=torch.float16,
        device=device,
    )

    float_inputs = [x, gate_w, up_w, down_w, route_w]
    metadata = [
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
    ]
    before_checksums = [checksum(tensor) for tensor in float_inputs]
    metadata_copies = [tensor.clone() for tensor in metadata]

    print(
        "MULTISHAPE_CASE_BEGIN",
        case_name,
        n_experts,
        d_hidden,
        d_expert,
        group_sum,
        padded_sum,
        flush=True,
    )
    module.run_kernel(
        x,
        gate_w,
        up_w,
        down_w,
        route_w,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
        out,
    )
    torch.cuda.synchronize()

    max_abs_error = 0.0
    padding_max = 0.0
    for expert_id, count in enumerate(counts):
        padded_start = group_padded_offsets[expert_id]
        raw_start = group_offsets[expert_id]
        if count:
            expert_x = x[padded_start : padded_start + count].float()
            gate = expert_x @ gate_w[expert_id].float().t()
            up = expert_x @ up_w[expert_id].float().t()
            activated = (torch.nn.functional.silu(gate) * up).half()
            reference = activated.float() @ down_w[expert_id].float().t()
            reference *= route_w[raw_start : raw_start + count].float().view(-1, 1)
            reference = reference.half().float()
            actual = out[padded_start : padded_start + count].float()
            error = (actual - reference).abs()
            torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
            max_abs_error = max(max_abs_error, float(error.max().item()))
            del expert_x, gate, up, activated, reference, actual, error

        padded_end = (
            group_padded_offsets[expert_id + 1]
            if expert_id + 1 < n_experts
            else padded_sum
        )
        if padded_start + count < padded_end:
            padding_max = max(
                padding_max,
                float(out[padded_start + count : padded_end].abs().max().item()),
            )
        torch.cuda.empty_cache()

    assert padding_max == 0.0
    assert before_checksums == [checksum(tensor) for tensor in float_inputs]
    for tensor, copy in zip(metadata, metadata_copies):
        assert torch.equal(tensor, copy)

    expected_cache_size = int(case_name[-1])
    actual_cache_size = len(module._KERNEL_CACHE)
    assert actual_cache_size == expected_cache_size
    print("MAX_ABS_ERROR", case_name, max_abs_error, flush=True)
    print("PADDING_MAX", case_name, padding_max, flush=True)
    print(
        "MODULE_CACHE_SIZE_AFTER_CASE",
        case_name,
        actual_cache_size,
        flush=True,
    )
    print("MULTISHAPE_CASE_PASS", case_name, flush=True)

    del (
        x,
        gate_w,
        up_w,
        down_w,
        route_w,
        out,
        group_sizes_tensor,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
        metadata_copies,
    )
    gc.collect()
    torch.cuda.empty_cache()


def main():
    module = load_candidate()
    for config in CONFIGS:
        run_case(module, *config)
    assert len(module._KERNEL_CACHE) == 3
    print("MULTISHAPE_CORRECTNESS_PASS", flush=True)
    print("FINAL_MODULE_CACHE_SIZE", len(module._KERNEL_CACHE), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
