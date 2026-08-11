import gc
import importlib.util
import math
import statistics
import sys
import traceback

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v022_expert_tile_64_layout_fixed.py"
BLOCK_TOKEN = 128
WARMUP = 1
ITERATIONS = 1
CONFIGS = [
    {
        "name": "performance_1",
        "d_hidden": 7168,
        "d_expert": 2048,
        "n_experts": 8,
        "raw_rows": 4 * 8192 * 4,
        "seed": 81394,
    },
    {
        "name": "performance_2",
        "d_hidden": 3584,
        "d_expert": 1024,
        "n_experts": 4,
        "raw_rows": 8 * 4096 * 2,
        "seed": 81394,
    },
]


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def make_case(config):
    device = "cuda"
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    n_experts = config["n_experts"]
    raw_rows = config["raw_rows"]
    counts = [raw_rows // n_experts] * n_experts

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
        while expert_id + 1 < n_experts and padded_start >= group_padded_offsets[expert_id + 1]:
            expert_id += 1
        group_idx.append(expert_id)

    generator = torch.Generator(device=device)
    generator.manual_seed(config["seed"])
    x = torch.randn(
        (padded_offset, d_hidden),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    gate_w = torch.randn(
        (n_experts, d_expert, d_hidden),
        dtype=torch.float16,
        device=device,
        generator=generator,
    ) / math.sqrt(d_expert)
    up_w = torch.randn(
        (n_experts, d_expert, d_hidden),
        dtype=torch.float16,
        device=device,
        generator=generator,
    ) / math.sqrt(d_expert)
    down_w = torch.randn(
        (n_experts, d_hidden, d_expert),
        dtype=torch.float16,
        device=device,
        generator=generator,
    ) / math.sqrt(d_hidden)
    route_w = torch.rand(
        (raw_rows,), dtype=torch.float16, device=device, generator=generator
    )
    group_sizes = torch.tensor(counts, dtype=torch.int32, device=device)
    group_offsets_tensor = torch.tensor(group_offsets, dtype=torch.int32, device=device)
    group_padded_offsets_tensor = torch.tensor(
        group_padded_offsets, dtype=torch.int32, device=device
    )
    group_idx_tensor = torch.tensor(group_idx, dtype=torch.int32, device=device)
    out = torch.empty((padded_offset, d_hidden), dtype=torch.float16, device=device)

    args = (
        x,
        gate_w,
        up_w,
        down_w,
        route_w,
        group_sizes,
        group_offsets_tensor,
        group_padded_offsets_tensor,
        group_idx_tensor,
        out,
    )
    return args, counts, group_padded_offsets, padded_offset


def run_case(module, config):
    args, counts, padded_offsets, padded_rows = make_case(config)
    out = args[-1]
    print(
        "PERF_CASE_BEGIN",
        config["name"],
        config["d_hidden"],
        config["d_expert"],
        config["n_experts"],
        config["raw_rows"],
        padded_rows,
        flush=True,
    )

    for _ in range(WARMUP):
        module.run_kernel(*args)
    torch.cuda.synchronize()
    print("WARMUP_PASS", config["name"], WARMUP, flush=True)

    elapsed = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.run_kernel(*args)
        end.record()
        torch.cuda.synchronize()
        elapsed.append(float(start.elapsed_time(end)))

    padding_max = 0.0
    for expert_id, count in enumerate(counts):
        start = padded_offsets[expert_id] + count
        end = padded_offsets[expert_id + 1] if expert_id + 1 < len(counts) else padded_rows
        if start < end:
            padding_max = max(padding_max, float(out[start:end].abs().max().item()))

    assert torch.isfinite(out).all()
    assert padding_max == 0.0

    mean_ms = statistics.fmean(elapsed)
    print("TIMES_MS", config["name"], " ".join(f"{value:.6f}" for value in elapsed), flush=True)
    print("MEAN_MS", config["name"], mean_ms, flush=True)
    print("MIN_MS", config["name"], min(elapsed), flush=True)
    print("P50_MS", config["name"], percentile(elapsed, 0.5), flush=True)
    print("P90_MS", config["name"], percentile(elapsed, 0.9), flush=True)
    print("PADDING_MAX", config["name"], padding_max, flush=True)
    print("PERF_CASE_PASS", config["name"], flush=True)

    del args, out
    gc.collect()
    torch.cuda.empty_cache()


def main():
    module = load_candidate()
    selected = sys.argv[1] if len(sys.argv) > 1 else None
    configs = [config for config in CONFIGS if selected in (None, config["name"])]
    if not configs:
        raise ValueError(f"unknown performance case: {selected}")
    for config in configs:
        run_case(module, config)
    print("ALL_PERFORMANCE_CASES_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
