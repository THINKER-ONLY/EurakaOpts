import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("xpuoj_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_submission(module, call_args, warmup, iterations):
    for _ in range(warmup):
        module.run_kernel(*call_args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        module.run_kernel(*call_args)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def build_metadata(counts, block_m=128):
    raw_offsets = [0]
    padded_offsets = [0]
    for count in counts:
        raw_offsets.append(raw_offsets[-1] + count)
        padded_offsets.append(
            padded_offsets[-1] + math.ceil((count + 1) / block_m) * block_m
        )

    group_idx_for_bx = []
    for bx in range(padded_offsets[-1] // block_m):
        block_start = bx * block_m
        expert = max(
            index
            for index, padded_start in enumerate(padded_offsets[:-1])
            if block_start >= padded_start
        )
        group_idx_for_bx.append(expert)
    return raw_offsets, padded_offsets, group_idx_for_bx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--candidate-timing", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--counts", default="142,65,128")
    args = parser.parse_args()

    torch.manual_seed(81394)
    device = "cuda"
    counts = [int(value) for value in args.counts.split(",") if value]
    if not counts or any(count < 0 for count in counts):
        raise ValueError("counts must contain non-negative integers")
    hidden = args.hidden
    intermediate = args.intermediate
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("hidden and intermediate must be positive")
    experts = len(counts)
    raw_offsets, padded_offsets, block_map = build_metadata(counts)

    stacked = torch.randn(
        (padded_offsets[-1], hidden), device=device, dtype=torch.float16
    )
    gate = (
        torch.randn(
            (experts, intermediate, hidden), device=device, dtype=torch.float16
        )
        / math.sqrt(hidden)
    ).contiguous()
    up = (
        torch.randn(
            (experts, intermediate, hidden), device=device, dtype=torch.float16
        )
        / math.sqrt(hidden)
    ).contiguous()
    down = (
        torch.randn(
            (experts, hidden, intermediate), device=device, dtype=torch.float16
        )
        / math.sqrt(intermediate)
    ).contiguous()
    route_weights = torch.rand(
        raw_offsets[-1], device=device, dtype=torch.float32
    ).contiguous()
    group_sizes = torch.tensor(counts, device=device, dtype=torch.int32)
    group_offsets = torch.tensor(raw_offsets, device=device, dtype=torch.int32)
    group_padded_offsets = torch.tensor(
        padded_offsets, device=device, dtype=torch.int32
    )
    group_idx_for_bx = torch.tensor(block_map, device=device, dtype=torch.int32)

    reference = torch.zeros_like(stacked)
    for expert, count in enumerate(counts):
        padded_start = padded_offsets[expert]
        raw_start = raw_offsets[expert]
        x = stacked[padded_start : padded_start + count].float()
        gate_logits = x @ gate[expert].float().transpose(0, 1)
        up_logits = x @ up[expert].float().transpose(0, 1)
        hidden_act = torch.nn.functional.silu(gate_logits) * up_logits
        y = hidden_act @ down[expert].float().transpose(0, 1)
        y *= route_weights[raw_start : raw_start + count, None]
        reference[padded_start : padded_start + count] = y.half()

    submission = load_submission(args.submission.resolve())
    output = torch.zeros_like(stacked)
    call_args = (
        stacked,
        gate,
        up,
        down,
        route_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        output,
    )
    submission.run_kernel(*call_args)
    submission.run_kernel(*call_args)
    torch.cuda.synchronize()

    valid_mask = torch.zeros(output.shape[0], device=device, dtype=torch.bool)
    for expert, count in enumerate(counts):
        padded_start = padded_offsets[expert]
        valid_mask[padded_start : padded_start + count] = True

    torch.testing.assert_close(
        output[valid_mask].float(),
        reference[valid_mask].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        output[~valid_mask],
        torch.zeros_like(output[~valid_mask]),
        atol=0,
        rtol=0,
    )
    difference = (output[valid_mask].float() - reference[valid_mask].float()).abs()
    result = {
        "status": "pass",
        "counts": counts,
        "raw_rows": raw_offsets[-1],
        "padded_rows": padded_offsets[-1],
        "blocks": len(block_map),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "device": torch.cuda.get_device_name(0),
    }

    if args.candidate_timing:
        candidate_samples = [
            time_submission(
                submission,
                call_args,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for _ in range(2)
        ]
        result["cuda_proxy"] = {
            "candidate_ms": candidate_samples,
            "candidate_median_ms": statistics.median(candidate_samples),
            "warning": "stability check only; no baseline and no C500 evidence",
        }

    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
