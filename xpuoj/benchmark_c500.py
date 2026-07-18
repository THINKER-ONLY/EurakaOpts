import argparse
import gc
import importlib.util
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch


BLOCK_M = 128


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    hidden: int
    intermediate: int
    experts: int
    valid_rows: int
    seed: int = 81394


CASES = {
    case.name: case
    for case in (
        BenchmarkCase("oj_case1_proxy", 2048, 8192, 8, 4096),
        BenchmarkCase("oj_case2_proxy", 7168, 2048, 256, 4096),
        BenchmarkCase("oj_case3_proxy", 7168, 2048, 256, 32768),
    )
}


def load_submission(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_kernel", None)):
        raise AttributeError(f"{path} does not expose callable run_kernel")
    return module


def build_counts(experts: int, valid_rows: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    assignments = torch.randint(
        experts,
        (valid_rows,),
        generator=generator,
        device="cpu",
    )
    return torch.bincount(assignments, minlength=experts).tolist()


def build_metadata(counts):
    raw_offsets = [0]
    padded_offsets = [0]
    for count in counts:
        raw_offsets.append(raw_offsets[-1] + count)
        padded_offsets.append(
            padded_offsets[-1] + math.ceil((count + 1) / BLOCK_M) * BLOCK_M
        )

    block_map = []
    for expert, (start, end) in enumerate(
        zip(padded_offsets[:-1], padded_offsets[1:])
    ):
        block_map.extend([expert] * ((end - start) // BLOCK_M))
    return raw_offsets, padded_offsets, block_map


def allocate_inputs(case: BenchmarkCase, outputs: int):
    counts = build_counts(case.experts, case.valid_rows, case.seed)
    raw_offsets, padded_offsets, block_map = build_metadata(counts)
    device = "cuda"

    stacked = torch.empty(
        (padded_offsets[-1], case.hidden),
        device=device,
        dtype=torch.float16,
    ).fill_(0.01)
    gate = torch.empty(
        (case.experts, case.intermediate, case.hidden),
        device=device,
        dtype=torch.float16,
    ).fill_(0.01)
    up = torch.empty_like(gate).fill_(0.0125)
    down = torch.empty(
        (case.experts, case.hidden, case.intermediate),
        device=device,
        dtype=torch.float16,
    ).fill_(0.01)
    route_weights = torch.empty(
        raw_offsets[-1],
        device=device,
        dtype=torch.float32,
    ).fill_(0.5)
    group_sizes = torch.tensor(counts, device=device, dtype=torch.int32)
    group_offsets = torch.tensor(raw_offsets, device=device, dtype=torch.int32)
    group_padded_offsets = torch.tensor(
        padded_offsets,
        device=device,
        dtype=torch.int32,
    )
    group_idx_for_bx = torch.tensor(block_map, device=device, dtype=torch.int32)
    output_tensors = [torch.zeros_like(stacked) for _ in range(outputs)]
    common_args = (
        stacked,
        gate,
        up,
        down,
        route_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
    )
    return common_args, output_tensors, counts, padded_offsets


def valid_mask(counts, padded_offsets, device="cuda"):
    mask = torch.zeros(padded_offsets[-1], device=device, dtype=torch.bool)
    for expert, count in enumerate(counts):
        mask[padded_offsets[expert] : padded_offsets[expert] + count] = True
    return mask


def time_batch(module, call_args, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        module.run_kernel(*call_args)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def time_paired(baseline, baseline_args, candidate, candidate_args, iterations):
    baseline_events = []
    candidate_events = []
    for index in range(iterations):
        baseline_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        candidate_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        if index % 2:
            candidate_pair[0].record()
            candidate.run_kernel(*candidate_args)
            candidate_pair[1].record()
            baseline_pair[0].record()
            baseline.run_kernel(*baseline_args)
            baseline_pair[1].record()
        else:
            baseline_pair[0].record()
            baseline.run_kernel(*baseline_args)
            baseline_pair[1].record()
            candidate_pair[0].record()
            candidate.run_kernel(*candidate_args)
            candidate_pair[1].record()
        baseline_events.append(baseline_pair)
        candidate_events.append(candidate_pair)
    torch.cuda.synchronize()
    baseline_ms = statistics.mean(
        float(start.elapsed_time(end)) for start, end in baseline_events
    )
    candidate_ms = statistics.mean(
        float(start.elapsed_time(end)) for start, end in candidate_events
    )
    return baseline_ms, candidate_ms


def measure_case(
    baseline,
    candidate,
    case,
    warmup,
    iterations,
    samples,
):
    torch.cuda.reset_peak_memory_stats()
    module_count = 2 if candidate is not None else 1
    common_args, outputs, counts, padded_offsets = allocate_inputs(case, module_count)
    baseline_args = (*common_args, outputs[0])
    candidate_args = (*common_args, outputs[1]) if candidate is not None else None

    baseline.run_kernel(*baseline_args)
    if candidate is not None:
        candidate.run_kernel(*candidate_args)
    torch.cuda.synchronize()

    for index in range(warmup):
        if candidate is not None and index % 2:
            candidate.run_kernel(*candidate_args)
            baseline.run_kernel(*baseline_args)
        else:
            baseline.run_kernel(*baseline_args)
            if candidate is not None:
                candidate.run_kernel(*candidate_args)
    torch.cuda.synchronize()

    comparison = None
    if candidate is not None:
        mask = valid_mask(counts, padded_offsets)
        torch.testing.assert_close(
            outputs[1][mask].float(),
            outputs[0][mask].float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            outputs[1][~mask],
            torch.zeros_like(outputs[1][~mask]),
            atol=0,
            rtol=0,
        )
        difference = (outputs[1][mask].float() - outputs[0][mask].float()).abs()
        comparison = {
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        }
        del mask, difference

    baseline_samples = []
    candidate_samples = []
    for index in range(samples):
        if candidate is None:
            baseline_samples.append(time_batch(baseline, baseline_args, iterations))
        else:
            baseline_ms, candidate_ms = time_paired(
                baseline,
                baseline_args,
                candidate,
                candidate_args,
                iterations,
            )
            baseline_samples.append(baseline_ms)
            candidate_samples.append(candidate_ms)

    baseline_median = statistics.median(baseline_samples)
    result = {
        "name": case.name,
        "hidden": case.hidden,
        "intermediate": case.intermediate,
        "experts": case.experts,
        "valid_rows": case.valid_rows,
        "padded_rows": padded_offsets[-1],
        "blocks": len(common_args[-1]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "baseline_ms": baseline_samples,
        "baseline_median_ms": baseline_median,
    }
    if candidate is not None:
        candidate_median = statistics.median(candidate_samples)
        paired_improvements = [
            100.0 * (baseline_ms - candidate_ms) / baseline_ms
            for baseline_ms, candidate_ms in zip(
                baseline_samples,
                candidate_samples,
            )
        ]
        result.update(
            {
                "candidate_ms": candidate_samples,
                "candidate_median_ms": candidate_median,
                "paired_improvement_percent": paired_improvements,
                "improvement_percent": statistics.median(paired_improvements),
                "comparison": comparison,
            }
        )

    for module in (baseline, candidate):
        if module is not None and hasattr(module, "_WORKSPACE_CACHE"):
            module._WORKSPACE_CACHE.clear()
    del common_args, outputs, baseline_args, candidate_args
    gc.collect()
    torch.cuda.empty_cache()
    return result


def correctness_case(module, hidden, intermediate):
    case = BenchmarkCase(
        name=f"correctness_{hidden}x{intermediate}",
        hidden=hidden,
        intermediate=intermediate,
        experts=1,
        valid_rows=142,
    )
    common_args, outputs, counts, padded_offsets = allocate_inputs(case, 1)
    stacked, gate, up, down, route_weights = common_args[:5]
    count = counts[0]
    x = stacked[:count].float()
    gate_logits = x @ gate[0].float().transpose(0, 1)
    up_logits = x @ up[0].float().transpose(0, 1)
    activated = torch.nn.functional.silu(gate_logits) * up_logits
    reference = activated @ down[0].float().transpose(0, 1)
    reference *= route_weights[:count, None]

    module.run_kernel(*common_args, outputs[0])
    torch.cuda.synchronize()
    torch.testing.assert_close(
        outputs[0][:count].float(),
        reference,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        outputs[0][count : padded_offsets[-1]],
        torch.zeros_like(outputs[0][count : padded_offsets[-1]]),
        atol=0,
        rtol=0,
    )
    difference = (outputs[0][:count].float() - reference).abs()
    result = {
        "name": case.name,
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }
    if hasattr(module, "_WORKSPACE_CACHE"):
        module._WORKSPACE_CACHE.clear()
    del common_args, outputs, reference, difference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_cases(value):
    if value == "all":
        return list(CASES.values())
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - CASES.keys())
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return [CASES[name] for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("warmup must be non-negative; iterations and samples positive")

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve() if args.candidate else None
    baseline = load_submission(baseline_path, "c500_baseline")
    candidate = (
        load_submission(candidate_path, "c500_candidate")
        if candidate_path is not None
        else None
    )

    correctness = {}
    if not args.skip_correctness:
        correctness["baseline"] = [
            correctness_case(baseline, 2048, 8192),
            correctness_case(baseline, 7168, 2048),
        ]
        if candidate is not None:
            correctness["candidate"] = [
                correctness_case(candidate, 2048, 8192),
                correctness_case(candidate, 7168, 2048),
            ]

    results = []
    for case in parse_cases(args.cases):
        results.append(
            measure_case(
                baseline,
                candidate,
                case,
                warmup=args.warmup,
                iterations=args.iterations,
                samples=args.samples,
            )
        )

    report = {
        "device": torch.cuda.get_device_name(0),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path) if candidate_path else None,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": args.samples,
        "correctness": correctness,
        "cases": results,
    }
    if candidate is not None:
        baseline_total = sum(item["baseline_median_ms"] for item in results)
        candidate_total = sum(item["candidate_median_ms"] for item in results)
        aggregate_improvements = []
        for sample_index in range(args.samples):
            baseline_sample_total = sum(
                item["baseline_ms"][sample_index] for item in results
            )
            candidate_sample_total = sum(
                item["candidate_ms"][sample_index] for item in results
            )
            aggregate_improvements.append(
                100.0
                * (baseline_sample_total - candidate_sample_total)
                / baseline_sample_total
            )
        report["aggregate"] = {
            "baseline_ms": baseline_total,
            "candidate_ms": candidate_total,
            "paired_improvement_percent": aggregate_improvements,
            "improvement_percent": statistics.median(aggregate_improvements),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
