import argparse
import gc
import importlib.util
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch: int
    seq_len: int
    heads_kv: int
    heads_q: int
    dim: int
    selected_blocks: int
    block_size: int
    is_causal: int = 1
    index_mode: str = "recent"


CASES = {
    case.name: case
    for case in (
        BenchmarkCase("short_s1", 1, 1024, 1, 16, 64, 1, 16),
        BenchmarkCase("long_s1", 1, 16384, 1, 16, 64, 1, 16),
        BenchmarkCase("multi_s4", 4, 1024, 1, 16, 64, 4, 16),
        BenchmarkCase("wide_bs32", 2, 1024, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b1_l512", 1, 512, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b1_l1024", 1, 1024, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b2_l512", 2, 512, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b4_l512", 4, 512, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b4_l1024", 4, 1024, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b8_l512", 8, 512, 1, 16, 128, 1, 32),
        BenchmarkCase("wide_bs32_b8_l256", 8, 256, 1, 16, 128, 1, 32),
        BenchmarkCase("target_64k_s16", 1, 65536, 1, 16, 64, 16, 64),
        BenchmarkCase("official_d32_s1", 8, 2048, 1, 16, 32, 1, 16),
        BenchmarkCase("official_d128_s1", 4, 1024, 1, 16, 128, 1, 16),
        BenchmarkCase("official_d128_long_s1", 1, 8192, 1, 16, 128, 1, 16),
        BenchmarkCase("official_bs64_s1", 4, 128, 1, 16, 64, 1, 64),
        BenchmarkCase("official_s2", 4, 1024, 1, 16, 64, 2, 16),
        BenchmarkCase("official_s8", 4, 1024, 1, 16, 64, 8, 16),
        BenchmarkCase(
            "historical_d64_s1", 1, 16384, 1, 16, 64, 1, 16,
            index_mode="historical",
        ),
        BenchmarkCase(
            "historical_d128_s1", 1, 8192, 1, 16, 128, 1, 16,
            index_mode="historical",
        ),
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


def build_indices(case: BenchmarkCase):
    token = torch.arange(case.seq_len, device="cuda", dtype=torch.int32)
    current_block = token // case.block_size
    indices = []
    for selected in range(case.selected_blocks):
        if case.index_mode == "historical":
            block_count = torch.clamp(current_block, min=1)
            hashed = token.to(torch.int64) * 1103515245 + selected * 12345
            index = torch.remainder(
                hashed, block_count.to(torch.int64)
            ).to(torch.int32)
            valid = selected < block_count
        else:
            index = current_block - selected
            valid = index >= 0
        index = torch.where(
            valid,
            index,
            torch.full_like(index, case.seq_len),
        )
        indices.append(index)
    stacked = torch.stack(indices, dim=-1)
    return stacked[None, :, None, :].expand(
        case.batch, case.seq_len, case.heads_kv, case.selected_blocks
    ).contiguous()


def allocate_inputs(case: BenchmarkCase, outputs: int, seed: int = 81394):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (case.batch, case.seq_len, case.heads_q, case.dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (case.batch, case.seq_len, case.heads_kv, case.dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn(
        (case.batch, case.seq_len, case.heads_kv, case.dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    block_indices = build_indices(case)
    output_tensors = [torch.empty_like(q) for _ in range(outputs)]
    common_args = (
        q,
        k,
        v,
        block_indices,
        case.batch,
        case.seq_len,
        case.heads_kv,
        case.heads_q,
        case.dim,
        case.selected_blocks,
        case.block_size,
        case.is_causal,
    )
    return common_args, output_tensors


def call_args(common_args, output):
    return (*common_args[:4], output, *common_args[4:])


def reference(q, k, v, block_indices, block_size, is_causal):
    batch, seq_len, heads_q, dim = q.shape
    heads_kv = k.shape[2]
    groups = heads_q // heads_kv
    output = torch.empty_like(q, dtype=torch.float32)
    scale = dim**-0.5
    for batch_id in range(batch):
        for token in range(seq_len):
            for head in range(heads_kv):
                positions = []
                for block in block_indices[batch_id, token, head].tolist():
                    start = int(block) * block_size
                    if start < 0 or start > token or start >= seq_len:
                        continue
                    end = min(start + block_size, seq_len)
                    positions.extend(range(start, end))
                position_tensor = torch.tensor(positions, device="cuda")
                if is_causal:
                    position_tensor = position_tensor[position_tensor <= token]
                key = k[batch_id, position_tensor, head].float()
                value = v[batch_id, position_tensor, head].float()
                query = q[
                    batch_id,
                    token,
                    head * groups : (head + 1) * groups,
                ].float()
                scores = query @ key.transpose(0, 1) * scale
                attention = torch.softmax(scores, dim=-1)
                output[
                    batch_id,
                    token,
                    head * groups : (head + 1) * groups,
                ] = attention @ value
    return output


def correctness_case(module):
    case = BenchmarkCase("correctness", 1, 48, 1, 16, 64, 2, 16)
    common_args, outputs = allocate_inputs(case, 1, seed=17)
    args = call_args(common_args, outputs[0])
    expected = reference(
        *common_args[:4], case.block_size, bool(case.is_causal)
    )
    module.run_kernel(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0].float(), expected, atol=1e-2, rtol=1e-2)
    first_max_abs = float((outputs[0].float() - expected).abs().max().item())

    generator = torch.Generator(device="cuda")
    generator.manual_seed(23)
    for tensor in common_args[:3]:
        tensor.normal_(generator=generator)
    outputs[0].fill_(float("nan"))
    expected_changed = reference(
        *common_args[:4], case.block_size, bool(case.is_causal)
    )
    module.run_kernel(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        outputs[0].float(), expected_changed, atol=1e-2, rtol=1e-2
    )
    changed_max_abs = float(
        (outputs[0].float() - expected_changed).abs().max().item()
    )
    del common_args, outputs, expected, expected_changed
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "first_max_abs": first_max_abs,
        "changed_input_max_abs": changed_max_abs,
    }


def time_batch(module, args, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        module.run_kernel(*args)
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
    return (
        statistics.mean(float(a.elapsed_time(b)) for a, b in baseline_events),
        statistics.mean(float(a.elapsed_time(b)) for a, b in candidate_events),
    )


def measure_case(baseline, candidate, case, warmup, iterations, samples):
    torch.cuda.reset_peak_memory_stats()
    common_args, outputs = allocate_inputs(case, 2 if candidate else 1)
    baseline_args = call_args(common_args, outputs[0])
    candidate_args = call_args(common_args, outputs[1]) if candidate else None

    baseline.run_kernel(*baseline_args)
    if candidate:
        candidate.run_kernel(*candidate_args)
    torch.cuda.synchronize()

    for index in range(warmup):
        if candidate and index % 2:
            candidate.run_kernel(*candidate_args)
            baseline.run_kernel(*baseline_args)
        else:
            baseline.run_kernel(*baseline_args)
            if candidate:
                candidate.run_kernel(*candidate_args)
    torch.cuda.synchronize()

    comparison = None
    if candidate:
        torch.testing.assert_close(
            outputs[1].float(), outputs[0].float(), atol=1e-2, rtol=1e-2
        )
        difference = (outputs[1].float() - outputs[0].float()).abs()
        comparison = {
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        }

    baseline_samples = []
    candidate_samples = []
    for _ in range(samples):
        if candidate:
            baseline_ms, candidate_ms = time_paired(
                baseline,
                baseline_args,
                candidate,
                candidate_args,
                iterations,
            )
            baseline_samples.append(baseline_ms)
            candidate_samples.append(candidate_ms)
        else:
            baseline_samples.append(
                time_batch(baseline, baseline_args, iterations)
            )

    result = {
        "name": case.name,
        "batch": case.batch,
        "seq_len": case.seq_len,
        "dim": case.dim,
        "selected_blocks": case.selected_blocks,
        "block_size": case.block_size,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "baseline_ms": baseline_samples,
        "baseline_median_ms": statistics.median(baseline_samples),
    }
    if candidate:
        improvements = [
            100.0 * (base - cand) / base
            for base, cand in zip(baseline_samples, candidate_samples)
        ]
        result.update(
            {
                "candidate_ms": candidate_samples,
                "candidate_median_ms": statistics.median(candidate_samples),
                "paired_improvement_percent": improvements,
                "improvement_percent": statistics.median(improvements),
                "comparison": comparison,
            }
        )

    del common_args, outputs, baseline_args, candidate_args
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
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--case-json", type=Path)
    parser.add_argument(
        "--official-family",
        choices=(
            "all",
            "s1_bs16",
            "s1_d64_wide",
            "s2_bs16",
            "s4_s8_bs16",
            "historical_s1_probe",
            "historical_s1_fast_dims",
        ),
        default="all",
    )
    args = parser.parse_args()

    if args.case_json:
        raw_cases = json.loads(args.case_json.read_text())
        selected_cases = [
            BenchmarkCase(
                f"official_{index:03d}",
                item["B"],
                item["SEQ_LEN"],
                item["H"],
                item["HQ"],
                item["D"],
                item["S"],
                item["block_size"],
                int(item["is_causal"]),
                "historical",
            )
            for index, item in enumerate(raw_cases, start=1)
            if args.official_family == "all"
            or (
                args.official_family == "s1_bs16"
                and item["S"] == 1
                and item["block_size"] == 16
            )
            or (
                args.official_family == "s1_d64_wide"
                and item["S"] == 1
                and item["D"] == 64
                and item["block_size"] >= 32
            )
            or (
                args.official_family == "s2_bs16"
                and item["S"] == 2
                and item["block_size"] == 16
            )
            or (
                args.official_family == "s4_s8_bs16"
                and item["S"] in (4, 8)
                and item["block_size"] == 16
            )
            or (
                args.official_family == "historical_s1_probe"
                and (item["D"], item["B"], item["SEQ_LEN"])
                in (
                    (32, 8, 2048),
                    (32, 1, 16384),
                    (32, 2, 16384),
                    (64, 1, 16384),
                    (128, 1, 8192),
                )
            )
            or (
                args.official_family == "historical_s1_fast_dims"
                and item["S"] == 1
                and item["D"] in (32, 128)
            )
        ]
    else:
        selected_cases = parse_cases(args.cases)

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve() if args.candidate else None
    baseline = load_submission(baseline_path, "nsa_c500_baseline")
    candidate = (
        load_submission(candidate_path, "nsa_c500_candidate")
        if candidate_path
        else None
    )

    correctness = {}
    if not args.skip_correctness:
        correctness["baseline"] = correctness_case(baseline)
        if candidate:
            correctness["candidate"] = correctness_case(candidate)

    results = [
        measure_case(
            baseline,
            candidate,
            case,
            args.warmup,
            args.iterations,
            args.samples,
        )
        for case in selected_cases
    ]
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
    if candidate:
        aggregate = []
        for index in range(args.samples):
            base = sum(case["baseline_ms"][index] for case in results)
            cand = sum(case["candidate_ms"][index] for case in results)
            aggregate.append(100.0 * (base - cand) / base)
        report["aggregate"] = {
            "baseline_median_ms": sum(
                case["baseline_median_ms"] for case in results
            ),
            "candidate_median_ms": sum(
                case["candidate_median_ms"] for case in results
            ),
            "paired_improvement_percent": aggregate,
            "improvement_percent": statistics.median(aggregate),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
