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
    kv_ctx: int
    heads: int = 16
    kv_heads: int = 1
    dim: int = 512
    pe_dim: int = 64


CASES = {
    case.name: case
    for case in (
        BenchmarkCase("b1_ctx8192", 1, 8192),
        BenchmarkCase("b1_ctx2048", 1, 2048),
        BenchmarkCase("b1_ctx4096", 1, 4096),
        BenchmarkCase("b1_ctx16384", 1, 16384),
        BenchmarkCase("b1_ctx32768", 1, 32768),
        BenchmarkCase("b2_ctx8192", 2, 8192),
        BenchmarkCase("b2_ctx2048", 2, 2048),
        BenchmarkCase("b2_ctx16384", 2, 16384),
        BenchmarkCase("b2_ctx32768", 2, 32768),
        BenchmarkCase("b4_ctx8192", 4, 8192),
        BenchmarkCase("b4_ctx2048", 4, 2048),
        BenchmarkCase("b4_ctx16384", 4, 16384),
        BenchmarkCase("b4_ctx32768", 4, 32768),
        BenchmarkCase("b8_ctx8192", 8, 8192),
        BenchmarkCase("b8_ctx2048", 8, 2048),
        BenchmarkCase("b8_ctx16384", 8, 16384),
        BenchmarkCase("b8_ctx32768", 8, 32768),
        BenchmarkCase("b16_ctx8192", 16, 8192),
        BenchmarkCase("b16_ctx2048", 16, 2048),
        BenchmarkCase("b16_ctx16384", 16, 16384),
        BenchmarkCase("b16_ctx32768", 16, 32768),
        BenchmarkCase("b32_ctx8192", 32, 8192),
        BenchmarkCase("b32_ctx2048", 32, 2048),
        BenchmarkCase("b32_ctx16384", 32, 16384),
        BenchmarkCase("b32_ctx32768", 32, 32768),
        BenchmarkCase("b1_ctx65536", 1, 65536),
        BenchmarkCase("b2_ctx65536", 2, 65536),
        BenchmarkCase("b4_ctx65536", 4, 65536),
        BenchmarkCase("b8_ctx65536", 8, 65536),
        BenchmarkCase("b16_ctx65536", 16, 65536),
        BenchmarkCase("b32_ctx65536", 32, 65536),
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


def allocate_inputs(case: BenchmarkCase, outputs: int, seed: int = 81394):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (case.batch, case.heads, case.dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    q_pe = torch.randn(
        (case.batch, case.heads, case.pe_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    kv = torch.randn(
        (case.batch, case.kv_ctx, case.kv_heads, case.dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    k_pe = torch.randn(
        (case.batch, case.kv_ctx, case.kv_heads, case.pe_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    output_tensors = [
        torch.empty(
            (case.batch, case.heads, case.dim),
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(outputs)
    ]
    common_args = (
        q,
        q_pe,
        kv,
        k_pe,
        case.batch,
        case.heads,
        case.kv_heads,
        case.kv_ctx,
        case.dim,
        case.pe_dim,
    )
    return common_args, output_tensors


def call_args(common_args, output):
    return (*common_args[:4], output, *common_args[4:])


def reference(q, q_pe, kv, k_pe):
    qf = q.float()
    q_pef = q_pe.float()
    kvf = kv[:, :, 0, :].float()
    k_pef = k_pe[:, :, 0, :].float()
    scale = (q.shape[-1] + q_pe.shape[-1]) ** -0.5
    scores = torch.einsum("bhd,bnd->bhn", qf, kvf)
    scores += torch.einsum("bhp,bnp->bhn", q_pef, k_pef)
    attention = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhn,bnd->bhd", attention, kvf)


def correctness_case(module):
    case = BenchmarkCase("correctness", 1, 512)
    common_args, outputs = allocate_inputs(case, 1, seed=17)
    args = call_args(common_args, outputs[0])
    expected = reference(*common_args[:4])
    module.run_kernel(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0].float(), expected, atol=1e-2, rtol=1e-2)
    first_max_abs = float((outputs[0].float() - expected).abs().max().item())

    generator = torch.Generator(device="cuda")
    generator.manual_seed(23)
    for tensor in common_args[:4]:
        tensor.normal_(generator=generator)
    outputs[0].fill_(float("nan"))
    expected_changed = reference(*common_args[:4])
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
        "kv_ctx": case.kv_ctx,
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
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve() if args.candidate else None
    baseline = load_submission(baseline_path, "mla_c500_baseline")
    candidate = (
        load_submission(candidate_path, "mla_c500_candidate")
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
        for case in parse_cases(args.cases)
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
