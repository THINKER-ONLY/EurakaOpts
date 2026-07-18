import argparse
import hashlib
import importlib.util
import itertools
import json
import linecache
import statistics
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = SCRIPT_DIR / "benchmark_c500.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("c500_autotune_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark helpers from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


PROBE_CASES = {
    case.name: case
    for case in (
        bench.BenchmarkCase("probe_case1", 2048, 8192, 1, 512),
        bench.BenchmarkCase("probe_case2", 7168, 2048, 64, 1024),
        bench.BenchmarkCase("probe_case3", 7168, 2048, 8, 1024),
    )
}
ALL_CASES = {**PROBE_CASES, **bench.CASES}


@dataclass(frozen=True)
class TuneConfig:
    fc1_block_k: int | None = None
    fc1_block_n: int | None = None
    fc2_block_k: int | None = None
    fc2_block_n: int | None = None
    threads: int | None = None
    fc1_threads: int | None = None
    fc2_threads: int | None = None
    num_stages: int | None = None
    fc1_num_stages: int | None = None
    fc2_num_stages: int | None = None
    fc1_policy: str | None = None
    fc2_policy: str | None = None
    fc1_swizzle: str | None = None
    fc2_swizzle: str | None = None


AXIS_TYPES = {
    "fc1_block_k": int,
    "fc1_block_n": int,
    "fc2_block_k": int,
    "fc2_block_n": int,
    "threads": int,
    "fc1_threads": int,
    "fc2_threads": int,
    "num_stages": int,
    "fc1_num_stages": int,
    "fc2_num_stages": int,
    "fc1_policy": str,
    "fc2_policy": str,
    "fc1_swizzle": str,
    "fc2_swizzle": str,
}
POLICIES = {"square", "fullrow", "fullcol"}
SWIZZLES = {"off", "row1", "row5", "row10", "column10"}


def parse_cases(value):
    if value == "probes":
        return list(PROBE_CASES.values())
    if value == "all":
        return list(bench.CASES.values())
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - ALL_CASES.keys())
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return [ALL_CASES[name] for name in names]


def parse_axis(value):
    if "=" not in value:
        raise ValueError(f"axis must use NAME=VALUE1,VALUE2 syntax: {value}")
    name, raw_values = value.split("=", 1)
    name = name.strip()
    if name not in AXIS_TYPES:
        raise ValueError(f"unknown axis {name}; choose from {', '.join(AXIS_TYPES)}")
    values = [item.strip().lower() for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError(f"axis {name} has no values")
    converter = AXIS_TYPES[name]
    converted = [converter(item) for item in values]
    if name.endswith("_policy") and not set(converted) <= POLICIES:
        raise ValueError(f"{name} values must be in {sorted(POLICIES)}")
    if name.endswith("_swizzle") and not set(converted) <= SWIZZLES:
        raise ValueError(f"{name} values must be in {sorted(SWIZZLES)}")
    return name, converted


def expand_configs(axis_values):
    names = list(axis_values)
    for values in itertools.product(*(axis_values[name] for name in names)):
        yield TuneConfig(**dict(zip(names, values)))


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one {label} source pattern")
    return source.replace(old, new)


def replace_swizzles(source, fc1_value, fc2_value):
    marker = "            T.use_swizzle(10)"
    if source.count(marker) != 2:
        raise ValueError("expected exactly two swizzle calls")

    def expression(value):
        if value is None or value == "row10":
            return marker
        if value == "off":
            return "            T.use_swizzle(10, enable=False)"
        if value == "row1":
            return "            T.use_swizzle(1)"
        if value == "row5":
            return "            T.use_swizzle(5)"
        if value == "column10":
            return '            T.use_swizzle(10, order="column")'
        raise ValueError(f"unsupported swizzle: {value}")

    before_fc1, rest = source.split(marker, 1)
    between, after_fc2 = rest.split(marker, 1)
    return (
        before_fc1
        + expression(fc1_value)
        + between
        + expression(fc2_value)
        + after_fc2
    )


def apply_policy(source, prefix, value):
    if value is None or value == "square":
        return source
    enum_name = {"fullrow": "FullRow", "fullcol": "FullCol"}[value]
    if prefix == "fc1":
        calls = (
            "T.gemm(input_shared, gate_shared, gate_local, transpose_B=True)",
            "T.gemm(input_shared, up_shared, up_local, transpose_B=True)",
        )
    else:
        calls = ("T.gemm(up_shared, down_shared, out_local, transpose_B=True)",)
    for call in calls:
        replacement = call[:-1] + f", policy=T.GemmWarpPolicy.{enum_name})"
        source = replace_once(source, call, replacement, f"{prefix} GEMM")
    return source


def render_candidate(base_source, config):
    source = base_source
    if config.fc1_block_k is not None:
        source = replace_once(
            source,
            "    fc1_block_k = 64\n",
            f"    fc1_block_k = {config.fc1_block_k}\n",
            "FC1 BK",
        )
    if config.fc1_block_n is not None:
        source = replace_once(
            source,
            "    fc1_block_n = 128\n",
            f"    fc1_block_n = {config.fc1_block_n}\n",
            "FC1 BN",
        )
    if config.fc2_block_k is not None:
        source = replace_once(
            source,
            "    block_k = 64\n",
            f"    block_k = {config.fc2_block_k}\n",
            "FC2 BK",
        )
    if config.fc2_block_n is not None:
        old = """    fc2_block_n = (
        256
        if (hidden == 7168 and intermediate == 2048)
        or (hidden == 2048 and intermediate == 8192)
        else 128
    )
"""
        source = replace_once(
            source,
            old,
            f"    fc2_block_n = {config.fc2_block_n}\n",
            "FC2 BN",
        )
    if config.fc1_threads is not None or config.fc2_threads is not None:
        base_threads = config.threads if config.threads is not None else 256
        fc1_threads = config.fc1_threads if config.fc1_threads is not None else "threads"
        fc2_threads = config.fc2_threads if config.fc2_threads is not None else "threads"
        source = replace_once(
            source,
            "    threads = 256\n",
            f"    threads = {base_threads}\n"
            f"    fc1_threads = {fc1_threads}\n"
            f"    fc2_threads = {fc2_threads}\n",
            "stage-specific thread count",
        )
        if source.count("threads=threads)") != 2:
            raise ValueError("expected exactly two kernel thread arguments")
        source = source.replace("threads=threads)", "threads=fc1_threads)", 1)
        source = source.replace("threads=threads)", "threads=fc2_threads)", 1)
    elif config.threads is not None:
        source = replace_once(
            source,
            "    threads = 256\n",
            f"    threads = {config.threads}\n",
            "thread count",
        )
    if config.fc1_num_stages is not None or config.fc2_num_stages is not None:
        base_stages = config.num_stages if config.num_stages is not None else 1
        fc1_stages = (
            config.fc1_num_stages
            if config.fc1_num_stages is not None
            else "num_stages"
        )
        fc2_stages = (
            config.fc2_num_stages
            if config.fc2_num_stages is not None
            else "num_stages"
        )
        source = replace_once(
            source,
            "    num_stages = 1\n",
            f"    num_stages = {base_stages}\n"
            f"    fc1_num_stages = {fc1_stages}\n"
            f"    fc2_num_stages = {fc2_stages}\n",
            "stage-specific pipeline stage",
        )
        if source.count("num_stages=num_stages,") != 2:
            raise ValueError("expected exactly two pipeline stage arguments")
        source = source.replace(
            "num_stages=num_stages,", "num_stages=fc1_num_stages,", 1
        )
        source = source.replace(
            "num_stages=num_stages,", "num_stages=fc2_num_stages,", 1
        )
    elif config.num_stages is not None:
        source = replace_once(
            source,
            "    num_stages = 1\n",
            f"    num_stages = {config.num_stages}\n",
            "pipeline stage",
        )
    source = replace_swizzles(source, config.fc1_swizzle, config.fc2_swizzle)
    source = apply_policy(source, "fc1", config.fc1_policy)
    source = apply_policy(source, "fc2", config.fc2_policy)
    return source


def load_source_module(source, index):
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    module_name = f"c500_tune_{index}_{digest}"
    filename = f"<{module_name}>"
    module = types.ModuleType(module_name)
    module.__file__ = filename
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def summarize(config, case_results):
    baseline_total = sum(item["baseline_median_ms"] for item in case_results)
    candidate_total = sum(item["candidate_median_ms"] for item in case_results)
    aggregate = 100.0 * (baseline_total - candidate_total) / baseline_total
    per_case = [item["improvement_percent"] for item in case_results]
    return {
        "status": "pass",
        "config": asdict(config),
        "aggregate_improvement_percent": aggregate,
        "median_case_improvement_percent": statistics.median(per_case),
        "worst_case_improvement_percent": min(per_case),
        "cases": case_results,
    }


def tune(args):
    axis_values = {}
    for raw_axis in args.axis:
        name, values = parse_axis(raw_axis)
        if name in axis_values:
            raise ValueError(f"axis {name} was provided more than once")
        axis_values[name] = values
    configs = list(expand_configs(axis_values))
    if len(configs) > args.max_configs:
        raise ValueError(
            f"search expands to {len(configs)} configs; increase --max-configs "
            "or reduce the axes"
        )

    baseline_path = args.baseline.resolve()
    base_source = baseline_path.read_text()
    baseline = bench.load_submission(baseline_path, "c500_tune_baseline")
    cases = parse_cases(args.cases)
    records = []

    for index, config in enumerate(configs, start=1):
        print(
            json.dumps(
                {"event": "start", "index": index, "total": len(configs), "config": asdict(config)}
            ),
            flush=True,
        )
        try:
            source = render_candidate(base_source, config)
            candidate = load_source_module(source, index)
            case_results = [
                bench.measure_case(
                    baseline,
                    candidate,
                    case,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    samples=args.samples,
                )
                for case in cases
            ]
            record = summarize(config, case_results)
        except Exception as error:
            torch.cuda.empty_cache()
            record = {
                "status": "error",
                "config": asdict(config),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        records.append(record)
        print(json.dumps({"event": "result", **record}), flush=True)

    passing = sorted(
        (record for record in records if record["status"] == "pass"),
        key=lambda item: item["aggregate_improvement_percent"],
        reverse=True,
    )
    summary = {
        "baseline": str(baseline_path),
        "cases": [case.name for case in cases],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": args.samples,
        "records": records,
        "ranking": [
            {
                "config": item["config"],
                "aggregate_improvement_percent": item["aggregate_improvement_percent"],
                "worst_case_improvement_percent": item["worst_case_improvement_percent"],
            }
            for item in passing
        ],
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "summary", **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--axis",
        action="append",
        required=True,
        metavar="NAME=VALUES",
        help="repeatable Cartesian search axis, for example num_stages=0,1,2",
    )
    parser.add_argument(
        "--cases",
        default="probes",
        help="'probes', 'all', or comma-separated probe_case*/oj_case* names",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-configs", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("warmup must be non-negative; iterations and samples positive")
    tune(args)


if __name__ == "__main__":
    main()
