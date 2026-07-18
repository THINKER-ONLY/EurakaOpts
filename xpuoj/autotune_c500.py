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
        bench.BenchmarkCase("probe_case1", 2048, 8192, 2, 284),
        bench.BenchmarkCase("probe_case2", 7168, 2048, 4, 568),
        bench.BenchmarkCase("probe_case3", 7168, 2048, 8, 1136),
    )
}
ALL_CASES = {**PROBE_CASES, **bench.CASES}


@dataclass(frozen=True)
class TuneConfig:
    routing_block_m: int | None = None
    fuse_fc1_gate_up_gemm: bool | None = None
    fused_fc1_block_k: int | None = None
    fused_fc1_block_n: int | None = None
    fused_threads: int | None = None
    fused_num_stages: int | None = None
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
    fc1_min_blocks_per_sm: int | None = None
    fc2_min_blocks_per_sm: int | None = None
    aggressive_shared_memory_merge: bool | None = None
    disable_loop_unswitching: bool | None = None
    loop_unswitching_allow_non_trivial_else: bool | None = None
    lower_ldgstg_predicated: bool | None = None
    disable_vectorize_256: bool | None = None


AXIS_TYPES = {
    "routing_block_m": int,
    "fuse_fc1_gate_up_gemm": bool,
    "fused_fc1_block_k": int,
    "fused_fc1_block_n": int,
    "fused_threads": int,
    "fused_num_stages": int,
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
    "fc1_min_blocks_per_sm": int,
    "fc2_min_blocks_per_sm": int,
    "aggressive_shared_memory_merge": bool,
    "disable_loop_unswitching": bool,
    "loop_unswitching_allow_non_trivial_else": bool,
    "lower_ldgstg_predicated": bool,
    "disable_vectorize_256": bool,
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
    if converter is bool:
        invalid = sorted(set(values) - {"false", "true"})
        if invalid:
            raise ValueError(f"{name} boolean values must be true or false")
        converted = [item == "true" for item in values]
    else:
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
    if fc1_value is None and fc2_value is None:
        return source
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


def apply_min_blocks_per_sm(source, fc1_value, fc2_value):
    if fc1_value in (None, 1) and fc2_value in (None, 1):
        return source
    kernel_marker = ") as (bx, by):\n"
    if source.count(kernel_marker) != 2:
        raise ValueError("expected exactly two 2D kernel declarations")

    annotations = []
    for value in (fc1_value, fc2_value):
        if value is None or value == 1:
            annotations.append("")
        elif value > 1:
            annotations.append(f"            T.annotate_min_blocks_per_sm({value})\n")
        else:
            raise ValueError("min_blocks_per_sm must be positive")

    before_fc1, rest = source.split(kernel_marker, 1)
    between, after_fc2 = rest.split(kernel_marker, 1)
    return (
        before_fc1
        + kernel_marker
        + annotations[0]
        + between
        + kernel_marker
        + annotations[1]
        + after_fc2
    )


def apply_pass_config(source, key, value):
    if value is None:
        return source
    marker = "        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,\n"
    if source.count(marker) != 1:
        raise ValueError("expected v022 pass-config marker")
    setting = f"        tilelang.PassConfigKey.{key}: {value},\n"
    return source.replace(marker, marker + setting)


def indent_source_block(source, start_marker, end_marker, label):
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError(f"expected unique {label} guard markers")
    before, rest = source.split(start_marker, 1)
    body, after = rest.split(end_marker, 1)
    block = start_marker + body
    indented = "".join("    " + line if line.strip() else line for line in block.splitlines(True))
    return before + "            if actual_rows > 0:\n" + indented + end_marker + after


def apply_routing_block_m(source, value):
    if value is None or value == 128:
        return source
    if value <= 0 or value % 16:
        raise ValueError("routing_block_m must be a positive multiple of 16")

    source = replace_once(
        source,
        "    routing_block_m = 128\n",
        f"    metadata_block_m = 128\n    routing_block_m = {value}\n",
        "routing BM",
    )
    fc1_metadata = """            expert_id = group_idx_for_bx[bx]
            block_start = bx * routing_block_m
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
"""
    fc1_remapped = """            expert_id = group_idx_for_bx[bx]
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            metadata_subblock = (bx * metadata_block_m - padded_start) // metadata_block_m
            block_start = padded_start + metadata_subblock * routing_block_m
"""
    source = replace_once(source, fc1_metadata, fc1_remapped, "FC1 metadata remap")

    fc2_metadata = """            expert_id = group_idx_for_bx[bx]
            block_start = bx * routing_block_m
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
"""
    fc2_remapped = """            expert_id = group_idx_for_bx[bx]
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            metadata_subblock = (bx * metadata_block_m - padded_start) // metadata_block_m
            block_start = padded_start + metadata_subblock * routing_block_m
"""
    source = replace_once(source, fc2_metadata, fc2_remapped, "FC2 metadata remap")

    source = indent_source_block(
        source,
        "            T.clear(gate_local)\n",
        "\n        with T.Kernel(num_blocks_m, T.ceildiv(hidden, fc2_block_n), threads=threads) as (bx, by):",
        "FC1",
    )
    source = indent_source_block(
        source,
        "            T.clear(out_local)\n",
        "\n    return kernel\n\n\ndef _get_kernel",
        "FC2",
    )
    return source


def apply_fused_fc1_gate_up_gemm(
    source,
    enabled,
    block_k,
    block_n,
    threads,
    num_stages,
):
    if not enabled:
        return source

    factory_start = source.index("@tilelang.jit(")
    factory_end = source.index("\n\ndef _get_kernel")
    base_source = source
    source = source[factory_start:factory_end].replace(
        "def _moe_forward_kernel(",
        "def _moe_forward_kernel_fused(",
        1,
    )
    if block_k is not None:
        source = replace_once(
            source,
            "    fc1_block_k = 64\n",
            f"    fc1_block_k = {block_k}\n",
            "fused FC1 BK",
        )
    if block_n is not None:
        source = replace_once(
            source,
            "    fc1_block_n = 128\n",
            f"    fc1_block_n = {block_n}\n",
            "fused FC1 BN",
        )
    if threads is not None:
        source = replace_once(
            source,
            "    threads = 256\n",
            f"    threads = 256\n    fused_fc1_threads = {threads}\n",
            "fused threads",
        )
        source = source.replace(
            "threads=threads)",
            "threads=fused_fc1_threads)",
            1,
        )
    if num_stages is not None:
        source = replace_once(
            source,
            "    num_stages = 1\n",
            f"    num_stages = 1\n    fused_fc1_num_stages = {num_stages}\n",
            "fused pipeline stages",
        )
        source = source.replace(
            "num_stages=num_stages,",
            "num_stages=fused_fc1_num_stages,",
            1,
        )

    allocations = """            gate_shared = T.alloc_shared((fc1_block_n, fc1_block_k), dtype=dtype)
            up_shared = T.alloc_shared((fc1_block_n, fc1_block_k), dtype=dtype)
            gate_local = T.alloc_fragment((routing_block_m, fc1_block_n), dtype=accum_dtype)
            up_local = T.alloc_fragment((routing_block_m, fc1_block_n), dtype=accum_dtype)
"""
    fused_allocations = """            gate_up_shared = T.alloc_shared((fc1_block_n * 2, fc1_block_k), dtype=dtype)
            gate_up_local = T.alloc_fragment((routing_block_m, fc1_block_n * 2), dtype=accum_dtype)
            gate_local = T.alloc_fragment((routing_block_m, fc1_block_n), dtype=accum_dtype)
"""
    source = replace_once(source, allocations, fused_allocations, "FC1 allocations")
    source = replace_once(
        source,
        "            T.clear(gate_local)\n            T.clear(up_local)\n",
        "            T.clear(gate_up_local)\n",
        "FC1 accumulator clear",
    )

    gemms = """                T.copy(
                    gate_w[
                        expert_id,
                        by * fc1_block_n : (by + 1) * fc1_block_n,
                        k * fc1_block_k : (k + 1) * fc1_block_k,
                    ],
                    gate_shared,
                )
                T.gemm(
                    input_shared,
                    gate_shared,
                    gate_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * fc1_block_n : (by + 1) * fc1_block_n,
                        k * fc1_block_k : (k + 1) * fc1_block_k,
                    ],
                    up_shared,
                )
                T.gemm(
                    input_shared,
                    up_shared,
                    up_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
"""
    fused_gemm = """                T.copy(
                    gate_w[
                        expert_id,
                        by * fc1_block_n : (by + 1) * fc1_block_n,
                        k * fc1_block_k : (k + 1) * fc1_block_k,
                    ],
                    gate_up_shared[0:fc1_block_n, 0:fc1_block_k],
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * fc1_block_n : (by + 1) * fc1_block_n,
                        k * fc1_block_k : (k + 1) * fc1_block_k,
                    ],
                    gate_up_shared[fc1_block_n : fc1_block_n * 2, 0:fc1_block_k],
                )
                T.gemm(
                    input_shared,
                    gate_up_shared,
                    gate_up_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
"""
    source = replace_once(source, gemms, fused_gemm, "FC1 gate/up GEMMs")

    epilogue_loop = "            for i, j in T.Parallel(routing_block_m, fc1_block_n):\n"
    aliases_and_loop = """            T.copy(
                gate_up_local[0:routing_block_m, 0:fc1_block_n],
                gate_local,
            )
            for i, j in T.Parallel(routing_block_m, fc1_block_n):
"""
    source = replace_once(source, epilogue_loop, aliases_and_loop, "FC1 epilogue loop")
    fused_factory = replace_once(
        source,
        "up_local[i, j] * gate_local[i, j]",
        "gate_up_local[i, j + fc1_block_n] * gate_local[i, j]",
        "FC1 up epilogue",
    )
    source = base_source[:factory_end] + "\n\n" + fused_factory + base_source[factory_end:]
    return replace_once(
        source,
        "        kernel = _moe_forward_kernel(*key)\n",
        "        factory = (\n"
        "            _moe_forward_kernel_fused\n"
        "            if key[0] == 7168 and key[1] == 2048 and key[2] in (1, 32)\n"
        "            else _moe_forward_kernel\n"
        "        )\n"
        "        kernel = factory(*key)\n",
        "shape-specialized FC1 factory",
    )


def render_candidate(base_source, config):
    source = base_source
    source = apply_fused_fc1_gate_up_gemm(
        source,
        config.fuse_fc1_gate_up_gemm,
        config.fused_fc1_block_k,
        config.fused_fc1_block_n,
        config.fused_threads,
        config.fused_num_stages,
    )
    source = apply_routing_block_m(source, config.routing_block_m)
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
    source = apply_min_blocks_per_sm(
        source,
        config.fc1_min_blocks_per_sm,
        config.fc2_min_blocks_per_sm,
    )
    source = apply_pass_config(
        source,
        "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE",
        config.aggressive_shared_memory_merge,
    )
    source = apply_pass_config(
        source,
        "TL_DISABLE_LOOP_UNSWITCHING",
        config.disable_loop_unswitching,
    )
    source = apply_pass_config(
        source,
        "TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE",
        config.loop_unswitching_allow_non_trivial_else,
    )
    source = apply_pass_config(
        source,
        "TL_ENABLE_LOWER_LDGSTG_PREDICATED",
        config.lower_ldgstg_predicated,
    )
    source = apply_pass_config(
        source,
        "TL_DISABLE_VECTORIZE_256",
        config.disable_vectorize_256,
    )
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


def generated_source_report(baseline, candidate):
    baseline_kernels = getattr(baseline, "_KERNEL_CACHE", {})
    candidate_kernels = getattr(candidate, "_KERNEL_CACHE", {})
    common_keys = sorted(set(baseline_kernels) & set(candidate_kernels))
    records = []
    for key in common_keys:
        baseline_source = baseline_kernels[key].kernel_source
        candidate_source = candidate_kernels[key].kernel_source
        records.append(
            {
                "shape_key": list(key),
                "baseline_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate_source.encode()).hexdigest(),
                "equal": baseline_source == candidate_source,
            }
        )
    return {
        "all_equal": bool(records) and all(item["equal"] for item in records),
        "kernels": records,
    }


def summarize(config, case_results, source_report):
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
        "generated_source": source_report,
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
            record = summarize(
                config,
                case_results,
                generated_source_report(baseline, candidate),
            )
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
    if args.save_best:
        if not passing:
            raise RuntimeError("cannot save best candidate because no configuration passed")
        best_config = TuneConfig(**passing[0]["config"])
        args.save_best.write_text(render_candidate(base_source, best_config))
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
        default="all",
        help="'probes', 'all', or comma-separated probe_case*/oj_case* names",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-configs", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-best", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("warmup must be non-negative; iterations and samples positive")
    tune(args)


if __name__ == "__main__":
    main()
