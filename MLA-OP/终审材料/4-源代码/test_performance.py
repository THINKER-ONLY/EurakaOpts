#!/usr/bin/env python3
"""性能测试：逐 shape 测量 TileLang MLA Decode 内核的耗时。

对被计分的 6 个 shape 各测一组：先预热，再用 CUDA event 计时若干轮，
取每轮平均值的中位数与最小值。同时给出 KV 读取的有效带宽，以及
按官方基线折算的 display 估计值。

用法：

    PYTHONPATH=/opt/tilelang-metax-v0.1.10 python test_performance.py [--kernel PATH]

注意：本机 C500 是 50% 带宽切片，绝对耗时会比判题机偏高 1～2 微秒；
表中 "display 估计" 只用于横向比较不同版本，最终成绩以 OJ 判题为准。
"""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch

DEFAULT_KERNEL = Path(__file__).with_name("mla_decode_submission.py")

SCORED_CASES = (2048, 4096, 8192, 16384, 32768, 65536)

BATCH = 1
HEADS = 16
KV_HEADS = 1
DIM = 512
PE_DIM = 64

# 赛题官方基线耗时（毫秒），用于折算 display
BASE_MS = {
    2048: 0.419,
    4096: 0.789,
    8192: 1.526,
    16384: 3.0,
    32768: 5.958,
    65536: 11.84,
}


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("mla_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(ctx: int, device: str = "cuda"):
    q = torch.randn(BATCH, HEADS, DIM, device=device, dtype=torch.float16)
    q_pe = torch.randn(BATCH, HEADS, PE_DIM, device=device, dtype=torch.float16)
    kv = torch.randn(BATCH, ctx, KV_HEADS, DIM, device=device, dtype=torch.float16)
    k_pe = torch.randn(BATCH, ctx, KV_HEADS, PE_DIM, device=device, dtype=torch.float16)
    out = torch.empty(BATCH, HEADS, DIM, device=device, dtype=torch.float16)
    return q, q_pe, kv, k_pe, out


def time_case(module, ctx: int, warmup: int, iterations: int, samples: int):
    tensors = make_inputs(ctx)
    q, q_pe, kv, k_pe, out = tensors

    def call():
        module.run_kernel(
            q, q_pe, kv, k_pe, out, BATCH, HEADS, KV_HEADS, ctx, DIM, PE_DIM
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_launch = []
    for _ in range(samples):
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.cuda.synchronize()
        per_launch.append(start.elapsed_time(end) / iterations * 1000.0)

    del tensors, q, q_pe, kv, k_pe, out
    torch.cuda.empty_cache()
    return {
        "median_us": statistics.median(per_launch),
        "min_us": min(per_launch),
        "samples": per_launch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法执行测试。")
        return 2

    print("性能测试：MLA Decode 内核耗时")
    print(f"设备      : {torch.cuda.get_device_name(0)}")
    print(f"内核文件  : {args.kernel}")
    print(
        f"计时方式  : CUDA event，预热 {args.warmup} 次，"
        f"每轮 {args.iterations} 次，共 {args.samples} 轮，取中位数"
    )
    print("")
    print(
        "  kv_ctx   中位数us    最小us   KV读取GB/s   display估计"
    )
    print(
        "  ------  ---------  --------  ----------  -----------"
    )

    module = load_kernel(args.kernel)
    displays = []
    for ctx in SCORED_CASES:
        result = time_case(module, ctx, args.warmup, args.iterations, args.samples)
        med = result["median_us"]
        # KV + K_pe 是主要流量，用它折算有效带宽
        moved = ctx * (DIM + PE_DIM) * 2
        gbps = moved / (med * 1e-6) / 1e9
        base = BASE_MS[ctx]
        display = int(100.0 * base / (base + med / 1000.0))
        displays.append(display)
        spread = ", ".join(f"{x:.2f}" for x in result["samples"])
        print(
            f"  {ctx:6d}  {med:9.2f}  {result['min_us']:8.2f}  "
            f"{gbps:10.1f}  {display:11d}    [{spread}]",
            flush=True,
        )

    print("")
    print(f"display 估计均值：{sum(displays) / len(displays):.2f}（共 {sum(displays)}/6）")
    print("")
    print("说明：本机为 50% 带宽切片，绝对耗时略高于判题机；")
    print("      display 估计仅用于不同实现之间横向比较，最终以 OJ 判题结果为准。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
