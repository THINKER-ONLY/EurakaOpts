#!/usr/bin/env python3
"""功能测试：逐 shape 校验 TileLang MLA Decode 内核的数值正确性。

对被计分的 6 个 shape（batch=1，kv_ctx 见 SCORED_CASES）分别执行：

1. 固定种子生成输入，跑内核，与 fp32 参考实现比对绝对误差。
2. 换一组输入再跑一次，确认输出随输入变化（排除"只算一次"或"忽略输入"的假实现）。
3. 显式检查 NaN。

参考实现的语义与赛题一致：

    score = q @ kv^T + q_pe @ k_pe^T
    out   = softmax(score * (dim + pe_dim) ** -0.5) @ kv

用法：

    PYTHONPATH=/opt/tilelang-metax-v0.1.10 python test_functional.py [--kernel PATH]

默认加载同目录的 mla_decode_submission.py。全部通过时退出码为 0。
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

DEFAULT_KERNEL = Path(__file__).with_name("mla_decode_submission.py")

# 判题机计分的 6 个 shape：batch=1, heads=16, kv_heads=1, dim=512, pe_dim=64
SCORED_CASES = (2048, 4096, 8192, 16384, 32768, 65536)

BATCH = 1
HEADS = 16
KV_HEADS = 1
DIM = 512
PE_DIM = 64

# 容差。fp16 输入与 fp32 参考之间，实测最大绝对误差约 1.2e-4；
# 取 2e-3 留一个数量级余量，仍远小于错误实现会产生量级（常见 1e-1 以上）。
ATOL = 2e-3


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("mla_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(ctx: int, seed: int, device: str = "cuda"):
    # 直接用 torch.manual_seed 生成 CUDA 张量。MACA 这一版的 torch 在
    # "CPU generator + .half().to(cuda)" 组合下会触发段错误，不要改回去。
    torch.manual_seed(seed)
    q = torch.randn(BATCH, HEADS, DIM, device=device, dtype=torch.float16)
    q_pe = torch.randn(BATCH, HEADS, PE_DIM, device=device, dtype=torch.float16)
    kv = torch.randn(BATCH, ctx, KV_HEADS, DIM, device=device, dtype=torch.float16)
    k_pe = torch.randn(BATCH, ctx, KV_HEADS, PE_DIM, device=device, dtype=torch.float16)
    out = torch.empty((BATCH, HEADS, DIM), dtype=torch.float16, device=device)
    return q, q_pe, kv, k_pe, out


def reference(q, q_pe, kv, k_pe):
    """fp32 参考实现，不在原地修改输入。"""
    qf = q.float()
    q_pef = q_pe.float()
    kvf = kv[:, :, 0, :].float()
    k_pef = k_pe[:, :, 0, :].float()
    scale = (q.shape[-1] + q_pe.shape[-1]) ** -0.5
    scores = torch.einsum("bhd,bnd->bhn", qf, kvf)
    scores += torch.einsum("bhp,bnp->bhn", q_pef, k_pef)
    weights = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhn,bnd->bhd", weights, kvf)


def run_once(module, tensors, ctx: int):
    q, q_pe, kv, k_pe, out = tensors
    module.run_kernel(
        q, q_pe, kv, k_pe, out, BATCH, HEADS, KV_HEADS, ctx, DIM, PE_DIM
    )
    torch.cuda.synchronize()
    return out


def histogram(values, vmax, bins=4, width=30):
    """把绝对误差画成文本直方图，便于看分布形状。"""
    if vmax <= 0:
        vmax = 1e-12
    edges = [vmax * i / bins for i in range(bins + 1)]
    counts = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & (values < hi)
        if i == bins - 1:
            mask = (values >= lo) & (values <= hi)
        counts.append(int(mask.sum()))
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * max(0, round(c / peak * width))
        lines.append(f"[{edges[i]:.1e},{edges[i + 1]:.1e}) {c:5d} {bar}")
    return lines


def check_case(module, ctx: int, device: str = "cuda") -> bool:
    ok = True

    # 第一次：固定种子
    tensors = make_inputs(ctx, seed=ctx, device=device)
    expected = reference(*tensors[:4])
    got = run_once(module, tensors, ctx)
    diff = (got.float() - expected).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    has_nan = bool(torch.isnan(got).any())

    flat = diff.flatten()
    total = flat.numel()
    qs = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
    p50, p90, p99 = (float(x) for x in qs)
    rms = float(torch.sqrt((flat ** 2).mean()))
    over_1e4 = int((flat > 1e-4).sum())
    over_1e3 = int((flat > 1e-3).sum())

    finite = not has_nan and max_abs == max_abs  # max_abs != max_abs 捕获全 NaN
    passed = finite and max_abs < ATOL
    ok = ok and passed
    print(
        f"  ctx={ctx:6d}  max={max_abs:.3e}  mean={mean_abs:.3e}  "
        f"p50={p50:.2e}  p90={p90:.2e}  p99={p99:.2e}  rms={rms:.3e}",
        flush=True,
    )
    print(
        f"           >1e-4: {over_1e4}/{total}   >1e-3: {over_1e3}/{total}   "
        f"nan={'yes' if has_nan else 'no'}   {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    for line in histogram(flat, max_abs):
        print(f"           {line}", flush=True)

    # 第二次：换输入，确认输出确实随输入改变
    tensors = make_inputs(ctx, seed=ctx + 100_000, device=device)
    expected = reference(*tensors[:4])
    other = run_once(module, tensors, ctx).float()
    diff = (other - expected).abs()
    max_abs_2 = float(diff.max())
    passed_2 = (not torch.isnan(other).any()) and max_abs_2 < ATOL
    # 两次的输出若相同，说明内核忽略了输入
    differs = float((other - got.float()).abs().max()) > 1e-3
    ok = ok and passed_2 and differs
    print(
        f"          second seed: max_abs={max_abs_2:.3e}  "
        f"output_changed={differs}  {'PASS' if passed_2 and differs else 'FAIL'}",
        flush=True,
    )

    del tensors, expected, got, other
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法执行测试。")
        return 2

    print("功能测试：MLA Decode 内核数值正确性")
    print(f"设备      : {torch.cuda.get_device_name(0)}")
    print(f"内核文件  : {args.kernel}")
    print(
        f"shape     : batch={BATCH} heads={HEADS} kv_heads={KV_HEADS} "
        f"dim={DIM} pe_dim={PE_DIM}"
    )
    print(f"容差      : max_abs < {ATOL:.0e}")
    print("")

    module = load_kernel(args.kernel)

    all_ok = True
    for ctx in SCORED_CASES:
        all_ok = check_case(module, ctx) and all_ok

    print("")
    print(f"结果：{'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
