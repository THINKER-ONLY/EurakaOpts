#!/usr/bin/env python3
"""编译检查：对 6 个计分 shape 依次触发 TileLang JIT 编译，记录结果。

本内核是单文件、无外部依赖的 TileLang 实现，编译在首次 run_kernel 调用时
由 tilelang.jit 自动完成（按 shape 特化，每个 shape 一份 kernel）。此脚本
把这件事显式化，便于确认环境可用、以及每个 shape 的编译耗时。

用法：

    PYTHONPATH=/opt/tilelang-metax-v0.1.10 python build_all.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import torch

DEFAULT_KERNEL = Path(__file__).with_name("mla_decode_submission.py")
SCORED_CASES = (2048, 4096, 8192, 16384, 32768, 65536)

BATCH, HEADS, KV_HEADS, DIM, PE_DIM = 1, 16, 1, 512, 64


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("mla_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    kernel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KERNEL
    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        return 2

    print(f"设备      : {torch.cuda.get_device_name(0)}")
    print(f"内核文件  : {kernel_path}")
    print(f"torch     : {torch.__version__}")
    try:
        import tilelang
        print(f"tilelang  : {tilelang.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"tilelang  : 导入失败 {exc}")
        return 2
    print("")
    print("  kv_ctx   编译+首次运行(ms)   状态")
    print("  ------  ------------------  ------")

    module = load_kernel(kernel_path)
    all_ok = True
    for ctx in SCORED_CASES:
        q = torch.randn(BATCH, HEADS, DIM, device="cuda", dtype=torch.float16)
        q_pe = torch.randn(BATCH, HEADS, PE_DIM, device="cuda", dtype=torch.float16)
        kv = torch.randn(BATCH, ctx, KV_HEADS, DIM, device="cuda", dtype=torch.float16)
        k_pe = torch.randn(
            BATCH, ctx, KV_HEADS, PE_DIM, device="cuda", dtype=torch.float16
        )
        out = torch.empty(BATCH, HEADS, DIM, device="cuda", dtype=torch.float16)
        start = time.perf_counter()
        try:
            module.run_kernel(
                q, q_pe, kv, k_pe, out, BATCH, HEADS, KV_HEADS, ctx, DIM, PE_DIM
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            status = "OK"
            ok = not torch.isnan(out).any().item()
            if not ok:
                status = "NaN"
            all_ok = all_ok and ok
        except Exception as exc:  # pragma: no cover
            elapsed = (time.perf_counter() - start) * 1000.0
            status = f"FAIL {type(exc).__name__}: {str(exc)[:60]}"
            all_ok = False
        print(f"  {ctx:6d}  {elapsed:18.1f}  {status}", flush=True)
        del q, q_pe, kv, k_pe, out
        torch.cuda.empty_cache()

    print("")
    print(f"结果：{'ALL COMPILED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
