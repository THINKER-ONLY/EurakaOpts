#!/usr/bin/env python3
"""Build (and cache) the kernel for a module/ctx/split without benchmarking."""
import importlib.util, sys
import torch

path = sys.argv[1]
ctxs = [int(x) for x in sys.argv[2].split(",")]
splits = [int(x) for x in sys.argv[3].split(",")]
spec = importlib.util.spec_from_file_location("mla_mod", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.cuda.init()
scale = (512 + 64) ** -0.5
for ctx in ctxs:
    for ns in splits:
        try:
            k = m.flashattn(1, 16, 1, ctx, 512, 64, 32, 16, ns, scale)
            print("built", path, ctx, ns, "params", len(getattr(k, "params", [])), flush=True)
        except Exception as e:
            print("FAIL", path, ctx, ns, type(e).__name__, str(e)[:300], flush=True)
