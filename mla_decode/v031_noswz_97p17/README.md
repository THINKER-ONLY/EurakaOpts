# MLA Decode v031: no swizzle on the v030 base (OJ 97.17, best internal mean)

- XPU-OJ contest 6, submission 167557, Accepted, display 97.17, rank 7.
- Internal mean ratio 97.346 (v030: 97.321).
- Per-case (ms): 0.018 / 0.018 / 0.023 / 0.035 / 0.059 / 0.104 / 0.195.
- Only change vs v030: drop T.use_swizzle(10) (64 CTAs fill the device exactly;
  rasterization no longer helps).

Final MLA line: v028 96.83 -> v029 no-max + fused merge 97.0 -> v030 deferred divide
97.17 -> v031 no-swizzle 97.17 (best mean).
