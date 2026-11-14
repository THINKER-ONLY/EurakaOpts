# MLA Decode v030: deferred normalization + no-max + fused merge (OJ 97.17)

## Result

- XPU-OJ contest 6, submission 167549, Accepted, display **97.17**, rank **7**.
- Per-case kernel times (ms): 0.018 / 0.018 / 0.024 / 0.036 / 0.059 / 0.104 / 0.194.
- Case displays: 95 / 97 / 97 / 98 / 98 / 98.

## Change vs v029

Defer the per-split normalization into the merge. Without max subtraction,
`L_s = 2^{lse_s}`, so the merge weights collapse to a constant:

    out = sum_s o_s / sum_s L_s

The main kernel no longer divides `acc_o` by `logsum` (8192 fp32 divides per CTA);
the fused merge sums the unnormalized partials and divides once by
`sum_s exp2(lse_s)`. Precision stays at 1.2e-4 max abs error (fp16 partials hold
the unnormalized values; fp32 merge accumulation).

## Stack

v028 (96.83) -> v029 no-max + fused merge (97.0) -> v030 deferred divide (97.17).
Pure TileLang; no torch compute, no import_source/call_extern, no mctlass,
no cross-call result reuse (only kernels and the monotonic barrier counter persist).
