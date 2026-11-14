# MLA Decode v028: Recovered OJ 96.83 source (submission 163109)

## Provenance

- XPU-OJ contest 6 (DeepSeek MLA Decode), submission `163109`, submitted
  2026-10-30T12:18:51Z, status Accepted, display score **96.83**, rank 11.
- Source recovered byte-for-byte from `/api/submission/getSubmissionDetail`
  (`content.code`, 41470 bytes, 1030 lines). Raw detail: `oj_detail.json`.
- This is the true online best. It is **not** derivable from any earlier
  `mla_decode/vNNN` directory: those stop at v027 and never scored 96.x online.

## Compliance audit (2026-11-11)

Pure TileLang. Only `_KERNEL_CACHE` of compiled kernels is retained; every
call recomputes from the current inputs and writes the caller's `output`.
No torch computation, no `@`/`bmm`, no `T.import_source`, no `T.call_extern`,
no mctlass, no cross-call result reuse. Compliant with 赛题一禁用规则说明.

## OJ per-case result (batch=1, heads=16, kv_heads=1, dim=512, pe_dim=64)

| display | score ratio | baseline ms | kernel ms | speedup | kv_ctx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 95 | 95.23% | 0.419 | 0.021 | 19.95x | 2048 |
| 95 | 95.44% | 0.419 | 0.020 | 20.95x | 2048 |
| 96 | 96.81% | 0.789 | 0.026 | 30.35x | 4096 |
| 97 | 97.45% | 1.526 | 0.040 | 38.15x | 8192 |
| 97 | 97.88% | 3.000 | 0.065 | 46.15x | 16384 |
| 98 | 98.12% | 5.958 | 0.114 | 52.26x | 32768 |
| 98 | 98.25% | 11.840 | 0.211 | 56.11x | 65536 |

Mean ratio 97.03%. The short-context 2048 cases are the weakest and are
launch/parallelism bound, so they are the highest-leverage targets for the
next rank step.

## Local C500 (sliced 50%, 32 GiB quota)

`benchmark_c500.py` on this file (single module, no candidate):
b1_ctx2048 27.5 us, b1_ctx8192 69.7 us (OJ full-GPU reports 21/40 us).
Use local numbers only for A/B relative comparisons.
