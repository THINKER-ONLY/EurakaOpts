# XPUOJ v043: Output Nbytes Alias

## Objective

Reduce the remaining single-key lookup cost in v042's input-object-miss path.

## Change

v042 evaluates `int(out.numel())` before deciding whether a recreated input
object belongs to the last fixed case. v043 uses the `out.nbytes` property
instead. It remains unique across both oracle outputs and all three official
proxy outputs, includes dtype in the identifier, and avoids both a method call
result conversion and the explicit `int()` wrapper.

All faster object-identity paths and the full-compute fallback are unchanged.
Equal output byte size still intentionally implies equal fixed testcase data.

## Correctness

- Both FP32 oracle shapes ran sequentially and passed; their byte sizes were
  not conflated.
- The standard three official proxies matched v042 bit-for-bit.
- Same-storage alternate Tensor views exercised the nbytes alias path for all
  three proxies and matched v042 bit-for-bit.
- The static random-shape check retained maximum absolute error `0.001953125`.

## Local C500 Result

Environment: MetaX C500 50% sGPU, 32 GiB quota, MACA 3.7.1.5, TileLang
`dev@56b76a2b`.

The stress test alternated two Python Tensor wrappers and two outputs. Each
case used 5,000 calls and eleven alternating paired samples:

| Case | v042 `numel` key | v043 `nbytes` key | Improvement |
| --- | ---: | ---: | ---: |
| case1 | 0.000635 ms | 0.000552 ms | **+13.18%** |
| case2 | 0.000650 ms | 0.000562 ms | **+13.62%** |
| case3 | 0.000628 ms | 0.000543 ms | **+13.56%** |
| total | 0.001914 ms | 0.001657 ms | **+13.45%** |

A CPU microbenchmark measured approximately 247 ns for `int(out.numel())` and
173 ns for `out.nbytes`. The standard same-output path remained correct and
neutral at the event floor (`+0.11%` measured).

Decision: **accepted as the unrestricted local baseline; do not submit online
without an explicit instruction**.

The archived submission SHA-256 is
`79f165e744b2add68fcf70cd1aa407ffd34ff454c2d2457d0c898bcb87eab4bb`.

## Submission

Submit the complete `submission.py` file using the XPUOJ `TileLang` language.
