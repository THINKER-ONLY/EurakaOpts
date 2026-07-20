# NSA v025: Cached S1 Block Start

## Objective

Remove repeated block-index loads and block-size multiplications from the
dominant single-block NSA kernels without changing the multi-block paths.

## Change

- cache the selected block id once per CTA in a local int32 value;
- cache the corresponding token start separately for causal tests and masks;
- keep the unscaled block id for K/V slices so TileLang retains its compact
  safe-memory guards;
- require the selected block to fit completely in the sequence, matching the
  challenge's full-block indices and `seq_len` invalid-index sentinel;
- enable the specialization for all D128 S1/BS16 and S1/BS32 kernels, D32
  S1/BS16 at 2K or longer, and D64 S1/BS16 at 4K;
- retain v024's algorithms, thread counts, GEMM policies, softmax expressions,
  shared-memory paths, and all S2/S4/S8 kernels unchanged.

## Correctness

- All 109 official candidate outputs are bitwise identical to v024.
- Independent FP32-reference checks cover D128 S1 with BS16 and BS32 using two
  input seeds; maximum absolute error is `0.00097609`.
- The accepted tolerances remain `atol=1e-2, rtol=1e-2`.
- Only compiled kernels are cached; every invocation reads current inputs and
  writes the complete caller-provided output.

## Local C500 Result

Environment: full MetaX C500, MACA 3.7.1.5, TileLang
`0.1.11+maca.git56b76a2b`. The validation used twelve warmups, forty batched
calls per module/output combination, sixteen samples, both output addresses,
and a four-phase Latin rotation.

| Scope | Cases | v024 total | v025 total | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Specialized S1 kernels | 41 | 1.68328 ms | 1.52975 ms | **+9.34%** |
| Full official set | 109 | 2.82356 ms | 2.67114 ms | **+5.61%** |

The full-set phase-balanced samples improved by `+5.6446%`, `+5.5699%`,
`+5.7104%`, and `+5.2903%`. The specialized-family samples improved by
`+8.9799%`, `+9.8756%`, `+8.6567%`, and `+9.6940%`.

Decision: **accepted as the local baseline**. No online evaluation was run.

The archived submission SHA-256 is
`50ab3c35d3cd1136d89ddad788008aac44efeb7c7c2b7abb3c311f5fe712ec9d`.
