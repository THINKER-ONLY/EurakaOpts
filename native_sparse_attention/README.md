# Native Sparse Attention optimization history

This directory archives self-contained XPUOJ submissions and local MetaX C500
measurements. Each accepted optimization receives one version directory and one
Git commit. Online evaluation is disabled until explicitly requested.

## Versions

| Version | Change | Local result | Decision |
| --- | --- | ---: | --- |
| `v001_official_baseline` | Official TileLang submission template | 81.301 ms proxy total | baseline |
| `v002_wide_block_threads128` | 128-thread shared-score path for BS32/64 | 50.167 ms, +38.30% | accepted locally |
| `v003_serial_shared_reuse` | Serial block loop, K/V shared alias, direct output, BS64 threads256 | 12.705 ms, +74.66% | accepted locally |

## Local benchmark

```bash
python native_sparse_attention/benchmark_c500.py \
  native_sparse_attention/v001_official_baseline/submission.py
```

The correctness gate evaluates two different input sets with the same shape.
The performance proxy includes short contexts, multiple selected blocks, and a
64K sequence case.
