# DeepSeek MLA Decode optimization history

This directory archives self-contained XPUOJ submissions and local MetaX C500
measurements. Each accepted optimization receives one version directory and one
Git commit. Online evaluation is disabled until explicitly requested.

## Versions

| Version | Change | Local result | Decision |
| --- | --- | ---: | --- |
| `v001_official_baseline` | Official TileLang submission template | 31.764 ms proxy total | baseline |

## Local benchmark

```bash
python mla_decode/benchmark_c500.py \
  mla_decode/v001_official_baseline/submission.py
```

The correctness gate evaluates two different input sets with the same shape.
The performance proxy covers short and 64K contexts at several batch sizes.
