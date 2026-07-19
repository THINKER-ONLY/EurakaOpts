# DeepSeek MLA Decode optimization history

This directory archives self-contained XPUOJ submissions and local MetaX C500
measurements. Each accepted optimization receives one version directory and one
Git commit. Online evaluation is disabled until explicitly requested.

## Versions

| Version | Change | Local result | Decision |
| --- | --- | ---: | --- |
| `v001_official_baseline` | Official TileLang submission template | 31.764 ms proxy total | baseline |
| `v002_split8_output_quarters` | 8-way context split and four 128-column output partitions | 15.706 ms, +50.75% | accepted locally |
| `v003_batch_split_policy` | Select split 16/8/4 from batch size | 14.969 ms, +4.87% | accepted locally |

## Local benchmark

```bash
python mla_decode/benchmark_c500.py \
  mla_decode/v001_official_baseline/submission.py
```

The correctness gate evaluates two different input sets with the same shape.
The performance proxy covers short and 64K contexts at several batch sizes.
