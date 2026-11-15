# v039 — split 边界按 tile 对齐：97.67

原来的切分公式是 `kv_start = seqlen_kv * bz // num_split`，每个 CTA 的起点落在任意 key 上，104 条并发流的分段访问把拷贝效率拖掉一半。

改为先算 tile 下标再乘回 block_n：

```python
tiles_total = seqlen_kv // block_n
kv_start = ((tiles_total * bz) // num_split) * block_n
kv_end   = ((tiles_total * (bz + 1)) // num_split) * block_n
```

改完后 104 split 全面优于 64 split。65536 从 0.157 ms 降到 0.111 ms。

OJ 提交号 168534，display 97.67。
