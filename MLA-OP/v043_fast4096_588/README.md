# v043 — first 588. submission 168830.

Routes ctx=4096/8192 into the long-context fast path (`main_split_pref`: 3 barriers/tile,
next-tile prefetch split across two barrier intervals, 512 threads / 8-warp PV), plus two
bank-conflict fixes gated to `seqlen_kv <= 8192`.
  4096: 19.70 -> 17.22us (-2.48)   -> judge 0.016 => display 98  (the 588th point)
  8192: 29.03 -> 24.24us (-4.79)   -> judge 0.023 => display 98
Bit-exact at all six contexts. Superseded by v044.
