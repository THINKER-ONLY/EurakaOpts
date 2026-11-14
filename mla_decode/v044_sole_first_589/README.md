# v043 — SOLE FIRST PLACE. submission 168830, 98.00 (588), rank 1.

Judge (168830): 2048 .013/96, 4096 **.016/98**, 8192 .023/98, 16384 .034/98,
32768 .053/99, 65536 .090/99 -> mean_ratio 98.2503, display 98.00 (588/6).
Second place 97.83. Progression this session: 586 -> 587 (v964) -> 588 (v992).

`submission.py` = /tmp/wf_4096b/v990_4096b.py.

## The winning change: route ctx=4096/8192 to the long-context fast path
`main_split_pref` (3 barriers/tile, next-tile prefetch split across two barrier
intervals, 512 threads / 8-warp PV) was gated at `seqlen_kv >= 16384`. Lowering the
gate to `>= 4096` and extending `use_512` likewise gave:
  4096: 19.70 -> 17.22us (-2.48, 12.6%)
  8192: 29.03 -> 24.24us (-4.79, 16.5%)
plus two bank-conflict fixes gated to `seqlen_kv <= 8192`:
  - O_shared drain transpose was exactly 16-way conflicted (row stride 512 halves =
    256 words = 0 mod 32, so all 16 rows of a warp hit one bank). Fixed with a
    rotational index skew `(oj + (oi%16)*4) % dim`  -> -0.71us. An XOR form
    scalarized the global store to 2B and gave nothing.
  - S_shared publish skewed 4 halves per row (`(sj + si*4)%32`, 8-way ->
    conflict-free) -> -0.47us.
Both are bit-exact at all six contexts (max|v-b| = 0.0).

## Negative results worth keeping
- 512 threads alone on the OLD `main_split` body is NOT viable: its register
  prefetch is hardcoded for 256 threads (`kpe_s` row = `tx_s//8`), so at 512
  threads it indexes rows 0..63 and faults. The 512-thread config and the
  prefetch-split body are inseparable.
- Storing acc_o straight into a thread-major permuted `output_partial` (no
  O_shared) regressed +2.1us: a warp then writes 16 scattered 32B sectors.
  The shared staging transpose is the right design; only its bank pattern was waste.
- Two tiles cannot share one QK/PV pipeline more cheaply on this shared budget
  (64-key tile needs 64KB; dynamic shared is 56576B of 65536).

## Also in this directory
v993 (same 589 score, worse long-ctx margins) was dropped during the final tidy; v044/submission.py is the best artifact.
(-1.78us at 2048) + volatile-load arrival spin. Local: 2048 13.64, 4096 17.10,
8192 24.15. Bit-exact. Gives a second shot at 2048's 0.012 tier if re-submitted.

## UPDATE — v993 = 589 (submission 168837), still rank 1
Judge (168837): 2048 **.012/97**, 4096 .016/98, 8192 .023/98, 16384 .033/98,
32768 .052/99, 65536 .090/99 -> mean_ratio 98.3217, **display 98.17 (589/6)**.
Second place 97.83. Lead 0.34.

v993 = v992 + two more fixes, both bit-exact:
- `O_shared` padded to `[16, dim+8]` on the ctx=2048 path (v992's skew is gated to the
  pref path only), slicing `O_shared[0:vh, 0:dim]` at both the drain and the store.
  2048: 15.42 -> 13.64 local.
- arrival spin polls with `T.atomic_load(State[0], memory_order="acquire")` (a plain
  volatile load on MACA) instead of `atomic_add(...,0)`; removes a contended RMW on the
  one counter all 64/104 CTAs touch, and kills the straggler tail.
That moved 2048's judge reading from .013/96 to .012/97, which is the 589th point.

Practical ceiling from here: 2048 97->98 needs k<=.00855 (now .012), 8192 98->99 needs
.0154 (now .023), 16384 98->99 needs .0303 (now .033; measured floor ~33us, bandwidth-bound),
4096 98->99 needs .00797 (now .016). 32768/65536 are already 99 and 100 is impossible.

## FINAL — rank 1 at 98.17 (589). 590 was chased and closed.
Best artifacts, both 589: `submission_FINAL_589.py` (= v994_skew, judge 168869) and
`v1015_qksplit` (judge 168938). v994 is preferred (better long-ctx margins).

Attempts to win the 590th point (ctx=16384 needs kernel_ms <= 0.030, floor observed 0.031):
  v966 pref-split ............. 16384 0.033 (later 0.031 via v994)
  v994 skew + spin gate ....... 16384 0.031   <- best
  v1013 prologue reorder + hand-written QK + dead-tile skip ... 0.032 (regressed)
  v1015 QK K-dim split across two warp groups (-1.35us local) . 0.031 (no judge change)
The v1015 result is the decisive diagnostic: it DID transfer negatively at 32768
(0.050 -> 0.052) and 65536 (0.087 -> 0.090), so the judge is sensitive to compute changes --
yet at 16384 halving the per-tile MMA work is invisible. **The judge's ctx=16384 is
memory-bound, not compute-bound** (18.9 MB in 31us = ~610 GB/s effective). That also
explains why every local memory-side win (the host is a 50%-bandwidth slice) failed to transfer.
Remaining gaps to any further point: 2048 needs <= 0.00855 (launch floor alone is 2.87us),
8192 needs <= 0.0154 (now 0.023), 4096 needs <= 0.00797 (now 0.016), 32768/65536 already 99.
