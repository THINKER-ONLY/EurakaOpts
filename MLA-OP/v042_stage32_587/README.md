# v042 — register-staged 32-key KV prefetch. submission 168733. 97.83 (587), RANK 1.

Judge (168733): 2048 .015/96, 4096 .019/97, 8192 .028/98, 16384 .037/98,
32768 **.057/99**, 65536 **.098/99** -> mean_ratio 97.9925, display 97.83 (587).
Previous best was 168534 at 97.67 (586, 32768 .064/98).

The only change vs v960 is inside `main_split`'s `pv_specialize` tile loop:
the loop-head synchronous global->shared KV copy is replaced by staging ALL 32
keys in registers. LDGs are issued at the loop head (right after the previous
tile's barrier, pinned before the S_shared barrier); the STS drains them back
into KV_shared after PV has consumed the current tile. It slots into the
existing L181 barrier, so no new barrier and no extra shared memory.
Registers 238, 0 B stack; dyn_smem unchanged at 56576.

Three things that made it work:
1. `mtregisters` ceiling is 254 and mxcc demotes a local array to the stack if it
   is accessed ONLY through `*(uint4*)` casts. Fix: cast-WRITE the staging
   buffer, element-wise READ it. (cast-only => 148 B stack => 101 us.)
2. TileLang reuses ONE `KVT` uint4 temp per chunk, WAR-serializing every LDG.
   Use distinct local buffers (kv_s0..kv_s7) so each LDG gets its own dest reg.
3. A `T.sync_threads()` forces outstanding LDGs to complete before it passes, so
   the prefetch cover is only the code between the issue point and the NEXT
   barrier (~0.8 us PV half vs ~1.4 us round trip). That is why the 50.5 us
   "delete the copy entirely" ceiling is unreachable in this barrier structure.

Also fixed a pre-existing deadlock in BASE: the fused-merge software barrier
computes `fuse_target = (fuse_old // num_split + 1) * num_split`, which spins
forever if one process launches two different split counts (64 then 104).
`_STATE` is now a per-shape dict, so each shape's counter only ever grows by its
own num_split.
