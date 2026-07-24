# DECISIONS — VecU (acu/vecu)

Append-only. *what · why · date*. (Migrated from the parked `acu/DECISIONS.md` at import 2026-07-22.)

- **Decode softmax slice only** (single-row online softmax + 64-entry exp LUT); full programmable
  VecU + RoPE/RMSNorm are later · the decode datapath doesn't need them · 2026-07-21.
- **exp LUT carries ~2% error vs exact softmax** · inherent to a 64-entry linear-interp LUT over
  [-16,0]; cosim tolerances are set FROM it, not tighter · 2026-07-21.
- **Pipeline the exp/rescale/accumulate chain to one fp32-op/cycle** · a long combinational fp path
  won't close at the GF180 ss corner; decode is latency-tolerant so extra cycles are free · 2026-07.
- **RoPE + RMSNorm built as real RTL** (`rope.sv`, `rmsnorm.sv`) · needed for the chip-top raw-Q/K +
  hidden-state path (cosim tiles are pre-RoPE'd/normed); part of the December full-chip and more
  GF180/Sky130 area numbers to port to 16nm · 2026-07-23.
- **RoPE/RMSNorm reuse the softmax fp16/fp32 IEEE datapath, fp32-internal + one fp16 rounding on
  emit** · same primitives as mate_pv_fp16/vecu_softmax; no SystemVerilog `real` (non-synthesizable);
  keeps the block bit-exact to a pure-Python LUT golden · 2026-07-23.
- **RoPE sin/cos + RMSNorm rsqrt are LUTs (codebook-ROM style), interpolated in fp32** · LUT error
  ~7e-4 rel (< 5e-3 bar); models the actual HW, so RTL is bit-exact to the golden rather than to
  exact math · 2026-07-23.
- **RoPE/RMSNorm micro-sequenced (one fp32-op/cycle), same as vecu_softmax** · closes GF180 ss with
  normal resizing; decode is latency-tolerant · 2026-07-23.
- **RoPE/RMSNorm harden configs staged but P&R run deferred** · RTL+TB+synth (Yosys latch-clean) are
  the priority + the valuable artifact; a full librelane run is heavy — noted follow-up · 2026-07-23.
- **RoPE/RMSNorm hardened to GDSII on GF180 + Sky130 (LibreLane 3.0.5)** · needed real area/timing for
  16nm porting + the December chip; loose clock (single fp32-op micro-sequenced path). GF180 closes
  with the 5 hard checks (setup/hold/DRC/LVS/antenna) = 0; the ss-corner max-slew/max-cap register-array
  residual is noted (same known item as vecu_softmax), not gating · 2026-07-23.
- **Sky130 rope/rmsnorm needed density relief (util 30/28, density 40/38 + GRT_ALLOW_CONGESTION)** ·
  the fp32+LUT combinational datapath congests global routing at the 45 % util that closed vecu_softmax,
  on Sky130's tighter tracks; area is not the constraint for these micro-blocks so loosening is free ·
  2026-07-23.
