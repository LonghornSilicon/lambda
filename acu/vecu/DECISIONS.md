# DECISIONS — VecU (acu/vecu)

Append-only. *what · why · date*. (Migrated from the parked `acu/DECISIONS.md` at import 2026-07-22.)

- **Decode softmax slice only** (single-row online softmax + 64-entry exp LUT); full programmable
  VecU + RoPE/RMSNorm are later · the decode datapath doesn't need them · 2026-07-21.
- **exp LUT carries ~2% error vs exact softmax** · inherent to a 64-entry linear-interp LUT over
  [-16,0]; cosim tolerances are set FROM it, not tighter · 2026-07-21.
- **Pipeline the exp/rescale/accumulate chain to one fp32-op/cycle** · a long combinational fp path
  won't close at the GF180 ss corner; decode is latency-tolerant so extra cycles are free · 2026-07.
