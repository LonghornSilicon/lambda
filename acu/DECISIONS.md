<!-- ACU-level decisions, PARKED here while acu/ is a placeholder (import held 2026-07-22).
     When MatE/VecU are imported, migrate these into acu/mate/DECISIONS.md and acu/vecu/DECISIONS.md.
     Seeded from docs/prototypes/DECISIONS.seed.md. Format: what · why · date. Append-only. -->

# DECISIONS — ACU (parked until import; see acu/README.md)

## MatE (→ acu/mate)
- **P·V accumulator = INT32, not INT24** · P·V reduces over the TOKEN axis, so width scales with
  context; INT24 overflows past ~520 flat tokens, INT32 covers ~133k · 2026-07-20.
- **8×8 grid = 64 PEs = 128 GOPS** · the reference-model default was a stale 16×16/256; corrected to
  match arch.yml/STATUS · 2026-07-21.
- **FP16 P·V escape path exists** · attention P·V routes per-tile INT8/FP16 via the precision
  controller (`max·N > 10·Σ`); weight/FFN GEMMs stay INT8×INT4 · 2026-07-18.

## VecU (→ acu/vecu)
- **Decode softmax slice only** (single-row online softmax + 64-entry exp LUT); full programmable
  VecU + RoPE/RMSNorm are later · the decode datapath doesn't need them · 2026-07-21.
- **exp LUT carries ~2% error vs exact softmax** · inherent to a 64-entry linear-interp LUT over
  [-16,0]; cosim tolerances are set FROM it, not tighter · 2026-07-21.
