# DECISIONS — MatE (acu/mate)

Append-only. *what · why · date*. (Migrated from the parked `acu/DECISIONS.md` at import 2026-07-22.)

- **P·V accumulator = INT32, not INT24** · P·V reduces over the TOKEN axis, so width scales with
  context; INT24 overflows past ~520 flat tokens, INT32 covers ~133k · 2026-07-20.
- **8×8 grid = 64 PEs = 128 GOPS** · the reference-model default was a stale 16×16/256; corrected to
  match arch.yml/STATUS · 2026-07-21.
- **FP16 P·V escape path exists** · attention P·V routes per-tile INT8/FP16 via the precision
  controller (`max·N > 10·Σ`); weight/FFN GEMMs stay INT8×INT4 · 2026-07-18.
- **Shared synth/EDA harness lives in `mate/rtl/`** · `Makefile`/`synth.ys`/`sweep_synth.py`/`eda/*.tcl`
  build all ACU tiles (not just MatE); parked with the engine hub rather than duplicated · 2026-07-22.
- **FP16 verified against a sequential-fp32 golden, not numpy `@`** · BLAS pairwise summation reorders
  the MACs, so bit-exactness to numpy is impossible; tolerance is `rel_err < 5e-3` · 2026-07.
