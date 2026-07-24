# DECISIONS — Precision Controller (acu/precision_controller)

Append-only. *what · why · date*. (Established with the ACU import 2026-07-22.)

- **Gate = pre-softmax ratio test `max(|S|)·N > 10·Σ|S|`, not entropy** · entropy needs softmax →
  exponentials → expensive in hardware; the raw-score ratio is entropy-equivalent for peakedness and
  is one shift + a ×10 + one comparator, decided in 1 cycle · (from the APA research) 2026.
- **Threshold = 10** · perfectly separates uniform vs peaked across 19,488 real tiles (gap
  `1.5 < ratio < 3.5` is essentially empty) · 2026.
- **FF budget = 30, closed-form `2·SCORE_WIDTH + log₂N + 2`** · asserted in CI (`rtl-synthesis`) so a
  refactor that changes it is caught · 2026.
- **OpenLane hardens the `.v` (Verilog) source** (`pdk/sky130/openlane/precision_controller/src/
  precision_controller.v`), while `rtl/precision_controller.sv` is the SystemVerilog DUT · keep both
  in sync · 2026-07-22.
