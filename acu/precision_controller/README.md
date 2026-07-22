# Precision Controller (ACU)

The per-tile INT8-vs-FP16 gate. A streaming, 1-cycle pre-softmax ratio test decides whether an
attention tile can run on the cheap INT8 P·V path or must fall back to FP16:

```
   max(|S|) · N  >  10 · Σ|S|     →  FP16 (peaked tile)   else INT8
```

No division, no transcendentals — a left shift, a ×10 (two shifts + add), one comparator. ~30
flip-flops. ~79% of tiles end up on INT8 with no accuracy loss vs FlashAttention-2.

## Layout
- `rtl/precision_controller.sv` + `rtl/tb/tb_precision_controller.sv` (253/253 cases).
- `sw/reference_model/precision_controller_ref.{py,cpp,hpp}` + `test_*` (bit-exact vs RTL).
- `pdk/sky130/openlane/precision_controller/` — Sky130A sign-off. Note the synthesizable RTL source
  under `.../src/precision_controller.v` is the Verilog variant used by the OpenLane flow.
- `pdk/asap7/orfs/asap7/precision_controller/` — predictive 7nm bracket.
- `pdk/gf180/librelane/precision_controller.yaml` — GF180 tape-out config.
- `docs/isa/precision_controller_isa.{md,tex,pdf}` — the ISA spec (pc-isa-0.1).

## Signed off (Sky130A)
80 MHz (SS/100°C/1.60V), 30 FFs, 3,438 µm² stdcell, DRC/LVS/antenna/IR 0/0/0/0. 16nm projections
(Sky130-scaled vs ASAP7-derived) differ ~5× and do not reconcile — treat as a range.

## Known gotchas
- **Threshold-of-10 is robust** but the separation gap is `1.5 < ratio < 3.5` (nearly empty over the
  19,488-tile corpus) — don't tighten it without re-checking the bimodality figure.
- **FF count has a closed form** `2·SCORE_WIDTH + log₂N + 2` (= 30) — CI asserts it; keep it in sync.
- **The `.v` (Verilog) source, not the `.sv`, is what the OpenLane flow reads** — edit both or regenerate.

See `DECISIONS.md` and `AGENTS.md`.
