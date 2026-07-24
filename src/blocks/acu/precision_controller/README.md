# Precision Controller (ACU)

The per-tile **INT8-vs-FP16 gate**. A streaming, 1-cycle pre-softmax ratio test decides whether an
attention tile can run on the cheap INT8 P·V path or must fall back to FP16:

```
   max(|S|) · N  >  10 · Σ|S|     →  FP16 (peaked tile)   else INT8
```

No division, no transcendentals — a left shift, a ×10 (two shifts + add), one comparator. 30
flip-flops. ~79% of tiles end up on INT8 with no accuracy loss vs FlashAttention-2. Sits under
`src/blocks/acu/`; mirrors to `lambda-precision-controller`.

## Layout
Canonical block template (`sw/ rtl/ pdk/ docs/ research/`, per SOP §5.1):
- `rtl/precision_controller.sv` + `rtl/tb/tb_precision_controller.sv` (self-checking, RTL-vs-ref parity).
- `sw/reference_model/precision_controller_ref.{py,cpp,hpp}` + `test_*` (bit-exact golden).
- `pdk/sky130/openlane/precision_controller/` — Sky130A sign-off. The OpenLane flow reads the
  **Verilog** source `.../src/precision_controller.v`, not the `.sv` DUT.
- `pdk/asap7/orfs/asap7/precision_controller/` — ASAP7 route-clean bracket (ORFS).
- `pdk/gf180/librelane/precision_controller.yaml` — GF180 config (declared, not run).
- `docs/isa/precision_controller_isa.{md,tex,pdf}` — the ISA spec (pc-isa).

## Status
Per-PDK, traced to `pdk/**/results/*metrics.json` and `docs/PROGRESS.md` (sign-off defs: SOP §5.2).

| PDK | Flow | Status | Die | Freq | Notes |
|---|---|---|---|---|---|
| sky130 | OpenLane | **signed-off** | 8,642 µm² (0.0086 mm²) | 80 MHz | DRC/LVS/antenna/setup/hold = 0; 30 FFs; 3,438 µm² stdcell |
| asap7 | ORFS | **route-clean** | — | 1.18 GHz | route-clean only — no Magic-DRC / no LVS step |
| gf180 | LibreLane | **config-only** | — | — | declared, not run |

Sky130A is signed off at the SS/100 °C/1.60 V corner with the tightest setup slack in the ACU
(~0.14 ns at 80 MHz). ASAP7 is a predictive 7nm bracket, **not** full sign-off. 16nm projections
(Sky130-scaled vs ASAP7-derived) differ ~5× and do not reconcile — treat as a range.

## Known gotchas
- **`.v`, not `.sv`, is what OpenLane hardens.** `src/precision_controller.v` and `rtl/*.sv` must be
  kept in sync by hand (see `DECISIONS.md`); edit both or regenerate.
- **Silent-drift risk with the standalone repo.** `precision_controller.sv` is byte-identical to
  Chaithu's out-of-band `attention-compute-unit` repo (one-time manual import 2026-07-22, **no
  auto-sync in either direction**). Per SOP §7, future changes originate in the monorepo; re-import
  only deliberately and log it in `DECISIONS.md`.
- **Sky130 SS corner carries 33 max-slew violations.** `design__max_slew_violation__count = 33`
  (all at `nom_ss_100C_1v60`) is outside the DRC/LVS/antenna/timing headline that credits sign-off.
  SOP §5.2 wants such corner near-misses disclosed in a `results/SIGNOFF.md` — none exists here yet.
- **Threshold-of-10 is robust** but the separation gap is `1.5 < ratio < 3.5` (nearly empty over the
  19,488-tile corpus) — don't tighten it without re-checking the bimodality figure.
- **FF count has a closed form** `2·SCORE_WIDTH + log₂N + 2` (= 30) — CI asserts it; keep it in sync.

## Branch model
`main` is a clean scaffold — **no `.sv`/`.v` RTL**. The RTL lives on the `rev0` revision branch
(contributors PR into `rev0`; a lead blesses → merges to `main`). To view/work on RTL:
`git checkout rev0`. Full model: `docs/REVISION_SYNC_SOP.md` §6a.

See `DECISIONS.md` and `AGENTS.md`.
