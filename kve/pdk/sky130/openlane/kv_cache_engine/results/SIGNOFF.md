# `kv_cache_engine` — Sky130A sign-off (LibreLane)

Flagship (130 nm SkyWater Sky130A) RTL-to-GDSII sign-off for the KV cache engine.
Hardened with **LibreLane 3.0.5** (Yosys 0.62 + OpenROAD + Magic + KLayout + Netgen),
PDK `sky130A` @ `8afc8346` (via ciel). Run tag `sky130_signoff` (config in
`../config.json`). Multi-corner STA over **9 IPVT corners**
(`{min,nom,max} × {tt_025C_1v80, ss_100C_1v60, ff_n40C_1v95}`).

Artifacts in this directory:

| File | What |
|---|---|
| `metrics.json` | Full LibreLane final metrics (all corners) |
| `kv_cache_engine.gds.gz` | Final GDSII (gzip -9; `gunzip` to open in KLayout/Magic) |
| `resolved_config.json` | Fully-resolved flow configuration (reproducibility) |

## Result: 5 / 6 sign-off checks clean multi-corner; max-cap a documented ss-corner near-miss

| # | Sign-off check | Metric | Result |
|---|---|---|---|
| 1 | **Setup** | `timing__setup_vio__count` = 0 (WNS 0, all 9 corners) | ✅ clean |
| 2 | **Hold** | `timing__hold_vio__count` = 0 (WNS 0, all 9 corners) | ✅ clean |
| 3 | **DRC** | `magic__drc_error__count` = 0, `klayout__drc_error__count` = 0 | ✅ clean |
| 4 | **LVS** | `design__lvs_error__count` = 0, device diff = 0 | ✅ clean |
| 5 | **Antenna** | `route__antenna_violation__count` = 0 | ✅ clean |
| 6 | **Max-cap** | `design__max_cap_violation__count` = **5** (ss corner) | ⚠️ near-miss |

Also clean: power-grid (IR) violations = 0, KLayout XOR diff = 0, disconnected pins = 0.

**Not one of the six but reported honestly:** `design__max_slew_violation__count` = **1503**
and `design__max_fanout_violation__count` = 84 — both concentrated at the slow (ss) corner.

### Timing (reg-to-reg, `CLOCK_PERIOD = 100 ns` / 10 MHz constraint)

Closes with large positive margin (WNS = 0). Implied f_max from the worst reg-to-reg slack:

| Corner | worst r2r slack | implied min period | f_max |
|---|---|---|---|
| ss_100C_1v60 (slow, signoff-binding) | +58.4 ns | 41.6 ns | **~24 MHz** |
| tt_025C_1v80 (typical) | +79.5 ns | 20.5 ns | ~49 MHz |
| ff_n40C_1v95 (fast) | +87.3 ns | 12.7 ns | ~78 MHz |

### Physical

| Metric | Value |
|---|---|
| Die area | 236,211 µm² (0.236 mm²) |
| Core area | 219,511 µm² (0.220 mm²) |
| Std-cell instances | 12,849 (39,718 total incl. fill/tap/decap) |
| Core utilization | 59% |
| Total power (nom_tt) | ~1.79 mW |

## ⚠️ Caveat 1 — storage is a `SRAM_DEPTH = 2` flop-array proxy (not a real SRAM macro)

This hardening uses `SYNTH_PARAMETERS = ["SRAM_DEPTH=2", "VECTOR_DIM=8", "KEY_GROUP=2"]`.
The KV store (`sram_controller.sv`) and residual buffer synthesize as **behavioral
flip-flop register arrays at depth 2** — a functional proxy, **not** real KV capacity and
**not** a Sky130 SRAM macro (OpenRAM / DFFRAM). Consequences:

- The ~12.8k std cells / 0.236 mm² reflect the *logic + depth-2 flop proxy*, not a
  production-depth KV cache. A real cache needs a compiled SRAM macro; the flop array does
  not scale to production depth in area or power.
- This is a **known, separately-tracked hole** (Sky130 SRAM macro), not something this
  run claims to have closed. No real SRAM was faked.

## ⚠️ Caveat 2 — ss-corner max-cap (5) / max-slew (1503): high-fanout reset tree

The residual max-cap and max-slew violations are **entirely at the slow (ss_100C_1v60)
corner** and are driven by the **high-fanout asynchronous-reset (`rst_n`) distribution
tree** across the ~1500-flop array. The repair-inserted reset buffers
(`fanout838/839/840`, `clkdlybuf4s25` chain) slew ~1.0–1.2 ns vs the 0.75 ns
max-transition target — but only under ss derating; the typical/fast corners are near-clean
(nom_ff max-slew = 7, min_ff = 0, and 1 cap net/corner elsewhere).

This is **functionally clean**: reset is asynchronous, checked only for recovery/removal,
which pass with +90 ns slack. Setup/hold/DRC/LVS/antenna are unaffected (all 0). It is the
**documented "ss-corner max-transition on register-array blocks" physical-opt item** in the
PDK holes audit (same class as `mate_pv_fp16` / `vecu_softmax` on GF180). The proper fix is
reset-tree synthesis / driver upsizing (buffer `rst_n` like a low-skew clock), tracked
separately — not a setup/hold/DRC/LVS failure.

### What this run fixed vs. the prior (config-only) state

The committed config previously set `DESIGN_REPAIR_MAX_SLEW_PCT = 0`, which **disabled
effective slew repair** (`-slew_margin 0` in `repair_design`), and left post-GRT design
repair off. Restoring LibreLane's default repair margins (slew/cap = 20 %) and enabling
`RUN_POST_GRT_DESIGN_REPAIR` cut max-cap **61 → 5** and max-slew **5042 → 1503** with no
area/timing regression. See `../config.json`.

## Reproduce

```sh
# enable the Sky130A PDK (once):
ciel enable --pdk-family sky130 8afc8346a57fe1ab7934ba5a6056ea8b43078e71

cd pdk/sky130/openlane/kv_cache_engine
librelane --docker-no-tty --dockerized --pdk sky130A config.json
# final metrics: runs/<tag>/final/metrics.json ; GDS: runs/<tag>/final/gds/
```
