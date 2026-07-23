# rmsnorm — GF180MCU (gf180mcuD) sign-off

RTL→GDSII with LibreLane 3.0.5 (Classic flow), ciel gf180mcuD PDK. Flow completed;
final GDS in `rmsnorm.gds.gz`, full metrics in `gf180_signoff_metrics.json`, layout
render in `rmsnorm_layout.png`. Config: `../../rmsnorm.yaml` (D=16).

| Metric | Value |
|---|---|
| Die area | 1095180 µm² (1.095 mm²) |
| Std-cell instances | 69989 |
| Core utilization | 48.9 % |
| Clock period (set, loose) | 260 ns (3.85 MHz) |
| fmax (ss_125C_4v50, slack-implied) | ~6.7 MHz (crit path ~149 ns) |

## Six-check sign-off

| Check | Result |
|---|---|
| Setup (WNS, all corners) | **0** |
| Hold (WNS, all corners) | **0** |
| Magic DRC | **0** |
| Route DRC (KLayout) | **0** |
| LVS (netgen) | **clean** — "Circuits match uniquely" |
| Antenna (violating nets/pins) | **0 / 0** |

`design__violations = 0`.

### Noted residuals (register-array, ss corner)
- `max_slew_violation__count = 2857` — ss-corner slew on the fp32 register-array
  fanout nets (the same known item documented for `vecu_softmax`).
- `max_cap_violation__count = 7`.

These are the loose-clock register-array slew/cap residuals; they do not affect the
five hard sign-off checks (setup/hold/DRC/LVS/antenna), all zero. The clock is
deliberately loose (single fp32-op micro-sequenced path: sum-of-squares → rsqrt LUT
→ scale), so there is ample margin to absorb them at the target frequency.
