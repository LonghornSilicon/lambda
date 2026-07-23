# rope — GF180MCU (gf180mcuD) sign-off

RTL→GDSII with LibreLane 3.0.5 (Classic flow), ciel gf180mcuD PDK. Flow completed;
final GDS in `rope.gds.gz`, full metrics in `gf180_signoff_metrics.json`, layout
render in `rope_layout.png`. Config: `../../rope.yaml` (HEAD_DIM=8, MAX_POS=16).

| Metric | Value |
|---|---|
| Die area | 432140 µm² (0.432 mm²) |
| Std-cell instances | 27677 |
| Core utilization | 54.2 % |
| Clock period (set, loose) | 260 ns (3.85 MHz) |
| fmax (ss_125C_4v50, slack-implied) | ~8.5 MHz (crit path ~117 ns) |

## Six-check sign-off

| Check | Result |
|---|---|
| Setup (WNS, all corners) | **0** (worst +142.9 ns @ ss) |
| Hold (WNS, all corners) | **0** (worst +0.40 ns @ ff) |
| Magic DRC | **0** |
| KLayout DRC | **0** |
| LVS (netgen) | **clean** — "Circuits match uniquely" |
| Antenna (violating nets/pins) | **0 / 0** |

`design__violations = 0`.

### Noted residuals (register-array, ss corner)
- `max_slew_violation__count = 1002` (concentrated at the ss corner on the fp32
  register-array fanout nets — the same known item documented for `vecu_softmax`).
- `max_cap_violation__count = 2`.

These are the loose-clock register-array slew/cap residuals; they do not affect the
five hard sign-off checks (setup/hold/DRC/LVS/antenna), which are all zero. The
clock is deliberately loose (single fp32-op micro-sequenced path), so there is
enormous timing margin to absorb them at the target frequency.
