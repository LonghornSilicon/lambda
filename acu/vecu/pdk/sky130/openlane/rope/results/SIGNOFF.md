# rope — Sky130A sign-off

RTL→GDSII with LibreLane 3.0.5, ciel sky130A PDK. Flow completed; final GDS in
`rope.gds.gz`, full metrics in `sky130_signoff_metrics.json`, layout render in
`rope_layout.png`. Config: `../config.json` (HEAD_DIM=8, MAX_POS=16).

| Metric | Value |
|---|---|
| Die area | 277632 µm² (0.278 mm²) |
| Std-cell instances | 58855 |
| Core utilization | 40.5 % |
| Clock period (set, loose) | 105 ns (9.5 MHz) |
| fmax (ss_100C_1v60, slack-implied) | ~16.2 MHz (crit path ~61.6 ns) |

## Six-check sign-off

| Check | Result |
|---|---|
| Setup (WNS, all corners) | **0** |
| Hold (WNS, all corners) | **0** |
| Magic DRC | **0** |
| KLayout DRC / route DRC | **0** |
| LVS (netgen) | **clean** — "Circuits match uniquely" |
| Antenna | **0 / 0** |
| Max-cap | **0** |

`design__violations = 0`.

### Noted residual
- `max_slew_violation__count = 524` — ss-corner fp32 register-array fanout slew
  (same known item as `vecu_softmax`); does not affect the five hard checks or
  max-cap, all zero. Absorbed by the loose clock.

Hardened at FP_CORE_UTIL 30 / PL_TARGET_DENSITY_PCT 40 + GRT_ALLOW_CONGESTION: the
fp32+sin/cos-LUT combinational datapath congests global routing at the 45 % util
that closes on GF180, given Sky130's tighter routing tracks. Area is not a
constraint for this micro-block, so the looser floorplan is free.
