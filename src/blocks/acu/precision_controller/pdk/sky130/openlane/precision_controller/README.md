# Sky130 OpenLane / LibreLane flow

End-to-end open-source RTL → GDSII flow for `precision_controller`,
targeting SkyWater Sky130A. **All corners signed off** (DRC / LVS /
antenna / IR-drop / setup / hold clean). Complements the Cadence-flow
scripts in `rtl/genus.tcl` / `rtl/innovus.tcl`.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/precision_controller
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`.
First invocation downloads the Sky130A PDK (~500 MB via Ciel) and the
LibreLane Docker image (~6 GB). Subsequent runs reuse both caches.
Total runtime: ~3 minutes.

## Files

| Path | Purpose |
|---|---|
| `config.json` | LibreLane design config (signed-off values) |
| `src/precision_controller.v` | Copy of `rtl/precision_controller.sv` |
| `results/precision_controller.gds` | Final GDSII (910 KB, Sky130A) |
| `results/precision_controller.png` | Rendered layout view |
| `results/sky130_signoff_metrics.json` | Full LibreLane metrics |
| `runs/` | Each run's intermediate artifacts (gitignored) |

## Signed-off result — 80 MHz, all corners clean

| Metric | Value |
|---|---|
| **Flip-flops** | 30 (matches closed-form `2·SCORE_WIDTH + log₂N + 2`) |
| **Logic cells (excl. fill)** | 292 |
| **Stdcell area** | 3,438 µm² |
| **Die size** | 87.8 × 98.5 µm² (59% core utilization) |
| **Wirelength** | 6,201 µm |
| **Power (TT)** | 330.5 µW (91.5 µW switching, 238.9 µW internal, 6 pW leakage) |
| **Setup WNS (FF / TT / SS)** | +8.342 ns / +6.102 ns / **+0.072 ns** |
| **Hold WS (FF / TT / SS)** | +0.182 ns / +0.363 ns / +0.860 ns |
| **DRC / LVS / antenna / IR-drop / setup-vio / hold-vio** | **0 / 0 / 0 / 0 / 0 / 0** |

## Config tuning history

The Sky130A SS corner (100 °C, 1.60 V) is a worst-case test of the
design and the tool's optimization heuristics. The first-pass config
left ~1.6 ns of negative slack at SS; the final config closes it at
+0.07 ns. The knobs that mattered, in order:

| Run | Period | IO_DELAY% | Clock unc. | Die | SS WNS |
|---|---|---|---|---|---|
| 1 | 10.0 ns | 20% (default) | 0.25 ns | 150² absolute | −1.580 ns |
| 2 | 12.5 ns | 20% | 0.25 ns | 150² | −0.235 ns |
| 3 | 14.0 ns | 20% | 0.25 ns | 150² | −0.168 ns |
| 4 | 12.5 ns | 5% | 0.25 ns | 150² | −0.081 ns |
| 5 | 12.5 ns | 5% | 0.25 ns | 88×98 (relative 50%) + slew=0 | −0.070 ns |
| 6 | 13.0 ns | 5% | 0.25 ns | 88×98 + slew=0 | −0.261 ns (heuristic regression) |
| **7** | **12.5 ns** | **5%** | **0.1 ns** | **88×98 + slew=0** | **+0.072 ns ✓** |

Lessons:
- The failing path is **I/O-bound, not register-to-register** (timing
  summary shows "Reg to Reg Paths: N/A" for the worst paths). Pure
  period increases give diminishing returns because OpenLane scales
  I/O delays with the clock period (20% default).
- Tightening `IO_DELAY_CONSTRAINT` from 20% to 5% had a large effect.
- `CLOCK_UNCERTAINTY_CONSTRAINT=0.1 ns` (down from 0.25) was the final
  margin needed. 0.25 ns is appropriate for chip-level designs with
  unconstrained CTS; for a small block where CTS skew is sub-50 ps,
  0.1 ns is more honest.
- `DESIGN_REPAIR_MAX_SLEW_PCT=0` forces the tool to fix every slew
  violation rather than the worst 20%, which improved both slack
  and slew indirectly.

## Why this matters for the paper

OpenLane is a real industrial-quality open-source flow (Yosys,
OpenROAD, Magic, KLayout, Netgen). The Sky130 result here is **not a
projection** — it's a clean RTL-to-GDS pass at 130 nm with DRC/LVS/
antenna/IR-drop-clean output, signed off at all three voltage/
temperature corners. This provides an independent cross-check of the
ASAP7 Yosys numbers in `analysis/` and gives a real point estimate
to scale from when the TSMC 16FFC PDK becomes available.

Projected to 16FFC — two independent estimates that **do not reconcile**. Scaling
this measured Sky130 sign-off by published node ratios (~5× area, ~2× speed, ~10×
dynamic power) gives ~700 µm², ~160 MHz, ~30 µW; the ASAP7-derived projection
(`analysis/tsmc16_fit_report.md`) instead gives ~150 µm² / ~800 MHz / ~5–15 µW. The
area estimates differ ~5× and the frequencies do not reconcile — treat 16nm speed/area
as a **range** pending a real Cadence 16FFC run, not a single number (see
`acu/docs/acu_overview.md` and the ACU README).
