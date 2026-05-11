# Sky130 OpenLane / LibreLane flow

End-to-end open-source RTL → GDSII flow for `precision_controller`,
targeting SkyWater Sky130A. Complements the Cadence-flow scripts in
`rtl/genus.tcl` / `rtl/innovus.tcl`.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/precision_controller
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`. First
invocation downloads the Sky130A PDK (~500 MB via Ciel) and the LibreLane
Docker image (~6 GB). Subsequent runs reuse both caches.

Total runtime for `precision_controller` on a 4-core machine: ~3 minutes.

## Files

| Path | Purpose |
|---|---|
| `config.json` | LibreLane design config (clock period, die size, util) |
| `src/precision_controller.v` | Copy of `rtl/precision_controller.sv` for the OpenLane flow |
| `results/precision_controller.gds` | Final GDSII (1.3 MB, Sky130A) |
| `results/precision_controller.png` | Rendered layout view |
| `results/sky130_100MHz_metrics.json` | Full LibreLane metrics dump |
| `runs/` | Each run's intermediate artifacts (gitignored) |

## Results — first pass (100 MHz target)

Run `RUN_2026-05-11_23-17-21`, `CLOCK_PERIOD = 10.0 ns`.

| Metric | Value |
|---|---|
| Logic cells (excl. fill) | 591 |
| Flip-flops | 30 |
| Stdcell area | ~4,000 µm² |
| Die area | 150 × 150 µm² (22.5% util) |
| Wirelength | 7,187 µm |
| Total power (TT corner) | 431 µW |
| Setup WNS (TT) | +2.89 ns ✓ |
| Setup WNS (FF) | +4.76 ns ✓ |
| Setup WNS (SS) | −1.58 ns ✗ |
| Hold WS (all corners) | +0.12 ns ✓ |
| DRC / LVS / antenna / IR-drop violations | 0 / 0 / 0 / 0 |

The SS corner fails at 100 MHz — critical path at SS is ~11.6 ns. A
12.5 ns period (80 MHz) closes SS with ~1 ns margin. See subsequent
runs in `results/` for the closed-timing result.

## Why this matters for the paper

OpenLane is a real industrial-quality open-source flow (Yosys, OpenROAD,
Magic, KLayout, Netgen). The Sky130 result is not a projection — it's a
clean RTL-to-GDS pass at 130 nm with DRC/LVS-clean output, providing an
independent cross-check of the ASAP7 Yosys numbers in `analysis/` and
strengthening the case that the design scales cleanly through the Cadence
16FFC flow when that PDK is available.
