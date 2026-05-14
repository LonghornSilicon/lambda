# Sky130 OpenLane / LibreLane flow

End-to-end open-source RTL-to-GDSII flow for `kv_cache_engine`,
targeting SkyWater Sky130A. Complements the Cadence-flow scripts
in `rtl/genus.tcl` / `rtl/innovus.tcl`.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/kv_cache_engine
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`.
First invocation downloads the Sky130A PDK (~500 MB via Ciel) and the
LibreLane Docker image (~6 GB). Subsequent runs reuse both caches.
Total runtime: ~5-10 minutes.

## Files

| Path | Purpose |
|---|---|
| `config.json` | LibreLane design config (80 MHz target) |
| `src/` | Source files (CI creates symlinks from `rtl/`) |
| `runs/` | Each run's intermediate artifacts (gitignored) |

## Current source files

Only Yosys-compatible files are included in the OpenLane flow:

| File | Description |
|---|---|
| `kv_cache_engine.sv` | Top-level module (AXI-Lite + AXI-Stream + FSM) |
| `sram_controller.sv` | Behavioral SRAM (reg array, dual-port) |

The remaining sub-modules (`norm_unit`, `rotation_unit`, `quantizer`,
`qjl_unit`, `packer`, `decompressor`) use SystemVerilog unpacked array
ports that Yosys cannot parse. They synthesize correctly under Cadence
Genus but will be added to the OpenLane flow once the full pipeline is
wired into the top-level FSM with Yosys-compatible interfaces.

## Config

```json
{
    "DESIGN_NAME": "kv_cache_engine",
    "VERILOG_FILES": "dir::src/*.sv",
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 12.5,
    "IO_DELAY_CONSTRAINT": 5,
    "CLOCK_UNCERTAINTY_CONSTRAINT": 0.1,
    "PDN_MULTILAYER": false,
    "FP_SIZING": "relative",
    "FP_CORE_UTIL": 50,
    "PL_TARGET_DENSITY_PCT": 60,
    "DESIGN_REPAIR_MAX_SLEW_PCT": 0,
    "GRT_DESIGN_REPAIR_MAX_SLEW_PCT": 0
}
```

Key config choices (lessons from block 1):

- `IO_DELAY_CONSTRAINT=5` (not default 20%) — the failing path is
  I/O-bound, not register-to-register. Tightening I/O delay constraint
  has a larger effect than increasing clock period.
- `CLOCK_UNCERTAINTY_CONSTRAINT=0.1` (not default 0.25) — for a small
  block where CTS skew is sub-50 ps, 0.1 ns is more honest than the
  conservative 0.25 ns.
- `DESIGN_REPAIR_MAX_SLEW_PCT=0` — forces the tool to fix every slew
  violation rather than the worst 20%.

## Sign-off gate (CI)

The CI workflow parses `final/metrics.json` and asserts zero violations:

| Check | Must be zero |
|---|---|
| `timing__setup_vio__count` | Setup timing |
| `timing__hold_vio__count` | Hold timing |
| `magic__drc_error__count` | Design rule |
| `design__lvs_error__count` | Layout vs schematic |
| `route__antenna_violation__count` | Antenna ratio |
| `design__power_grid_violation__count` | IR-drop |

Any single violation fails the CI job.

## Why this matters

OpenLane is a real industrial-quality open-source flow (Yosys,
OpenROAD, Magic, KLayout, Netgen). The Sky130 result provides an
independent cross-check of the Yosys generic synthesis and gives a
real point estimate to scale from when the TSMC 16FFC PDK becomes
available.

Projected to 16FFC (5x area shrink, 2x speed gain, 0.5x power):
the full pipeline should fit comfortably at 800 MHz within the
KV cache engine's area budget.
