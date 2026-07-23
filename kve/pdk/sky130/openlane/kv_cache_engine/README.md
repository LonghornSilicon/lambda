# Sky130 LibreLane / OpenLane flow — `kv_cache_engine`

End-to-end open-source RTL-to-GDSII sign-off for `kv_cache_engine`, targeting
SkyWater **Sky130A** (the flagship / closest-to-real-chip PDK). Complements the
Cadence-flow scripts in `rtl/genus.tcl` / `rtl/innovus.tcl`.

**A committed sign-off lives in [`results/`](results/) — see
[`results/SIGNOFF.md`](results/SIGNOFF.md) for the full report.**

## Run it

Requires Docker (~25 GB free disk). The `ghcr.io/librelane/librelane:3.0.5` image
carries Yosys 0.62, OpenROAD, Magic, KLayout and Netgen.

```sh
# enable the Sky130A PDK once (ciel manages PDK_ROOT):
ciel enable --pdk-family sky130 8afc8346a57fe1ab7934ba5a6056ea8b43078e71

cd pdk/sky130/openlane/kv_cache_engine
librelane --docker-no-tty --dockerized --pdk sky130A config.json
```

First invocation downloads the Sky130A PDK (~500 MB via ciel) and the LibreLane
image (~6 GB). Total runtime: ~8–10 minutes (the depth-2 flop-array KVE is
~13k std cells, so detailed routing + DRC dominate).

## Files

| Path | Purpose |
|---|---|
| `config.json` | LibreLane design config (Sky130A, 100 ns clock, repair margins) |
| `src/` | Yosys-compatible source (CI symlinks these from `rtl/`) |
| `results/` | **Committed** sign-off artifacts (metrics JSON + final GDS + report) |
| `runs/` | Each run's intermediate artifacts (gitignored) |

### Source file set

The KVE has behavioral `real` / `$fscanf` reference views that abort Yosys; the flow
uses the **synthesizable `*_syn` set** (confirmed synthesizable — 12,849 std cells, no
`real`, no latches). `config.json` globs `dir::src/*.sv`, where `src/` symlinks:

| File | Role |
|---|---|
| `kv_cache_engine.sv` | Top: AXI-Lite + AXI-Stream + FSM |
| `cq_value_path.sv` / `cq_key_path.sv` | Per-token INT4 value path + grouped per-channel INT4 key path |
| `cq_units_syn.sv` | Synthesizable ChannelQuant quant/dequant/scale units |
| `amax_unit.sv`, `scale_bank.sv` | Absmax + per-group scale storage |
| `residual_buffer.sv` | Residual accumulation buffer (flop proxy) |
| `sram_controller.sv` | Behavioral KV store — **flop register array**, dual-port (see caveat) |

## Sign-off result (committed)

LibreLane 3.0.5 · Sky130A `8afc8346` · multi-corner STA over **9 IPVT corners**
(`{min,nom,max} × {tt_025C_1v80, ss_100C_1v60, ff_n40C_1v95}`).

| Check | Result |
|---|---|
| Setup (`timing__setup_vio__count`) | ✅ 0 (WNS 0, all 9 corners) |
| Hold (`timing__hold_vio__count`) | ✅ 0 (WNS 0, all 9 corners) |
| DRC (Magic + KLayout) | ✅ 0 / 0 |
| LVS (Netgen, incl. device diff) | ✅ 0 |
| Antenna (`route__antenna_violation__count`) | ✅ 0 |
| **Max-cap** (`design__max_cap_violation__count`) | ⚠️ **5** — ss corner only (near-miss) |

Also clean: power-grid IR = 0, XOR diff = 0. Reported honestly (not among the 6):
max-slew = 1503, max-fanout = 84, both **ss-corner only**.

**Area / timing:** die **0.236 mm²**, core 0.220 mm² (59 % util), ~1.79 mW.
Timing closes with huge margin at the 100 ns constraint; implied f_max
**~24 MHz (ss)** / ~49 MHz (tt) / ~78 MHz (ff).

### ⚠️ Caveat 1 — flop-array storage proxy

`SRAM_DEPTH = 2`: the KV store and residual buffer synthesize as **behavioral flip-flop
register arrays at depth 2** — a functional proxy, **not** real KV capacity and **not** a
Sky130 SRAM macro (OpenRAM / DFFRAM). The real Sky130 SRAM macro is a separately-tracked
hole. No real SRAM is faked here.

### ⚠️ Caveat 2 — ss-corner max-cap / max-slew (high-fanout reset tree)

The residual max-cap (5) and max-slew (1503) are **entirely at the slow ss corner** and
come from the **high-fanout asynchronous-reset (`rst_n`) tree** across ~1500 flops (repair
buffers slew ~1.0–1.2 ns vs the 0.75 ns target under ss derating). Functionally clean —
reset is async (recovery/removal slack +90 ns); setup/hold/DRC/LVS/antenna all 0. This is
the documented "ss-corner max-transition on register-array blocks" physical-opt item; the
fix is reset-tree synthesis / driver upsizing, tracked separately.

## Config notes

- `CLOCK_PERIOD = 100` ns — the KVE runs at decode cadence; huge setup margin remains.
- `IO_DELAY_CONSTRAINT = 5`, `CLOCK_UNCERTAINTY_CONSTRAINT = 0.1` — the failing path is
  I/O-bound on a small block; tightening these beats padding the period (block-1 lesson).
- `DESIGN_REPAIR_MAX_SLEW_PCT = 20`, `GRT_DESIGN_REPAIR_MAX_SLEW_PCT = 20`,
  `RUN_POST_GRT_DESIGN_REPAIR = true` — **restored from a prior `…=0` that disabled slew
  repair** (`-slew_margin 0`). This cut max-cap 61 → 5 and max-slew 5042 → 1503 with no
  area/timing regression.

## Sign-off gate (CI)

CI parses `results/metrics.json` (or `runs/<tag>/final/metrics.json`) and asserts the
setup/hold/DRC/LVS/antenna/power-grid counts are zero. The ss-corner max-cap/max-slew
residual is a tracked physical-opt item (reset-tree buffering), documented above.

## Why this matters

OpenLane/LibreLane is a real industrial-quality open-source flow. The Sky130 result is an
independent cross-check of the Yosys generic synthesis and a real 130 nm point estimate to
scale from toward the Lambda (16 nm) target.
