# CI pipeline overview

End-to-end walkthrough of what runs on every push, what's gated, what's
saved, and where the FPGA bitstream step will plug in when the
ZCU102/104 arrives.

The thin caller lives in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
It delegates everything to the
[LonghornSilicon shared `block-ci` reusable workflow](https://github.com/LonghornSilicon/.github/blob/main/.github/workflows/block-ci.yml),
which defines 8 gates that apply to every block in the organization.

Setup for the runner is in [`docs/ci_setup.md`](ci_setup.md).

## Trigger flow

```
   you push to master / open a PR / click "Run workflow"
                          │
                          ▼
              ┌───────────────────────┐
              │  ci.yml (thin caller) │
              └───────────┬───────────┘
                          │ calls shared block-ci.yml@main
                          ▼
              ┌───────────────────────┐
              │  block-ci.yml         │
              │  (8 gates, parallel)  │
              └───────────┬───────────┘
                          │
       ┌──────┬──────┬────┴────┬──────┬──────┐
       ▼      ▼      ▼         ▼      ▼      ▼
   1.func  3.synth 4.formal 5.refmod 6.sky130 ...
   verif  (Yosys) (equiv)  (C++/Py) (OpenLane)
       │      │      │         │      │
       ▼      ▼      ▼         ▼      ▼
   pass/fail across all gates
```

Jobs run concurrently. The whole CI run completes when the slowest one
finishes (currently OpenLane at ~5-10 min for image pull + PnR).

## Block-specific inputs

The thin caller passes these to the shared workflow:

```yaml
block-name: kv_cache_engine
expected-ff-count: 3914          # ChannelQuant default gate proxy (apt-yosys 0.33)
has-reference-model: true
has-paper: false
has-coverage-gate: false        # TB doesn't compile under Verilator yet
has-formal-equivalence: true
extra-rtl-sources: "sram_controller.sv cq_value_path.sv cq_key_path.sv residual_buffer.sv scale_bank.sv amax_unit.sv cq_units_syn.sv"
```

The `extra-rtl-sources` input tells the synthesis, formal equivalence,
and coverage jobs to read the ChannelQuant datapath modules alongside the
top. Single-file blocks (like `precision_controller`) leave this empty.

## Gate-by-gate: what's checked

### 1. RTL functional verification — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install iverilog + numpy | `apt-get` the simulator + Python deps |
| `make testvectors` | Python regenerates hex test vectors from reference model |
| `make sim` | Compile RTL + directed TB (17 cases) |
| `make sim_kpath` / `sim_top` / `sim_cq` | ChannelQuant bit-exact vs golden vectors |
| `make sim_realdata` | Compile RTL + replay TB, run hex vector replay cases |

**Gate**: workflow greps for `ALL TESTS PASSED` or `Tests: N  Pass: N`.

### 2. Line coverage gate — disabled

Set `has-coverage-gate: false` because the kv_cache_engine testbench
uses SystemVerilog constructs that don't compile under Verilator yet.
Will be enabled once the TB is Verilator-clean.

### 3. RTL synthesis (Yosys) — GitHub Ubuntu, ~30 sec

| Step | What it does |
|---|---|
| Install yosys | `apt-get` |
| `read_verilog -sv kv_cache_engine.sv <extra-rtl-sources>` | Reads top + ChannelQuant datapath |
| `synth -flatten -top kv_cache_engine` | Generic gate netlist + cell-count breakdown |
| Awk-extract FF count | Sum every cell line containing `DFF` |

**Gate**: FF count must equal 3914 (the expected-ff-count input, the ChannelQuant
default gate proxy). Catches accidental state additions on every push.

**Records**: `rtl/synth.log` uploaded as artifact (30-day retention).

### 4. Formal equivalence — GitHub Ubuntu, ~1 min

Uses Yosys `equiv_make` + `equiv_induct -seq 5` to prove the
synthesized netlist is bit-exactly equivalent to the original RTL.
Catches synthesis bugs that don't show up in simulation.

Both `gold` (RTL) and `gate` (post-synth) reads include
`sram_controller.sv` via `extra-rtl-sources`.

### 5. Reference model tests — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install numpy + g++ | `pip` + `apt-get` |
| `make test-all` | Build C++ test binary, run 64 C++ tests, run 120 Python tests |

**Gate**: all tests must pass. The Makefile returns non-zero on any failure.

### 6. OpenLane Sky130 sign-off — GitHub Ubuntu, ~5-10 min

| Step | What it does |
|---|---|
| Install `librelane` | `pip install` |
| Compute config dir | Defaults to `pdk/sky130/openlane/kv_cache_engine/` |
| `librelane --docker-no-tty --dockerized --condensed config.json` | Full Sky130 flow |
| Parse `final/metrics.json` | Extract every violation count |

**Gate** — these all must be exactly zero:

| Metric | What it means |
|---|---|
| `timing__setup_vio__count` | Setup-timing violations across all corners |
| `timing__hold_vio__count` | Hold-timing violations across all corners |
| `magic__drc_error__count` | Design-rule violations (geometry/spacing/width) |
| `design__lvs_error__count` | Layout-vs-schematic mismatch |
| `route__antenna_violation__count` | Antenna ratio violations (fab-process hazard) |
| `design__power_grid_violation__count` | IR-drop / power-grid integrity |

**Note on source files**: the committed symlinks in
`pdk/sky130/openlane/kv_cache_engine/src/` point to `rtl/kv_cache_engine.sv` and
`rtl/sram_controller.sv`. `config.json` uses `"VERILOG_FILES": "dir::src/*.sv"`
to pick them up. Only Yosys-compatible files are included — sub-modules
with unpacked array ports will be added once wired into the top-level FSM.

**Records**: GDS, metrics.json, and layout render uploaded as artifacts.

### 7. Paper build — disabled

Set `has-paper: false`. No block-level paper yet.

### 8. Cadence 16FFC sign-off — disabled

Requires TSMC 16FFC PDK + Genus/Innovus licenses on a self-hosted runner.
Will be enabled when the licensed runner is available.

## Where the FPGA bitstream step plugs in

When the ZCU102 or ZCU104 arrives, a Vivado bitstream job can be added
to the shared workflow as gate 9 — or run as a separate caller-side job
if it's KV-cache-engine-specific.

## Where to look at run results

- **Live job logs** during a run:
  https://github.com/LonghornSilicon/lambda/actions
- **Downloadable artifacts** (GDS, PNG, logs): bottom of each
  completed run page, "Artifacts" section
- **Pass/fail status badge**: green check or red X next to each
  commit on the commits page
