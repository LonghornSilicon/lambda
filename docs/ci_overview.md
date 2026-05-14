# CI pipeline overview

End-to-end walkthrough of what runs on every push, what's gated, what's
saved, and where the FPGA bitstream step will plug in when the
ZCU102/104 arrives.

The pipeline lives in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
Setup for the runner is in [`docs/ci_setup.md`](ci_setup.md).

## Trigger flow

```
   you push to master / open a PR / click "Run workflow"
                          │
                          ▼
              ┌───────────────────────┐
              │  CI workflow fires    │
              │  (ci.yml)             │
              └───────────┬───────────┘
                          │ four jobs start in parallel
       ┌─────────────┬────┴────┬─────────────────┐
       ▼             ▼         ▼                  ▼
   rtl-func-     rtl-synth  openlane-sky130   reference-
   verification             (GitHub Ubuntu)   model
   (Ubuntu)      (Ubuntu)                     (Ubuntu)
       │             │         │                  │
       ▼             ▼         ▼                  ▼
   pass/fail   pass/fail   pass/fail         pass/fail
                          │
              ┌───────────┴───────────┐
              │ green check on commit │
              │ artifacts available   │
              │ for 30-day retention  │
              └───────────────────────┘
```

Jobs run concurrently — they don't wait on each other. The whole CI run
completes when the slowest one finishes (currently OpenLane at ~5-10 min
for image pull + synthesis + PnR).

## Job-by-job: what's checked vs. what's recorded

### 1. `rtl-functional-verification` — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install iverilog + numpy | `apt-get` the simulator + Python deps |
| `make testvectors` | Python regenerates hex test vectors from reference model |
| `make sim` | Compile RTL + directed TB, run 14 cases against integer reference |
| `make sim_realdata` | Compile RTL + replay TB, run hex vector replay cases |

**Gate**: every case must pass. Workflow greps for `ALL TESTS PASSED`
in both testbench outputs. One failure → job red.

**Records**: nothing as artifact (logs visible in the job page). Could
add `*.vcd` waveforms here if useful for debugging.

### 2. `rtl-synthesis` — GitHub Ubuntu, ~30 sec

| Step | What it does |
|---|---|
| Install yosys | `apt-get` |
| `yosys -s synth.ys` | RTL → generic gate netlist + NAND mapping + cell-count breakdown |
| Awk-extract FF/cell count | Sum every cell line containing `DFF`, extract total cell count |

**Gate**: synthesis must complete without error. FF and cell counts are
reported in the GitHub step summary for tracking.

**Records**: `rtl/synth.log` uploaded as the `yosys-synth-log`
artifact (30-day retention).

### 3. `openlane-sky130` — GitHub Ubuntu, ~5-10 min

| Step | What it does |
|---|---|
| Install `librelane` + verify `docker` | Smoke check the tooling |
| Create source symlinks | Links Yosys-compatible SV files into `openlane/kv_cache_engine/src/` |
| `librelane --dockerized config.json` | Full Sky130 flow: Yosys synth → floorplan → place → CTS → route → STA → DRC → LVS → antenna → IR-drop → GDS |
| Parse `final/metrics.json` | Extract every violation count |

**Gate** — these all must be exactly zero:

| Metric | What it means |
|---|---|
| `timing__setup_vio__count` | Setup-timing violations across all corners |
| `timing__hold_vio__count` | Hold-timing violations across all corners |
| `magic__drc_error__count` | Design-rule violations (geometry/spacing/width) |
| `design__lvs_error__count` | Layout-vs-schematic mismatch (layout ≢ netlist) |
| `route__antenna_violation__count` | Antenna ratio violations (fab-process hazard) |
| `design__power_grid_violation__count` | IR-drop / power-grid integrity |

A single violation in any of these and the Python assertion in the
workflow exits non-zero, failing the job. This is the **real sign-off
gate** — the same checks you'd run before sending GDS to a fab.

**Records**: uploaded as the `sky130-signoff` artifact (30-day retention):

- `kv_cache_engine.gds` — final layout (tape-out-ready modulo Sky130 vs 16FFC)
- `metrics.json` — every PPA number (area, timing per corner, power per category, wirelength)
- `*.png` — rendered layout image

**Note on source files**: only `kv_cache_engine.sv` and
`sram_controller.sv` are included in the OpenLane flow. The remaining
sub-modules (`rotation_unit`, `quantizer`, `qjl_unit`, `packer`,
`decompressor`) use SystemVerilog unpacked array ports that Yosys
cannot parse. They will be added once the full compression pipeline
is wired into the top-level FSM.

### 4. `reference-model` — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install numpy + g++ | `pip` + `apt-get` |
| `make test-all` | Build C++ test binary, run 64 C++ tests, run 120 Python tests |

**Gate**: all tests must pass. The Makefile returns non-zero on any failure.

**Records**: nothing as artifact (logs visible in the job page).

## Where the FPGA bitstream step plugs in

When the ZCU102 or ZCU104 arrives, a fifth job will sit parallel to the
others (gated on `rtl-functional-verification` passing):

```yaml
  vivado-zcu102-bitstream:
    name: Vivado ZCU102 bitstream
    runs-on: [self-hosted, linux, x64, vivado]
    needs: rtl-functional-verification
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4

      - name: Vivado synth + impl + bitstream
        working-directory: fpga/zcu102
        run: |
          vivado -mode batch -source impl.tcl \
                 -log vivado.log -journal vivado.jou

      - name: Assert utilization within budget
        run: |
          # Parse impl_1/post_route_util.rpt
          # Assert: LUTs < budget, FFs < budget, BRAM = 0, DSP = 0
          ...

      - name: Upload bitstream + reports
        uses: actions/upload-artifact@v4
        with:
          name: zcu102-bitstream
          path: |
            fpga/zcu102/impl_1/*.bit
            fpga/zcu102/impl_1/*.hwh
            fpga/zcu102/impl_1/*.rpt
          retention-days: 30
```

## What's not yet checked (future gates worth adding)

| Future gate | What it would catch |
|---|---|
| `power__total < X µW` at TT | Combinational logic added without need |
| `design__instance__area__stdcell < Y µm²` | Design bloat |
| SS WNS ≥ +50 ps floor | Critical path slowly getting tighter |
| FF count vs closed-form | Full pipeline regression guard |
| Compression ratio regression | Algorithm or fixed-point width change |

## Where to look at run results

- **Live job logs** during a run:
  https://github.com/LonghornSilicon/kv-cache-engine/actions
- **Downloadable artifacts** (GDS, PNG, logs): bottom of each
  completed run page, "Artifacts" section
- **Pass/fail status badge**: green check or red X next to each
  commit on the commits page
