# CI pipeline overview

End-to-end walkthrough of what runs on every push, what's gated, what's
saved, and where the FPGA bitstream step will plug in when the
ZCU102/104 arrives.

The pipeline lives in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
It is a thin caller for the shared workflow at
`LonghornSilicon/.github/.github/workflows/block-ci.yml@main`, which
runs all the actual jobs on GitHub-hosted Ubuntu (no self-hosted
runner required). The shared workflow includes line-coverage and
formal-equivalence gates in addition to functional verification,
synthesis, OpenLane Sky130 sign-off, and paper build. Each job's
first step ("About this gate") prints a description of its purpose,
method, and what silicon-bug class it catches — click any job on the
run page and expand that first step to read it.

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
       ▼             ▼         ▼                 ▼
   rtl-func-     rtl-synth  openlane-sky130   paper-
   verification             (your cloud box)  build
   (Ubuntu)      (Ubuntu)   (self-hosted)     (Ubuntu)
       │             │         │                 │
       ▼             ▼         ▼                 ▼
   pass/fail   pass/fail   pass/fail        pass/fail
                          │
              ┌───────────┴───────────┐
              │ green check on commit │
              │ artifacts available   │
              │ for 30-day retention  │
              └───────────────────────┘
```

Jobs run concurrently — they don't wait on each other. The whole CI run
completes when the slowest one finishes (currently OpenLane at ~3 min
once the LibreLane image + Sky130 PDK are cached).

## Job-by-job: what's checked vs. what's recorded

### 1. `rtl-functional-verification` — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install iverilog + numpy | `apt-get` the simulator + Python deps |
| `make testvectors` | Python regenerates 143 INT8-quantized replay tiles into hex files |
| `make sim` | Compile RTL + directed TB, run 110 cases against integer reference |
| `make sim_realdata` | Compile RTL + replay TB, run 143 cases (real attention scores) |

**Gate**: every case must pass. Workflow greps for `ALL TESTS PASSED`
(directed) and `Tests: 143  Pass: 143` (replay). One failure → job red.

**Records**: nothing as artifact (logs visible in the job page). Could
add `*.vcd` waveforms here if useful for debugging.

### 2. `rtl-synthesis` — GitHub Ubuntu, ~30 sec

| Step | What it does |
|---|---|
| Install yosys | `apt-get` |
| `yosys -p "read_verilog ... synth ... stat"` | RTL → generic gate netlist + cell-count breakdown |
| Awk-extract FF count | Sum every cell line containing `DFF` |

**Gate**: `FF_COUNT == 30` — matches the closed-form prediction
`2·SCORE_WIDTH + log₂N + 2` for the default parameters
(`BLOCK_M = BLOCK_N = 64`, `SCORE_WIDTH = 8`). If someone refactors the
RTL and accidentally adds a register, CI flags it.

**Records**: `rtl/synth.log` uploaded as the `yosys-synth-log`
artifact (30-day retention).

### 3. `openlane-sky130` — self-hosted runner, ~3–10 min

| Step | What it does |
|---|---|
| Verify `librelane` + `docker` on runner | Smoke check the tooling |
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

- `precision_controller.gds` — final layout (~900 KB, tape-out-ready
  modulo Sky130 vs 16FFC)
- `metrics.json` — every PPA number (area, timing per corner, power
  per category, wirelength)
- `precision_controller.png` — rendered layout image

### 4. `paper-build` — GitHub Ubuntu, ~1 min

| Step | What it does |
|---|---|
| Install texlive | `apt-get` |
| `pdflatex → bibtex → pdflatex → pdflatex` | Standard 4-pass LaTeX build |
| Grep `.log` for undefined references | Sanity check |

**Gate**: no LaTeX errors, no undefined references, no missing
citations. Overfull-hbox warnings are non-fatal.

**Records**: `adaptive_precision_attention.pdf` uploaded as the
`paper-pdf` artifact.

## Where the FPGA bitstream step plugs in

When the ZCU102 or ZCU104 arrives, a fifth job will sit parallel to the
others (gated on `rtl-functional-verification` passing — no point
burning 30 min of Vivado time on RTL that doesn't pass sim):

```yaml
  vivado-zcu102-bitstream:
    name: Vivado ZCU102 bitstream
    runs-on: [self-hosted, linux, x64, vivado]   # extra label
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
          # Assert: LUTs < 1000, FFs < 200, BRAM = 0, DSP = 0
          ...

      - name: Assert WNS > 0 at 200 MHz target
        run: |
          # Parse impl_1/post_route_timing_summary.rpt
          # Look for 'Worst Negative Slack' >= 0
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

**Gates** (mirroring the Sky130 pattern):

- Vivado synth + impl complete without errors
- Utilization within budget (~600 LUT-equivalents for our design —
  trivial on the ZCU102's 500K LUTs)
- Post-route WNS ≥ 0 at the 200 MHz target already noted in
  `rtl/precision_controller.sv`

**Records**:

- `.bit` — bitstream to flash on the board
- `.hwh` — hardware handoff for PYNQ Python integration
- Utilization + timing reports

**Why self-hosted**: Vivado is a ~30 GB install. We install it once on
the cloud box and every run reuses it. The extra `vivado` label
prevents the OpenLane job from accidentally landing on a Vivado-only
runner and vice versa.

**Why `needs:`-gated**: Vivado synth + impl is slow (~20–40 min even
for a small design — most of that is Vivado overhead, not our logic).
Not worth spending an hour when the RTL is broken anyway.

## What's not yet checked (future gates worth adding)

These are all `python3 -c "assert m[...] < X"` one-liners against
`metrics.json`. Add them when a regression bites once and we want it
to never happen again:

| Future gate | What it would catch |
|---|---|
| `power__total < 500 µW` at TT | Combinational logic added without need |
| `design__instance__area__stdcell < 4500 µm²` | Design bloat |
| SS WNS ≥ +50 ps floor | Critical path slowly getting tighter |
| Wirelength stability check | Placement/routing quality regression |
| Per-block FF count vs closed-form | When other LonghornSilicon blocks land |

## Where to look at run results

- **Live job logs** during a run:
  https://github.com/LonghornSilicon/attention-compute-unit/actions
- **Downloadable artifacts** (GDS, PNG, PDF, logs): bottom of each
  completed run page, "Artifacts" section
- **Pass/fail status badge**: green check or red X next to each
  commit on the commits page

## Extending to other LonghornSilicon blocks

When the KV Cache Engine, Token Importance Unit, or Memory Hierarchy
Controller repos come online, each gets the same shape:

1. `*-functional-verification` (block-specific testbench)
2. `*-synthesis` (block-specific FF-count assertion)
3. `openlane-sky130` (block-specific OpenLane config, same gate logic)
4. `vivado-zcu102-bitstream` (block-specific Vivado project)

If the runner has been moved to org-level (see the "Option B" section
in the README), the same self-hosted machine picks up all four blocks'
heavy jobs without any per-repo registration. See
[`docs/ci_setup.md`](ci_setup.md) for the migration walkthrough.
