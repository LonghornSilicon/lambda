# ASAP7 (predictive 7nm FinFET) synthesis + PnR — research bracket

**These are RESEARCH / PREDICTIVE numbers, not sign-off.** ASAP7 is a *predictive*
academic 7nm FinFET PDK (ASU/ARM). Its rule deck is realistic-shaped but **not
manufacturable**, its cell timing/power are model-based, and "clean" here means
*passes the predictive rules and closes timing*, not tape-out-ready. The point of
this directory is a **device-family bracket**: the same signed-off RTL that closed
on **Sky130 (130 nm planar)** is pushed through **ASAP7 (7 nm FinFET)** so we can
see the numbers on a FinFET node — a much better proxy for the chip's **TSMC
16 nm FinFET** target than planar Sky130. See `../../docs/pdk_bracket_asap7.md` for
the full Sky130-vs-ASAP7 table and the disclaimer discussion.

## Why ORFS (not LibreLane)
ASAP7 is a **first-class built-in platform in OpenROAD-flow-scripts** (`flow/platforms/asap7`),
shipped *with* ORFS — no separate PDK install. LibreLane/OpenLane2 is Sky130/GF180-oriented
and does not ship ASAP7 cleanly. So the Sky130 sign-off used OpenLane, and this ASAP7
bracket uses ORFS. Same RTL, same N=4 proxy, same floorplan-util intent.

## How it's run
Everything runs inside the prebuilt **`openroad/orfs:latest`** docker image (bundles
`openroad`, `yosys`, and the `asap7` platform). No local OpenROAD build needed.

```
./run_asap7.sh <block> <clk_ps> [target]
#   block  : mate_pv | mate_pv_fp16 | precision_controller
#   clk_ps : clock period in PICOSECONDS (ASAP7 liberty time_unit = 1 ps)
```

`run_asap7.sh` mounts the repo at `/work`, writes the requested clock period into
the block's `constraint.sdc`, runs `make` (synth → floorplan → place → CTS → route
→ final) with `WORK_HOME=orfs/asap7/build`, and pulls metrics from ORFS's
`metadata.json` / `6_finish.rpt`. To clean a run (files are root-owned because docker
runs as root):

```
docker run --rm -v "$PWD/../..":/work openroad/orfs:latest bash -lc 'rm -rf /work/orfs/asap7/build'
```

## Layout
```
mate_pv/               config.mk + constraint.sdc + results_asap7/ (final metrics, rpt, gds)
mate_pv_fp16/          "
precision_controller/  "
run_asap7.sh           docker driver
build/                 ORFS working tree (git-ignored; regenerate with run_asap7.sh)
```
Each `results_asap7/` holds the closing-run `*_metrics.json` (ORFS metadata),
`6_finish.rpt` (timing/power), `synth_stat.txt`, and the final GDS + layout webp.

## Two flow gotchas worked around (both in `run_asap7.sh`, documented inline)
1. **Clock-period extraction**: ORFS greps `set clk_period <N>` from the SDC for the
   ABC delay target, so the period must be a **literal integer** — the script seds it in.
2. **`signed` keyword**: yosys' `write_verilog` emits `input/wire/output signed [..]`
   and OpenROAD's `read_verilog` (dbSta) throws **STA-0171 syntax error** on `signed`.
   In a gate-mapped netlist `signed` is cosmetic (arithmetic is already lowered to
   cells), so the script strips it from the netlist *declaration* lines between
   synthesis and floorplan. **The signed-off RTL is never modified.**

## Correctness notes that make or break these numbers
- **ASAP7 4× draw-scale — RESOLVED as REAL µm².** ASAP7's *original* GDS/LEF are drawn
  at 4× real dimensions (areas 16× inflated). This ORFS platform ships the **de-scaled
  1× LEFs** (`asap7_tech_1x_201209.lef`, `asap7sc7p5t_..._1x_...lef`). Verified directly:
  the placement SITE is `SIZE 0.054 BY 0.270` µm — the *real* ASAP7 7.5-track cell height
  (270 nm) and CPP (54 nm). So **the µm² ORFS reports are REAL — no /16 applied or needed.**
  Cross-check: mate_pv shrinks ~90× vs Sky130, physically sane for 130 nm→7 nm; a raw
  16×-inflated number would imply an impossible ~6× shrink.
- **Corner / multi-Vt**: ORFS asap7 default lib set, primary **RVT** (BUF/INV/logic RVT),
  NLDM models. Timing corners: setup at SS 0.63 V/100 °C, hold at FF 0.77 V/25 °C,
  nominal TT 0.70 V/0 °C. Hold buffer cell is the RVT BUFx2.
- **No SRAM / no IO cells** used — all three targets are pure logic, so the missing
  ASAP7 SRAM/IO story is a non-issue here (noted for completeness).
- **Clock uncertainty**: setup = 5 % of period; hold = fixed **5 ps** (absolute). A flat
  %-of-period hold margin like Sky130's would put ~50–100 ps of hold uncertainty against
  7 nm min-delays and trigger a hold-buffer storm — the small absolute hold margin is
  standard practice for a fast node.
