#!/usr/bin/env bash
# Drive the ORFS asap7 flow for one APA logic block inside the openroad/orfs
# docker image. PREDICTIVE / RESEARCH flow -- not sign-off.
#
#   ./run_asap7.sh <block> <clk_ps> [target]
#
#   block   : mate_pv | mate_pv_fp16 | precision_controller
#   clk_ps  : clock period in PICOSECONDS (ASAP7 liberty time_unit = 1ps)
#   target  : final make target (default: finish -> synth..floorplan..route..final)
#
# Two gotchas this script works around, both discovered empirically:
#  1. ORFS's synth step greps 'set clk_period <N>' from the SDC, so the period
#     must be a literal integer -- we sed it in here (in ps).
#  2. Yosys' write_verilog emits `input/wire/output signed [..]` declarations,
#     and OpenROAD's read_verilog (dbSta) throws STA-0171 "syntax error" on the
#     `signed` keyword. In a gate-mapped netlist `signed` is purely cosmetic
#     (all arithmetic is already lowered to cells), so we strip it from the
#     DECLARATION lines of 1_2_yosys.v between synthesis and floorplan. The
#     signed-off RTL is never touched. The strip runs as root INSIDE the
#     container because docker writes the netlist root-owned.
#
# Outputs land on the host under orfs/asap7/build/{results,reports,logs}/asap7/<block>/base
set -euo pipefail
BLOCK="${1:?block}"; CLK_PS="${2:?clk_ps}"; TARGET="${3:-finish}"
REPO=/home/shadeform/lhs/adaptive-precision-attention
IMG=openroad/orfs:latest
SDC="$REPO/orfs/asap7/$BLOCK/constraint.sdc"

# Gotcha 1: bake the requested period into the SDC (literal, integer picoseconds).
sed -i -E "s/^set clk_period [0-9]+/set clk_period ${CLK_PS}/" "$SDC"
echo ">>> $BLOCK @ ${CLK_PS} ps ($(python3 -c "print(f'{1e6/${CLK_PS}:.0f} MHz')"))"

docker run --rm \
  -v "$REPO":/work \
  -w /OpenROAD-flow-scripts/flow \
  "$IMG" bash -lc "
    set -e
    source /OpenROAD-flow-scripts/env.sh >/dev/null 2>&1 || true
    CFG=/work/orfs/asap7/$BLOCK/config.mk
    WH=/work/orfs/asap7/build
    NETLIST=\$WH/results/asap7/$BLOCK/base/1_2_yosys.v
    # Phase 1: synthesis only -> 1_2_yosys.v
    make DESIGN_CONFIG=\$CFG WORK_HOME=\$WH \$NETLIST
    # Gotcha 2: strip 'signed' from declaration lines (safe on a mapped netlist).
    sed -i -E 's/^([[:space:]]*(input|output|inout|wire|reg))[[:space:]]+signed[[:space:]]+/\1 /' \$NETLIST
    # Phase 2: floorplan..route..final (reuses the patched, up-to-date netlist)
    make DESIGN_CONFIG=\$CFG WORK_HOME=\$WH $TARGET
  "
