#!/usr/bin/env bash
# submit.sh — assemble the Lambda ACU into the Chipathon 2026 GF180 workshop
# padring and drive the full-chip Chip flow to GDS.
#
#   scripts/submit.sh [FORK_DIR]
#
# Steps (idempotent):
#   1. clone Mauricio-xx/chipathon-2026-gf180mcu-padring   (if not present)
#   2. fetch the wafer.space GF180 PDK @ 1.8.0             (ws_io / ws_ip cells)
#   3. drop OUR chip_core.sv + lambda_acu.sv + spi_loader.sv + block RTL into
#      <fork>/src/, and OUR librelane/config_fullchip.yaml over the fork's
#      librelane/config.yaml
#   4. SLOT=workshop make librelane                        (LibreLane 3.0.5 Chip flow)
#
# This is the FLAT-core flow: the synthesizable KVE value-path lowering (*_syn,
# bit-exact vs behavioral) lets lambda_acu synthesize whole. See SUBMISSION.md §6.
# Run inside the librelane 3.0.5 docker (see harden.sh) or a native LibreLane env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"       # lambda monorepo root
GF180="$ROOT/chip/pdk/gf180"
FORK="${1:-$ROOT/../chipathon-2026-gf180mcu-padring}"
PDK_TAG="${PDK_TAG:-1.8.0}"

echo "== 1. padring fork =="
if [ ! -d "$FORK/.git" ]; then
  git clone https://github.com/Mauricio-xx/chipathon-2026-gf180mcu-padring.git "$FORK"
fi

echo "== 2. wafer.space GF180 PDK ($PDK_TAG) =="
if [ ! -d "$FORK/gf180mcu/gf180mcuD" ]; then
  git clone https://github.com/wafer-space/gf180mcu.git "$FORK/gf180mcu" --depth 1 --branch "$PDK_TAG"
fi

echo "== 3. drop in our core + block RTL =="
SRC="$FORK/src"
# our core (replaces the fork's trivial counter core)
cp "$ROOT/chip/rtl/chip_core.sv"          "$SRC/chip_core.sv"
cp "$ROOT/chip/rtl/lambda_acu.sv"         "$SRC/lambda_acu.sv"
cp "$ROOT/chip/rtl/spi_loader.sv"         "$SRC/spi_loader.sv"
# ACU compute blocks
cp "$ROOT/acu/mate/rtl/mate_qkt.sv"       "$SRC/mate_qkt.sv"
cp "$ROOT/acu/mate/rtl/mate_pv.sv"        "$SRC/mate_pv.sv"
cp "$ROOT/acu/mate/rtl/mate_pv_fp16.sv"   "$SRC/mate_pv_fp16.sv"
cp "$ROOT/acu/vecu/rtl/vecu_softmax.sv"   "$SRC/vecu_softmax.sv"
cp "$ROOT/acu/precision_controller/rtl/precision_controller.sv" "$SRC/precision_controller.sv"
# TIU
cp "$ROOT/tiu/rtl/token_importance_unit.sv" "$SRC/token_importance_unit.sv"
# KVE value path — SYNTHESIZABLE lowering (no `real`)
cp "$ROOT/kve/rtl/wht_unit_syn.sv"           "$SRC/wht_unit_syn.sv"
cp "$ROOT/kve/rtl/fp16_addsub_syn.sv"        "$SRC/fp16_addsub_syn.sv"
cp "$ROOT/kve/rtl/cq_units_syn.sv"           "$SRC/cq_units_syn.sv"
cp "$ROOT/kve/rtl/cq_value_path_wht_syn.sv"  "$SRC/cq_value_path_wht_syn.sv"
cp "$ROOT/kve/rtl/wht_inverse_out_syn.sv"    "$SRC/wht_inverse_out_syn.sv"
# our full-chip config over the fork's
cp "$GF180/librelane/config_fullchip.yaml" "$FORK/librelane/config.yaml"

echo "== 4. build (SLOT=workshop) =="
cd "$FORK"
SLOT=workshop make librelane
echo "== done — GDS at $FORK/final/gds/chip_top.gds (if the flow closed) =="
