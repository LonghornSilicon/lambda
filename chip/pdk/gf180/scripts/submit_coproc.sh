#!/usr/bin/env bash
# submit_coproc.sh — assemble the Lambda KV-cache COPROCESSOR into the Chipathon
# 2026 GF180 workshop padring and drive the full-chip Chip flow to a clean GDS.
#
#   scripts/submit_coproc.sh [FORK_DIR]
#
# THE chipathon submission: KVE CQ-3 value compressor + TIU H2O importance +
# precision gate (~0.5 mm2) — fits the 4.21 mm2 workshop core with wide margin
# (unlike the full fp16 attention datapath; see submit.sh + SUBMISSION.md).
#
# Steps (idempotent):
#   1. clone the padring fork          2. fetch wafer.space GF180 PDK @ 1.8.0
#   3. drop the coprocessor RTL + config_coproc.yaml into the fork
#   4. SLOT=workshop make librelane
# Run inside the librelane 3.0.5 docker (see harden.sh) or a native LibreLane env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
GF180="$ROOT/chip/pdk/gf180"
FORK="${1:-$ROOT/../chipathon-2026-gf180mcu-padring}"
PDK_TAG="${PDK_TAG:-1.8.0}"

echo "== 1. padring fork =="
[ -d "$FORK/.git" ] || git clone https://github.com/Mauricio-xx/chipathon-2026-gf180mcu-padring.git "$FORK"

echo "== 2. wafer.space GF180 PDK ($PDK_TAG) =="
[ -d "$FORK/gf180mcu/gf180mcuD" ] || \
  git clone https://github.com/wafer-space/gf180mcu.git "$FORK/gf180mcu" --depth 1 --branch "$PDK_TAG"

echo "== 3. drop in the coprocessor =="
SRC="$FORK/src"
cp "$ROOT/chip/rtl/chip_core_kv.sv"        "$SRC/chip_core.sv"     # coproc core (replaces fork's)
cp "$ROOT/chip/rtl/lambda_kv_coproc.sv"    "$SRC/lambda_kv_coproc.sv"
cp "$ROOT/chip/rtl/spi_loader.sv"          "$SRC/spi_loader.sv"
cp "$ROOT/acu/precision_controller/rtl/precision_controller.sv" "$SRC/precision_controller.sv"
cp "$ROOT/tiu/rtl/token_importance_unit.sv" "$SRC/token_importance_unit.sv"
cp "$ROOT/kve/rtl/wht_unit_syn.sv"          "$SRC/wht_unit_syn.sv"
cp "$ROOT/kve/rtl/fp16_addsub_syn.sv"       "$SRC/fp16_addsub_syn.sv"
cp "$ROOT/kve/rtl/cq_units_syn.sv"          "$SRC/cq_units_syn.sv"
cp "$ROOT/kve/rtl/cq_value_path_wht_syn.sv" "$SRC/cq_value_path_wht_syn.sv"
cp "$GF180/librelane/config_coproc.yaml"   "$FORK/librelane/config.yaml"

# Drop the port-less decorative id/logo macros (SYNTH_HIERARCHY_MODE: keep still
# opt_cleans empty instances; manual placement then aborts). Cosmetic. Idempotent.
python3 - "$SRC/chip_top.sv" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
for mod, inst in [("id", "chip_id"), ("logo", "wafer_space_logo")]:
    s = re.sub(r'^([ \t]*)(?:\(\* keep \*\) )?gf180mcu_ws_ip__%s %s \(\);' % (mod, inst),
               r'\1// [lambda] decorative macro dropped: gf180mcu_ws_ip__%s %s' % (mod, inst),
               s, flags=re.M)
open(p, 'w').write(s)
PYEOF

echo "== 4. build (SLOT=workshop) =="
cd "$FORK"
SLOT=workshop make librelane
echo "== done — GDS at $FORK/final/gds/chip_top.gds =="
