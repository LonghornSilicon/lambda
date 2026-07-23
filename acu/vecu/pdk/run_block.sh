#!/usr/bin/env bash
# run_block.sh <pdk> <config_rel_path> <run_tag>
#   pdk: gf180mcuD | sky130A
#   config_rel_path: path (relative to monorepo root) to the .yaml/.json config
#   run_tag: run tag (also the runs/<tag> dir name)
set -euo pipefail
PDK="$1"; CFG_REL="$2"; TAG="$3"
ROOT="/home/shadeform/lhs/lambda"
PDK_ROOT_HOST="/home/shadeform/.ciel"
IMG="ghcr.io/librelane/librelane:3.0.5"
CFG_DIR="$(dirname "$CFG_REL")"
CFG_BASE="$(basename "$CFG_REL")"
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e PDK_ROOT=/pdk -e HOME=/tmp \
  -v "$PDK_ROOT_HOST":/pdk \
  -v "$ROOT":/work \
  -w "/work/${CFG_DIR}" \
  "$IMG" \
  bash -lc "librelane ./${CFG_BASE} -p ${PDK} --pdk-root /pdk --manual-pdk --run-tag ${TAG} 2>&1"
