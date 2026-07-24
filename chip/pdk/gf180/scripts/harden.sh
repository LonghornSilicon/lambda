#!/usr/bin/env bash
# harden.sh — run a LibreLane Classic macro on GF180 inside the librelane
# docker image, with the whole monorepo worktree and the ciel PDK store mounted.
#
#   scripts/harden.sh <macro_yaml_basename>   # e.g. token_importance_unit
#
# Each block owns its GF180 config block-major, under
# <block>/pdk/gf180/librelane/<macro>.yaml (e.g. src/blocks/kve/pdk/gf180/librelane/kve.yaml,
# src/blocks/tiu/pdk/gf180/librelane/token_importance_unit.yaml, acu/*/pdk/gf180/librelane/*.yaml).
# This script locates the config by macro name anywhere under the worktree and
# runs it from its own directory (its RTL is referenced by paths relative to that
# dir — block sources of truth are each block's rtl/). The WHOLE worktree root is
# mounted at /work so cross-block relative paths resolve. Outputs land in that
# block's librelane/runs/<macro>/ (gitignored). PDK defaults to gf180mcuD.
#
# Verified on the submission node: `harden.sh token_importance_unit` closes with
# a fully clean signoff (Magic DRC 0, LVS 0, antenna 0, setup/hold met) → GDS+LEF.
set -euo pipefail

MACRO="${1:?usage: harden.sh <macro>}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # monorepo worktree root
PDK_ROOT_HOST="${PDK_ROOT_HOST:-/home/shadeform/.ciel}"
PDK="${PDK:-gf180mcuD}"
IMG="ghcr.io/librelane/librelane:3.0.5"

# Locate the block-major config for this macro (path relative to the worktree root).
CFG_REL="$(cd "$ROOT" && find . -path './.git' -prune -o \
  -path "*/pdk/gf180/librelane/${MACRO}.yaml" -print | head -1 | sed 's|^\./||')"
if [ -z "$CFG_REL" ]; then
  echo "harden.sh: no config found for macro '${MACRO}' (looked for */pdk/gf180/librelane/${MACRO}.yaml)" >&2
  exit 1
fi
CFG_DIR="$(dirname "$CFG_REL")"   # e.g. src/blocks/kve/pdk/gf180/librelane

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e PDK_ROOT=/pdk \
  -e HOME=/tmp \
  -v "$PDK_ROOT_HOST":/pdk \
  -v "$ROOT":/work \
  -w "/work/${CFG_DIR}" \
  "$IMG" \
  bash -lc "librelane ./${MACRO}.yaml -p ${PDK} --pdk-root /pdk --manual-pdk --run-tag ${MACRO} 2>&1"
