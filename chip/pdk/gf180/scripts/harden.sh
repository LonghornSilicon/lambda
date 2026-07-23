#!/usr/bin/env bash
# harden.sh — run a LibreLane Classic macro on GF180 inside the librelane
# docker image, with the whole monorepo worktree and the ciel PDK store mounted.
#
#   scripts/harden.sh <macro_yaml_basename>   # e.g. token_importance_unit
#
# The macro configs live in chip/pdk/gf180/librelane/<macro>.yaml and reference
# their RTL by paths relative to that dir (block sources of truth are under
# chip/verif/blocks/ and acu/*/rtl/), so the WHOLE worktree root is mounted at
# /work — not just chip/pdk/gf180. Outputs land in librelane/runs/<macro>/
# (gitignored). PDK defaults to gf180mcuD from the ciel store.
#
# Verified on the submission node: `harden.sh token_importance_unit` closes with
# a fully clean signoff (Magic DRC 0, LVS 0, antenna 0, setup/hold met) → GDS+LEF.
set -euo pipefail

MACRO="${1:?usage: harden.sh <macro>}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # monorepo worktree root
PDK_ROOT_HOST="${PDK_ROOT_HOST:-/home/shadeform/.ciel}"
PDK="${PDK:-gf180mcuD}"
IMG="ghcr.io/librelane/librelane:3.0.5"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e PDK_ROOT=/pdk \
  -e HOME=/tmp \
  -v "$PDK_ROOT_HOST":/pdk \
  -v "$ROOT":/work \
  -w /work/chip/pdk/gf180/librelane \
  "$IMG" \
  bash -lc "librelane ./${MACRO}.yaml -p ${PDK} --pdk-root /pdk --manual-pdk --run-tag ${MACRO} 2>&1"
