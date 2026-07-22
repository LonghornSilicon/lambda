#!/usr/bin/env bash
# harden.sh — run a LibreLane Classic macro on GF180 inside the librelane
# docker image, with this repo and the ciel PDK store mounted.
#
#   scripts/harden.sh <macro_yaml_basename>   # e.g. precision_controller
#
# Outputs land in librelane/runs/<macro>/ (gitignored). PDK is gf180mcuD.
set -euo pipefail

MACRO="${1:?usage: harden.sh <macro>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PDK_ROOT_HOST="/home/shadeform/.ciel"
IMG="ghcr.io/librelane/librelane:3.0.5"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e PDK_ROOT=/pdk \
  -e HOME=/tmp \
  -v "$PDK_ROOT_HOST":/pdk \
  -v "$REPO":/work \
  -w /work/librelane \
  "$IMG" \
  bash -lc "librelane ./${MACRO}.yaml -p gf180mcuD --run-tag ${MACRO} 2>&1"
