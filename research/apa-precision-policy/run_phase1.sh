#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure OpenEvolve is importable
export PYTHONPATH="$SCRIPT_DIR:/home/shadeform/openevolve:$PYTHONPATH"

echo "=== FlashAttention-5 Phase 1: Precision Policy Evolution ==="
echo "Using OpenEvolve with Claude Code CLI backend"
echo "Workdir: $SCRIPT_DIR"
echo ""

python3 /home/shadeform/openevolve/openevolve-run.py \
    phase1_policy/initial_policy.py \
    phase1_policy/evaluator.py \
    --config phase1_policy/config.yaml \
    --iterations 80 \
    "$@"

echo ""
echo "=== Phase 1 complete ==="
echo "Best policy saved in checkpoints/ directory"
