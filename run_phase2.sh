#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:/home/shadeform/openevolve:$PYTHONPATH"

echo "=== FlashAttention-5 Phase 2: Kernel Evolution ==="
echo "Using OpenEvolve with Claude Code CLI backend"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

python3 -u /home/shadeform/openevolve/openevolve-run.py \
    phase2_kernel/initial_kernel.py \
    phase2_kernel/evaluator.py \
    --config phase2_kernel/config.yaml \
    --iterations 60 \
    "$@"

echo ""
echo "=== Phase 2 complete ==="
echo "Best kernel saved in phase2_kernel/openevolve_output/best/"
