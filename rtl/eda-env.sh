# Source me to put the local EDA toolchain on PATH:  . rtl/eda-env.sh
# Built locally 2026-06-22 on this aarch64 host (no system simulator / no sudo).
#   - iverilog/vvp 12.0  -> built from source (see .tools/iverilog-12_0)
#   - verilator, gperf   -> micromamba env "eda" (conda-forge)
# Rebuild recipe is recorded in NOTES.md (2026-06-22 entry).
_LHS_TOOLS=/home/chaithu/lhs/.tools
export PATH="$_LHS_TOOLS/iverilog/bin:$_LHS_TOOLS/mamba/envs/eda/bin:$PATH"
# sanity: `iverilog -V | head -1`  should print "Icarus Verilog version 12.0"
