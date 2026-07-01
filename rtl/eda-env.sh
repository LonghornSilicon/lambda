# Source me to put the local EDA toolchain on PATH:  . rtl/eda-env.sh
# The toolchain lives OUTSIDE the repo at  <lhs>/.tools  (sibling of kv-cache-engine),
# provisioned per-host (not committed). iverilog/vvp 12.0 is the tool of record.
#   - x86_64 : micromamba create -n eda -c conda-forge iverilog verilator gperf
#   - aarch64: iverilog built from source (conda-forge has no aarch64 build);
#              verilator + gperf via micromamba env "eda".
# Per-host provenance + rebuild recipe are recorded in NOTES.md.

# Locate this script so the tools path is derived, not hard-coded to one host.
if [ -n "${BASH_SOURCE:-}" ]; then _src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then _src="${(%):-%N}"
else _src="$0"; fi
_LHS_TOOLS="$(cd "$(dirname "$_src")/../.." && pwd)/.tools"

# Prepend whichever layout exists: conda-forge env (x86_64) and/or from-source iverilog (aarch64).
for _p in "$_LHS_TOOLS/mamba/envs/eda/bin" "$_LHS_TOOLS/iverilog/bin"; do
  [ -d "$_p" ] && PATH="$_p:$PATH"
done
export PATH
unset _src _p _LHS_TOOLS
# sanity: `iverilog -V | head -1`  should print "Icarus Verilog version 12.0"
