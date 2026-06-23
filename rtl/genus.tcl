## genus.tcl — Cadence Genus synthesis flow for kv_cache_engine
##
## Targets TSMC 16FFC via the TSMC University Program PDK.
## Edit the three TSMC_* variables below to point at actual PDK paths, then:
##
##    genus -files genus.tcl -log reports/genus.log
##
## Reports land in reports/:
##   reports/area.rpt       — final cell area (µm²) and breakdown
##   reports/timing_*.rpt   — slack at SS / TT / FF corners
##   reports/power.rpt      — leakage + dynamic power
##   reports/qor.rpt        — overall quality of results
##   netlist/kv_cache_engine_mapped.v — gate-level netlist for Innovus

# ---------------------------------------------------------------------------
# Process / library setup — EDIT THESE for the chamber's PDK paths
# ---------------------------------------------------------------------------
set TSMC_LIB_DIR  "/path/to/tsmc16/stdcells"
set TSMC_LEF_DIR  "/path/to/tsmc16/lef"
set TSMC_QRC_DIR  "/path/to/tsmc16/qrc"

# Worst-case (SS / 0.72V / 125C) — sign-off corner
set LIB_SS_125C   "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvtssg0p72v125c.lib"
# Typical (TT / 0.80V / 25C) — datasheet corner
set LIB_TT_25C    "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvttt0p80v25c.lib"
# Best-case (FF / 0.88V / -40C) — hold timing corner
set LIB_FF_M40C   "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvtffg0p88vm40c.lib"

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
set TOP "kv_cache_engine"
# Synthesizable shell (top FSM + SRAM). The ChannelQuant datapath cores in
# cq_units.sv are a behavioral (real-valued) golden-equivalent model used for
# bit-exact parity, NOT synthesizable as-is; the fp16 fixed-function lowering is
# the P4 synthesis phase (see TEARDOWN.md / findings/channelquant_block_revamp.md).
set RTL_FILES [list \
    kv_cache_engine.sv \
    sram_controller.sv \
]
set SDC_FILE  "constraints/timing.sdc"

file mkdir reports
file mkdir netlist

# Set up multi-corner library
set_db library [list $LIB_SS_125C $LIB_TT_25C $LIB_FF_M40C]

# ---------------------------------------------------------------------------
# Read RTL
# ---------------------------------------------------------------------------
read_hdl -sv $RTL_FILES
elaborate $TOP

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
read_sdc $SDC_FILE

# Multi-corner / multi-mode setup
create_mode -name func -sdcs [list $SDC_FILE]
create_constraint_mode -name func -sdc_files [list $SDC_FILE]
create_library_set -name ss -timing [list $LIB_SS_125C]
create_library_set -name tt -timing [list $LIB_TT_25C]
create_library_set -name ff -timing [list $LIB_FF_M40C]

create_delay_corner -name ss_corner -library_set ss
create_delay_corner -name tt_corner -library_set tt
create_delay_corner -name ff_corner -library_set ff

create_analysis_view -name view_ss -constraint_mode func -delay_corner ss_corner
create_analysis_view -name view_tt -constraint_mode func -delay_corner tt_corner
create_analysis_view -name view_ff -constraint_mode func -delay_corner ff_corner

set_analysis_view -setup [list view_ss view_tt] -hold [list view_ff]

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
syn_generic
syn_map
syn_opt -incremental

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
report_qor                          > reports/qor.rpt
report_area    -depth 5             > reports/area.rpt
report_timing  -nworst 10 -view view_ss > reports/timing_ss.rpt
report_timing  -nworst 10 -view view_tt > reports/timing_tt.rpt
report_timing  -nworst 10 -view view_ff > reports/timing_ff.rpt
report_power                        > reports/power.rpt
report_gates   -power_domain        > reports/gates.rpt

# ---------------------------------------------------------------------------
# Output netlist (for Innovus PnR)
# ---------------------------------------------------------------------------
write_hdl -mapped > netlist/${TOP}_mapped.v
write_sdc          > netlist/${TOP}.sdc
write_sdf          > netlist/${TOP}.sdf

# Print a one-line summary
puts "==================================================================="
puts " Genus flow complete."
puts "   QoR        : reports/qor.rpt"
puts "   Area       : reports/area.rpt"
puts "   Timing SS  : reports/timing_ss.rpt"
puts "   Power      : reports/power.rpt"
puts "==================================================================="

exit
