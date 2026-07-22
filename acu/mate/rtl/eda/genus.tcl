## genus.tcl — Cadence Genus synthesis flow for precision_controller
##
## Targets TSMC 16FFC via the TSMC University Program PDK.
## On the chamber, edit the three TSMC_* variables below to point at the
## actual PDK paths, then:
##
##    genus -files genus.tcl -log reports/genus.log
##
## Reports land in reports/:
##   reports/area.rpt       — final cell area (µm²) and breakdown
##   reports/timing.rpt     — slack at SS / TT / FF corners
##   reports/power.rpt      — leakage + dynamic power
##   reports/qor.rpt        — overall quality of results
##   netlist/precision_controller.v  — gate-level netlist for Innovus

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
set TOP "precision_controller"
set RTL_FILES [list precision_controller.sv]
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

# Apply parameter overrides if running a sweep (defaults match the 64x64 reference)
# set_parameter BLOCK_M 64
# set_parameter BLOCK_N 64
# set_parameter SCORE_WIDTH 8

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
