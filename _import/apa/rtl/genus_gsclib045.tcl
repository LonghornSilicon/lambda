## genus_gsclib045.tcl — Cadence Genus synthesis flow for precision_controller
##
## Targets gsclib045 (Cadence Generic Std-Cell Library, 45 nm) on the TAMU
## chamber. This is a real industrial synth pass with real Liberty + LEF —
## ASAP7+yosys was our open-tool proxy; this is the Cadence cross-check.
##
## Paths are pinned to the chamber's PDK location and need no editing.
##
##   cd rtl/
##   genus -files genus_gsclib045.tcl -log reports/genus_gsclib045.log
##
## Reports land in reports/gsclib045/:
##   qor.rpt        — overall quality of results (read this FIRST)
##   area.rpt       — cell area (µm²) and breakdown
##   timing_ss.rpt  — slack at SS corner (sign-off setup)
##   timing_ff.rpt  — slack at FF corner (hold)
##   power.rpt      — leakage + dynamic power
##   gates.rpt      — gate-count breakdown by type
## Netlist:
##   netlist/gsclib045/precision_controller_mapped.v

# ---------------------------------------------------------------------------
# gsclib045 PDK paths (confirmed on TAMU chamber ae03ut01, 2026-05-10)
# ---------------------------------------------------------------------------
set GSCLIB "/process/hosted/gpdk/gpdk045/ip_libraries/gsclib045/v4p4/gsclib045"

# Slow corner — VDD=0.9V, T=125C — sign-off setup
set LIB_SS_BASIC    "$GSCLIB/timing/slow_vdd1v0_basicCells.lib"
set LIB_SS_MULTIDFF "$GSCLIB/timing/slow_vdd1v0_multibitsDFF.lib"

# Fast corner — VDD=1.1V, T=-40C — hold timing
set LIB_FF_BASIC    "$GSCLIB/timing/fast_vdd1v0_basicCells.lib"
set LIB_FF_MULTIDFF "$GSCLIB/timing/fast_vdd1v0_multibitsDFF.lib"

set LEF_TECH        "$GSCLIB/lef/gsclib045_tech.lef"
set LEF_MACRO       "$GSCLIB/lef/gsclib045_macro.lef"
set LEF_MULTIDFF    "$GSCLIB/lef/gsclib045_multibitsDFF.lef"

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
set TOP "precision_controller"
set RTL_FILES [list precision_controller.sv]
set SDC_FILE  "constraints/timing_gsclib045.sdc"

file mkdir reports/gsclib045
file mkdir netlist/gsclib045

# Set up the search/library
set_db library [list $LIB_SS_BASIC $LIB_SS_MULTIDFF]
set_db lef_library [list $LEF_TECH $LEF_MACRO $LEF_MULTIDFF]

# ---------------------------------------------------------------------------
# Read RTL
# ---------------------------------------------------------------------------
read_hdl -sv $RTL_FILES
elaborate $TOP

# ---------------------------------------------------------------------------
# Constraints + MMMC setup (2 corners: SS for setup, FF for hold)
# ---------------------------------------------------------------------------
read_sdc $SDC_FILE

create_constraint_mode -name func -sdc_files [list $SDC_FILE]
create_library_set -name ss -timing [list $LIB_SS_BASIC $LIB_SS_MULTIDFF]
create_library_set -name ff -timing [list $LIB_FF_BASIC $LIB_FF_MULTIDFF]

create_delay_corner -name ss_corner -library_set ss
create_delay_corner -name ff_corner -library_set ff

create_analysis_view -name view_ss -constraint_mode func -delay_corner ss_corner
create_analysis_view -name view_ff -constraint_mode func -delay_corner ff_corner

set_analysis_view -setup [list view_ss] -hold [list view_ff]

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
syn_generic
syn_map
syn_opt -incremental

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
report_qor                          > reports/gsclib045/qor.rpt
report_area    -depth 5             > reports/gsclib045/area.rpt
report_timing  -nworst 10 -view view_ss > reports/gsclib045/timing_ss.rpt
report_timing  -nworst 10 -view view_ff > reports/gsclib045/timing_ff.rpt
report_power                        > reports/gsclib045/power.rpt
report_gates                        > reports/gsclib045/gates.rpt

# ---------------------------------------------------------------------------
# Output netlist (for Innovus PnR, once admin restores the Innovus binary)
# ---------------------------------------------------------------------------
write_hdl -mapped > netlist/gsclib045/${TOP}_mapped.v
write_sdc          > netlist/gsclib045/${TOP}.sdc
write_sdf          > netlist/gsclib045/${TOP}.sdf

puts "==================================================================="
puts " Genus / gsclib045 flow complete."
puts "   QoR        : reports/gsclib045/qor.rpt"
puts "   Area       : reports/gsclib045/area.rpt"
puts "   Timing SS  : reports/gsclib045/timing_ss.rpt"
puts "   Power      : reports/gsclib045/power.rpt"
puts "   Netlist    : netlist/gsclib045/${TOP}_mapped.v"
puts "==================================================================="

exit
