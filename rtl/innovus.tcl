## innovus.tcl — Cadence Innovus place-and-route flow for kv_cache_engine
##
## Run after genus.tcl produces netlist/kv_cache_engine_mapped.v:
##    innovus -files innovus.tcl -log reports/innovus.log
##
## Outputs:
##   reports/pnr_summary.rpt    — final area / WNS / TNS / power
##   reports/timing_pnr_*.rpt   — sign-off slack at SS / FF corners
##   results/kv_cache_engine.gds — GDSII for tape-out
##   results/kv_cache_engine.def

# ---------------------------------------------------------------------------
# PDK setup — EDIT for chamber paths
# ---------------------------------------------------------------------------
set TOP            "kv_cache_engine"
set TSMC_LEF_DIR   "/path/to/tsmc16/lef"
set TSMC_LIB_DIR   "/path/to/tsmc16/stdcells"
set TSMC_QRC_DIR   "/path/to/tsmc16/qrc"

set TECH_LEF       "$TSMC_LEF_DIR/tsmc16_tech.lef"
set STD_CELL_LEFS  [glob $TSMC_LEF_DIR/tcbn16ffcllbwp7t30p140lvt*.lef]
set LIB_SS         "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvtssg0p72v125c.lib"
set LIB_TT         "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvttt0p80v25c.lib"
set LIB_FF         "$TSMC_LIB_DIR/tcbn16ffcllbwp7t30p140lvtffg0p88vm40c.lib"

# ---------------------------------------------------------------------------
# Read netlist
# ---------------------------------------------------------------------------
file mkdir reports results
set init_verilog        netlist/${TOP}_mapped.v
set init_top_cell       $TOP
set init_lef_file       [concat $TECH_LEF $STD_CELL_LEFS]
set init_mmmc_file      mmmc.tcl

# ---------------------------------------------------------------------------
# Floorplan
# ---------------------------------------------------------------------------
init_design

# ~6400 FFs → estimate ~250x250 µm at 16FFC with 70% utilization
floorPlan -site core7T -r 1.0 0.7 10.0 10.0 10.0 10.0

# Place power rings + stripes
addRing -nets {VDD VSS} -width 0.8 -spacing 0.4 -layer {bottom M3 top M3 left M4 right M4}
addStripe -nets {VDD VSS} -layer M5 -width 0.4 -spacing 0.8 -set_to_set_distance 10.0

# ---------------------------------------------------------------------------
# Place + CTS + route
# ---------------------------------------------------------------------------
setPlaceMode -place_global_place_io_pins true
place_design

setOptMode -fixCap true -fixTran true -fixFanoutLoad true
optDesign -preCTS

# Clock tree synthesis
ccopt_design

optDesign -postCTS

# Routing
setNanoRouteMode -routeWithTimingDriven true
routeDesign

optDesign -postRoute

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
report_design > reports/pnr_summary.rpt
report_timing -late  > reports/timing_pnr_setup.rpt
report_timing -early > reports/timing_pnr_hold.rpt
report_power -outfile reports/power_pnr.rpt
report_area  -depth 5 > reports/area_pnr.rpt

# ---------------------------------------------------------------------------
# Stream out GDS
# ---------------------------------------------------------------------------
streamOut results/${TOP}.gds -mapFile $TSMC_LEF_DIR/streamOut.map -libName ${TOP}LIB
defOut   results/${TOP}.def

puts "==================================================================="
puts " Innovus PnR complete."
puts "   GDS           : results/${TOP}.gds"
puts "   Setup slack   : reports/timing_pnr_setup.rpt"
puts "   Hold  slack   : reports/timing_pnr_hold.rpt"
puts "   Power         : reports/power_pnr.rpt"
puts "==================================================================="

exit
