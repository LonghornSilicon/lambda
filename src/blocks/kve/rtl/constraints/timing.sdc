# timing.sdc — SDC timing constraints for Cadence Genus/Innovus (ASIC flow)
#
# Primary target:
#   TSMC 16FFC (16nm FinFET Compact) via TSMC University Program
#   Target: 800 MHz = 1.25 ns period, signed off at SS 0.72V 125C
#
# Critical path estimate: The KV cache engine has deeper datapath than
# block #1 (WHT butterfly, QJL accumulation, norm sqrt). Conservative
# estimate: ~1.0 ns at TT → ~1.5 ns at SS. If sign-off fails at 800 MHz,
# fall back to 667 MHz (1.5 ns) or add pipeline stages in the WHT butterfly.
#
# For other process nodes, scale period:
#   TSMC 28nm HPC+:  target 500 MHz → period 2.0 ns
#   TSMC 65nm:       target 300 MHz → period 3.3 ns
#   Sky130 (OSS):    target  80 MHz → period 12.5 ns (see timing_sky130.sdc)

# Primary clock
create_clock -name clk -period 1.25 [get_ports clk]

# Clock uncertainty (jitter + skew budget)
set_clock_uncertainty 0.1 [get_clocks clk]

# Transition time on clock
set_clock_transition 0.05 [get_clocks clk]

# Input delays — block-level 5% of period (not chip-level 20%)
set all_inputs [get_ports {axil_awaddr* axil_awvalid axil_wdata* axil_wvalid \
    axil_bready axil_araddr* axil_arvalid axil_rready \
    s_axis_kv_tdata* s_axis_kv_tvalid s_axis_kv_tlast s_axis_kv_tuser \
    m_axis_kv_tready}]

set_input_delay  -clock clk -max 0.0625 $all_inputs
set_input_delay  -clock clk -min 0.0    $all_inputs

# Output delays — block-level 5% of period
set all_outputs [get_ports {axil_awready axil_wready axil_bresp* axil_bvalid \
    axil_arready axil_rdata* axil_rresp* axil_rvalid \
    s_axis_kv_tready \
    m_axis_kv_tdata* m_axis_kv_tvalid m_axis_kv_tlast \
    evict_needed evict_addr*}]

set_output_delay -clock clk -max 0.0625 $all_outputs
set_output_delay -clock clk -min 0.0    $all_outputs

# Reset: driven from a slow control plane, treat as false path
set_false_path -from [get_ports rst_n]

# Drive strength of input ports (model upstream driver)
set_driving_cell -lib_cell BUFX4 -pin Z $all_inputs

# Load on output ports (model downstream fanout)
set_load 0.05 $all_outputs
