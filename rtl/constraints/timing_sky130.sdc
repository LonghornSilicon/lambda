# timing_sky130.sdc — SDC timing constraints for Sky130 (OpenLane flow)
#
# Target: Sky130 HD standard cells, 80 MHz = 12.5 ns period
# Sign-off at SS corner (1.60V, -40C) — OpenLane default
#
# This is the tape-out-ready constraint set for the Sky130 shuttle.
# The design uses behavioral SRAM (reg arrays) at this node; real
# SRAM macros are reserved for the 16FFC target.

# Primary clock
create_clock -name clk -period 12.5 [get_ports clk]

# Clock uncertainty (jitter + skew budget)
set_clock_uncertainty 0.1 [get_clocks clk]

# Clock transition
set_clock_transition 0.15 [get_clocks clk]

# Input delays — block-level 5% of period = 0.625 ns
set all_inputs [get_ports {axil_awaddr* axil_awvalid axil_wdata* axil_wvalid \
    axil_bready axil_araddr* axil_arvalid axil_rready \
    s_axis_kv_tdata* s_axis_kv_tvalid s_axis_kv_tlast s_axis_kv_tuser \
    m_axis_kv_tready}]

set_input_delay  -clock clk -max 0.625 $all_inputs
set_input_delay  -clock clk -min 0.0   $all_inputs

# Output delays — block-level 5% of period
set all_outputs [get_ports {axil_awready axil_wready axil_bresp* axil_bvalid \
    axil_arready axil_rdata* axil_rresp* axil_rvalid \
    s_axis_kv_tready \
    m_axis_kv_tdata* m_axis_kv_tvalid m_axis_kv_tlast \
    evict_needed evict_addr*}]

set_output_delay -clock clk -max 0.625 $all_outputs
set_output_delay -clock clk -min 0.0   $all_outputs

# Reset: driven from a slow control plane, treat as false path
set_false_path -from [get_ports rst_n]

# Drive strength (Sky130 HD library)
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 -pin X $all_inputs

# Load on output ports
set_load 0.05 $all_outputs
