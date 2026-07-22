# timing.sdc — SDC timing constraints for Cadence Genus/Innovus (ASIC flow)
#
# Primary target:
#   TSMC 16FFC (16nm FinFET Compact) via TSMC University Program
#   Target: 800 MHz = 1.25 ns period, signed off at SS 0.72V 125C
#
# Justification: ASAP7 (open 7nm proxy) measured 430 ps critical path on the
# Q→sum_acc→comparator→d_fp16 chain. Scaling to 16FFC TT corner (~2x slower
# transistors) gives ~860 ps. SS worst-case ~1300 ps — at the edge of 1.25 ns.
# If sign-off slack is negative at SS we can either (a) lower target to
# 700 MHz (1.43 ns), or (b) add one register stage between sum_next and the
# 24-bit comparator (cuts critical path ~50%, costs ~24 extra FFs).
#
# For other process nodes, scale period:
#   TSMC 28nm HPC+:  target 500 MHz → period 2.0 ns
#   TSMC 65nm:       target 300 MHz → period 3.3 ns
#   Sky130 (OSS):    target 100 MHz → period 10 ns

# Primary clock
create_clock -name clk -period 1.25 [get_ports clk]

# Clock uncertainty (jitter + skew budget)
set_clock_uncertainty 0.1 [get_clocks clk]

# Transition time on clock
set_clock_transition 0.05 [get_clocks clk]

# Input delays (from upstream pipeline register to this module's input FF)
set_input_delay  -clock clk -max 0.3 [get_ports {s_valid s_data* s_last}]
set_input_delay  -clock clk -min 0.0 [get_ports {s_valid s_data* s_last}]

# Output delays (from this module's output FF to downstream register)
set_output_delay -clock clk -max 0.3 [get_ports {d_valid d_fp16}]
set_output_delay -clock clk -min 0.0 [get_ports {d_valid d_fp16}]

# Reset: driven from a slow control plane, treat as false path
set_false_path -from [get_ports rst_n]

# Drive strength of input ports (model upstream driver)
set_driving_cell -lib_cell BUFX4 -pin Z [get_ports {s_valid s_data* s_last}]

# Load on output ports (model downstream fanout)
set_load 0.05 [get_ports {d_valid d_fp16}]

# Operating conditions (set to match your PDK worst-case library)
# set_operating_conditions -library <lib_name> -condition SS_0P9V_125C
