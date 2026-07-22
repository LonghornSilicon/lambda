# zcu102.xdc — Timing constraints for precision_controller on Xilinx ZCU102
# Device: xczu9eg-ffvb1156-2-e  (ZCU102)
# Also valid for ZCU104: xczu7ev-ffvc1156-2-e (same UltraScale+ architecture)
#
# Target: 200 MHz (5.0 ns period)
# The precision controller is small combinatorial + register logic.
# It should comfortably meet 250+ MHz on UltraScale+ speed grade -2.

# Primary clock (connect to PS or PL clock in block design)
create_clock -period 5.0 -name clk [get_ports clk]

# Input timing (scores arrive from upstream dot-product pipeline)
# Assume upstream registers output 1ns before clock edge
set_input_delay  -clock clk -max 1.0 [get_ports {s_valid s_data s_last}]
set_input_delay  -clock clk -min 0.2 [get_ports {s_valid s_data s_last}]

# Output timing (d_valid/d_fp16 captured by downstream register)
set_output_delay -clock clk -max 1.0 [get_ports {d_valid d_fp16}]
set_output_delay -clock clk -min 0.2 [get_ports {d_valid d_fp16}]

# Reset is asynchronous from a different domain — treat as false path
set_false_path -from [get_ports rst_n]

# Optional: if clocking from PS (FCLK_CLK0 at 100 MHz), double-check clock divider
# set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk]
