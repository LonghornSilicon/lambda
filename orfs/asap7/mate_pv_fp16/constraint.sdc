# ASAP7 timing constraint for mate_pv_fp16 (FP16 P.V MAC).
# ASAP7 liberty time_unit = 1ps, so ALL values here are in PICOSECONDS.
# Clock period (ps) is rewritten per-run by run_asap7.sh so fmax can be swept
# (start loose, tighten to close -- the Sky130 methodology). ORFS's synth step
# greps 'set clk_period <N>' from this file, so N must stay a literal integer.
current_design mate_pv_fp16

set clk_name   clk
set clk_port   [get_ports clk]
set clk_period 3500  ;# picoseconds; rewritten per-run by run_asap7.sh
set clk_io_pct 0.15

create_clock -name $clk_name -period $clk_period $clk_port

# Separate setup/hold uncertainty. Sky130 used a flat 0.1ns; at ASAP7's ~1ns
# clocks a flat 10%-of-period on HOLD (=100ps) dwarfs 7nm min-delays and triggers
# a hold-buffer storm, so hold uses a small absolute margin (5ps skew+jitter) --
# standard practice. Setup keeps a 5%-of-period margin.
set_clock_uncertainty -setup [expr $clk_period * 0.05] [get_clocks $clk_name]
set_clock_uncertainty -hold  5                         [get_clocks $clk_name]

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
