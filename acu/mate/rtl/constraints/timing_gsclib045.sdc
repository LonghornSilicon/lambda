# timing_gsclib045.sdc — Cadence Generic Std-Cell Library 045 nm
#
# Targets gsclib045 (v4p4) at the slow corner: VDD=0.9V, T=125C.
# Conservative first-pass period: 4 ns (250 MHz). This is intentionally loose
# so the first synth pass closes cleanly; once we have a baseline QoR we can
# tighten and rerun. gsclib045 is a 45 nm generic kit — published Fmax for
# simple datapaths runs 400-700 MHz at SS, so 250 MHz leaves ~2x of slack
# to absorb tool heuristics on a small block like this.
#
# Sweep recipe once baseline closes:
#   4.0 ns (250 MHz) → 2.5 ns (400 MHz) → 2.0 ns (500 MHz) → 1.5 ns (667 MHz)

create_clock -name clk -period 4.0 [get_ports clk]

set_clock_uncertainty 0.2 [get_clocks clk]
set_clock_transition 0.1 [get_clocks clk]

set_input_delay  -clock clk -max 0.8 [get_ports {s_valid s_data* s_last}]
set_input_delay  -clock clk -min 0.0 [get_ports {s_valid s_data* s_last}]

set_output_delay -clock clk -max 0.8 [get_ports {d_valid d_fp16}]
set_output_delay -clock clk -min 0.0 [get_ports {d_valid d_fp16}]

set_false_path -from [get_ports rst_n]

# Drive strength (gsclib045 ships BUFX1/X2/X4/X8 — BUFX2 is a safe default)
set_driving_cell -lib_cell BUFX2 -pin Y [get_ports {s_valid s_data* s_last}]

set_load 0.05 [get_ports {d_valid d_fp16}]
