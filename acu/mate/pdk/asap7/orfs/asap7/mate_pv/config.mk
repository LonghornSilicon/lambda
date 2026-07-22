# ASAP7 (predictive 7nm FinFET) run for the INT8 P.V MAC tile.
# PREDICTIVE / RESEARCH numbers -- not sign-off. See docs/pdk_bracket_asap7.md.
# Same RTL + same N=4 proxy as the Sky130 sign-off (openlane/mate_pv/config.json).
export PLATFORM               = asap7

export DESIGN_NAME            = mate_pv
export DESIGN_NICKNAME        = mate_pv

# RTL is the identical file the Sky130 flow signed off (mounted at /work).
export VERILOG_FILES          = /work/rtl/mate_pv.sv
export SDC_FILE               = /work/orfs/asap7/mate_pv/constraint.sdc

# N=4 head-dim lanes -- matches SYNTH_PARAMETERS ["N=4"] in the Sky130 config.
export VERILOG_TOP_PARAMS     = N 4

# Floorplan knobs mirror the Sky130 run (FP_CORE_UTIL 50, PL density 60%).
export CORE_UTILIZATION       = 50
export CORE_ASPECT_RATIO      = 1
export CORE_MARGIN            = 2
export PLACE_DENSITY          = 0.60
