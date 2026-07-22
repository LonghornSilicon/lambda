# ASAP7 (predictive 7nm FinFET) run for the precision controller.
# PREDICTIVE / RESEARCH numbers -- not sign-off. See docs/pdk_bracket_asap7.md.
# Same RTL as the Sky130 sign-off (openlane/precision_controller/config.json).
# No parameter override: the Sky130 config carried no SYNTH_PARAMETERS, so this
# uses the RTL defaults (BLOCK_M=BLOCK_N=64) -- apples-to-apples with Sky130.
export PLATFORM               = asap7

export DESIGN_NAME            = precision_controller
export DESIGN_NICKNAME        = precision_controller

export VERILOG_FILES          = /work/rtl/precision_controller.sv
export SDC_FILE               = /work/orfs/asap7/precision_controller/constraint.sdc

# Floorplan knobs mirror the Sky130 run (FP_CORE_UTIL 50, PL density 60%).
export CORE_UTILIZATION       = 50
export CORE_ASPECT_RATIO      = 1
export CORE_MARGIN            = 2
export PLACE_DENSITY          = 0.60
