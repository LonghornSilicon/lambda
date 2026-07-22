## mmmc.tcl — Multi-Mode Multi-Corner setup for Innovus
## Sourced by innovus.tcl

create_library_set -name ss_lib -timing [list $LIB_SS]
create_library_set -name tt_lib -timing [list $LIB_TT]
create_library_set -name ff_lib -timing [list $LIB_FF]

create_rc_corner -name typ_rc -qx_tech_file $TSMC_QRC_DIR/qrcTechFile_typ
create_rc_corner -name max_rc -qx_tech_file $TSMC_QRC_DIR/qrcTechFile_max
create_rc_corner -name min_rc -qx_tech_file $TSMC_QRC_DIR/qrcTechFile_min

create_delay_corner -name ss_corner -library_set ss_lib -rc_corner max_rc
create_delay_corner -name tt_corner -library_set tt_lib -rc_corner typ_rc
create_delay_corner -name ff_corner -library_set ff_lib -rc_corner min_rc

create_constraint_mode -name func -sdc_files [list constraints/timing.sdc]

create_analysis_view -name view_ss_setup -constraint_mode func -delay_corner ss_corner
create_analysis_view -name view_tt_setup -constraint_mode func -delay_corner tt_corner
create_analysis_view -name view_ff_hold  -constraint_mode func -delay_corner ff_corner

set_analysis_view -setup {view_ss_setup view_tt_setup} -hold {view_ff_hold}
