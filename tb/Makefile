# cocotb test dispatcher for the Lambda ACU chipathon submission.
#
# Mirrors the sscs-chipathon-2026 multi-macro template's tb/Makefile.
#
# RTL sim (pre-synthesis):
#   make test-smoke        # elaborate chip_core + one SPI START frame
#   make test-all
#
# Per-macro tests (added as each macro's RTL is wired into the datapath):
#   make test-kve          # TODO
#   make test-tiu          # TODO
#   make test-pc           # TODO
#   make test-pv           # TODO
#   make test-pv-fp16      # TODO
#
# Gate-level sim (post-LibreLane, against gf180 cells) — TODO, follows the
# template's test-*-gl pattern once macros are hardened.
#
#   make clean
#
# Requires:  cocotb, icarus-verilog  (both in hpretl/iic-osic-tools).

.PHONY: test-smoke test-all clean

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus

RTL   = ../rtl
BLOCKS = $(RTL)/blocks

# ----- RTL sim -------------------------------------------------------------

# Smoke test: elaborate the workshop-slot chip_core through chip_core_wrap
# (pins the pad widths to SLOT_WORKSHOP 1/20/60) and drive one SPI frame.
test-smoke:
	$(MAKE) -f Makefile.cocotb \
	    TOPLEVEL=chip_core_wrap \
	    MODULE=test_smoke \
	    VERILOG_SOURCES="timescale.v chip_core_wrap.sv $(RTL)/chip_core.sv $(RTL)/lambda_acu.sv $(RTL)/spi_loader.sv" \
	    COMPILE_ARGS="-g2012" \
	    SIM_BUILD=sim_build_smoke

test-all: test-smoke

# ----- GF180 gate-level end-to-end (the Stage-1 deliverable) ---------------
# Compiles the GF180 LibreLane gate-level netlists (../librelane/runs/<macro>/
# final/nl/<macro>.nl.v) against the gf180mcu_fd_sc_mcu7t5v0 Verilog cell
# models + KVE value-path RTL + the tb_gls_e2e reference TB, and checks each
# block's output matches the reference (INT bit-exact / FP rel_err<5e-3).
# Requires the four macros hardened first (scripts/harden.sh).
#
#   PDK_VLOG=<...>/gf180mcu_fd_sc_mcu7t5v0/verilog make test-gls-e2e
PDK_VLOG ?= $(HOME)/.ciel/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/verilog
RUNS      = ../librelane/runs
KVE_RTL   = $(BLOCKS)/kve/cq_fp_pkg.sv $(BLOCKS)/kve/cq_units.sv \
            $(BLOCKS)/kve/wht_unit.sv $(BLOCKS)/kve/cq_value_path_wht.sv \
            $(BLOCKS)/kve/wht_inverse_out.sv
GL_NL     = $(RUNS)/mate_pv/final/nl/mate_pv.nl.v \
            $(RUNS)/mate_pv_fp16/final/nl/mate_pv_fp16.nl.v \
            $(RUNS)/precision_controller/final/nl/precision_controller.nl.v \
            $(RUNS)/token_importance_unit/final/nl/token_importance_unit.nl.v

test-gls-e2e:
	iverilog -g2012 -DFUNCTIONAL -I $(BLOCKS)/kve -o sim_gls_e2e.out \
	    timescale.v \
	    $(PDK_VLOG)/primitives.v $(PDK_VLOG)/gf180mcu_fd_sc_mcu7t5v0.v \
	    $(KVE_RTL) $(GL_NL) tb_gls_e2e.sv
	vvp sim_gls_e2e.out

clean:
	rm -rf sim_build sim_build_smoke results.xml sim_gls_e2e.out
