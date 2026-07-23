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

.PHONY: test-smoke test-fullchip test-coproc test-all clean

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus

RTL   = ../../../rtl
BLOCKS = ../../../verif/blocks
ACU    = ../../../../acu

# Hardened decode-attention macro RTL instantiated by lambda_acu (block sources
# of truth: KVE/TIU under chip/verif/blocks, ACU under acu/*/rtl).
KVE_SRC = $(BLOCKS)/kve/cq_fp_pkg.sv $(BLOCKS)/kve/cq_units.sv \
          $(BLOCKS)/kve/wht_unit.sv $(BLOCKS)/kve/wht_inverse_out.sv \
          $(BLOCKS)/kve/cq_value_path_wht.sv
ACU_SRC = $(ACU)/precision_controller/rtl/precision_controller.sv \
          $(ACU)/mate/rtl/mate_pv.sv $(ACU)/mate/rtl/mate_pv_fp16.sv \
          $(ACU)/mate/rtl/mate_qkt.sv $(ACU)/vecu/rtl/vecu_softmax.sv
TIU_SRC = $(BLOCKS)/tiu/token_importance_unit.sv
MACRO_SRC = $(KVE_SRC) $(ACU_SRC) $(TIU_SRC)

# ----- RTL sim -------------------------------------------------------------

# Smoke test: elaborate the workshop-slot chip_core through chip_core_wrap
# (pins the pad widths to SLOT_WORKSHOP 1/20/60) and drive one SPI frame.
test-smoke:
	$(MAKE) -f Makefile.cocotb \
	    TOPLEVEL=chip_core_wrap \
	    MODULE=test_smoke \
	    VERILOG_SOURCES="timescale.v chip_core_wrap.sv $(RTL)/chip_core.sv $(RTL)/lambda_acu.sv $(RTL)/spi_loader.sv $(MACRO_SRC)" \
	    COMPILE_ARGS="-g2012 -I$(BLOCKS)/kve" \
	    SIM_BUILD=sim_build_smoke

# Full-chip decode-attention pass: stream Q/K/V in over SPI, run one decode
# step through the assembled macros, stream the fp32 attention row back out,
# and check it against the host reference (chip/pdk/gf180/tb/test_fullchip.py).
test-fullchip:
	$(MAKE) -f Makefile.cocotb \
	    TOPLEVEL=chip_core_wrap \
	    MODULE=test_fullchip \
	    VERILOG_SOURCES="timescale.v chip_core_wrap.sv $(RTL)/chip_core.sv $(RTL)/lambda_acu.sv $(RTL)/spi_loader.sv $(MACRO_SRC)" \
	    COMPILE_ARGS="-g2012 -I$(BLOCKS)/kve" \
	    SIM_BUILD=sim_build_fullchip

# KV-cache compression coprocessor (KVE cq_value_path_wht_syn + TIU + PC + spi).
KVE_SYN = ../../../../kve/rtl/wht_unit_syn.sv ../../../../kve/rtl/fp16_addsub_syn.sv \
          ../../../../kve/rtl/cq_units_syn.sv ../../../../kve/rtl/cq_value_path_wht_syn.sv
COPROC_SRC = $(KVE_SYN) $(ACU)/precision_controller/rtl/precision_controller.sv \
             ../../../../tiu/rtl/token_importance_unit.sv
test-coproc:
	$(MAKE) -f Makefile.cocotb \
	    TOPLEVEL=chip_core_wrap \
	    MODULE=test_coproc \
	    VERILOG_SOURCES="timescale.v chip_core_wrap.sv $(RTL)/chip_core_kv.sv $(RTL)/lambda_kv_coproc.sv $(RTL)/spi_loader.sv $(COPROC_SRC)" \
	    COMPILE_ARGS="-g2012" \
	    SIM_BUILD=sim_build_coproc

test-all: test-smoke test-fullchip

# ----- GF180 gate-level end-to-end (the full compute-datapath deliverable) --
# Compiles the GF180 LibreLane gate-level netlists (../librelane/runs/<macro>/
# final/nl/<macro>.nl.v) against the gf180mcu_fd_sc_mcu7t5v0 Verilog cell
# models + KVE value-path RTL + the tb_gls_e2e reference TB, and checks each
# block's output matches the reference (INT bit-exact / FP rel_err<5e-3).
# The datapath: mate_qkt -> vecu_softmax -> mate_pv/_fp16, plus ACU + TIU.
# Requires all six compute macros hardened first (scripts/harden.sh).
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
            $(RUNS)/token_importance_unit/final/nl/token_importance_unit.nl.v \
            $(RUNS)/mate_qkt/final/nl/mate_qkt.nl.v \
            $(RUNS)/vecu_softmax/final/nl/vecu_softmax.nl.v

test-gls-e2e:
	iverilog -g2012 -DFUNCTIONAL -I $(BLOCKS)/kve -o sim_gls_e2e.out \
	    timescale.v \
	    $(PDK_VLOG)/primitives.v $(PDK_VLOG)/gf180mcu_fd_sc_mcu7t5v0.v \
	    $(KVE_RTL) $(GL_NL) tb_gls_e2e.sv
	vvp sim_gls_e2e.out

# ----- KVE store round-trip through the REAL GF180 SRAM macro ---------------
# Drives the KVE sram_controller (via the GF180 kv_sram tiling wrapper) wired to
# the real gf180mcu_fd_ip_sram__sram512x8m8wm1 hard-macro SIMULATION MODEL (the
# sign-off view of the IP): writes KV records into real SRAM and reads them back
# bit-exact. Proves the KV storage now lands on real SRAM, not flip-flops.
SRAM_VLOG ?= $(HOME)/.ciel/gf180mcuD/libs.ref/gf180mcu_fd_ip_sram/verilog
test-kv-sram-gf180:
	iverilog -g2012 -o sim_kv_sram_gf180.out \
	    $(BLOCKS)/kve/sram_controller.sv $(BLOCKS)/kve_gf180_sram/kv_sram.sv \
	    $(SRAM_VLOG)/gf180mcu_fd_ip_sram__sram512x8m8wm1.v \
	    tb_kv_sram_gf180.sv
	vvp sim_kv_sram_gf180.out 2>&1 | grep -vE 'WARNING: CEN|MESSAGE: CEN|@Time|scope ='

clean:
	rm -rf sim_build sim_build_smoke results.xml sim_gls_e2e.out sim_kv_sram_gf180.out
