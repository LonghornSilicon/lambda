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

clean:
	rm -rf sim_build sim_build_smoke results.xml
