# acu/vecu/rtl/Makefile — sim + Yosys synthesis smoke for the VecU RTL blocks.
#
# Run all targets from this directory (rtl/) so the TB $fopen paths resolve
# (the testbenches read their vector files from tb/).
#
# Blocks:
#   rope      — rotary position embedding (sin/cos LUT, fp16 rotation)
#   rmsnorm   — RMSNorm (fp32 sum-of-squares + rsqrt LUT + per-element gain)
# The decode online-softmax slice (vecu_softmax.sv) shares the MatE harness
# (`make -C ../../mate/rtl sim_vecu_softmax`); its RTL/TB live alongside these.

PYTHON   ?= python3
TB_DIR   := tb

SIM_ROPE_OUT := sim_rope.out
SIM_RMS_OUT  := sim_rmsnorm.out

.PHONY: all sim sim_rope sim_rmsnorm synth synth_rope synth_rmsnorm clean help

all: sim synth

help:
	@echo "Targets:"
	@echo "  sim           — run both bit-exact TBs (rope + rmsnorm)"
	@echo "  sim_rope      — rope bit-exact vs the RoPE LUT golden"
	@echo "  sim_rmsnorm   — rmsnorm bit-exact vs the RMSNorm LUT golden"
	@echo "  synth         — Yosys FF/area report + latch assertion for both"
	@echo "  synth_rope    — Yosys smoke for rope (asserts t:\$$dlatch = 0)"
	@echo "  synth_rmsnorm — Yosys smoke for rmsnorm (asserts t:\$$dlatch = 0)"
	@echo "  clean         — remove sim binaries + generated vectors"

sim: sim_rope sim_rmsnorm

# ---------------------------------------------------------------------------
# rope — bit-exact vs sw/reference_model/rope_ref.py
# ---------------------------------------------------------------------------
sim_rope: $(SIM_ROPE_OUT) $(TB_DIR)/rope_vectors.txt
	cd $(TB_DIR) && ./../$(SIM_ROPE_OUT)

$(TB_DIR)/rope_vectors.txt: $(TB_DIR)/gen_rope_vectors.py ../sw/reference_model/rope_ref.py
	cd $(TB_DIR) && $(PYTHON) gen_rope_vectors.py

$(SIM_ROPE_OUT): rope.sv $(TB_DIR)/tb_rope.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# rmsnorm — bit-exact vs sw/reference_model/rmsnorm_ref.py
# ---------------------------------------------------------------------------
sim_rmsnorm: $(SIM_RMS_OUT) $(TB_DIR)/rmsnorm_vectors.txt
	cd $(TB_DIR) && ./../$(SIM_RMS_OUT)

$(TB_DIR)/rmsnorm_vectors.txt: $(TB_DIR)/gen_rmsnorm_vectors.py ../sw/reference_model/rmsnorm_ref.py
	cd $(TB_DIR) && $(PYTHON) gen_rmsnorm_vectors.py

$(SIM_RMS_OUT): rmsnorm.sv $(TB_DIR)/tb_rmsnorm.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Yosys synthesis smoke (FF/gate counts + t:$dlatch = 0 assertion)
# ---------------------------------------------------------------------------
synth: synth_rope synth_rmsnorm

synth_rope: rope.sv synth_rope.ys
	yosys -q -l synth_rope.log -s synth_rope.ys
	@echo "rope synth OK (no latches) — see synth_rope.log"

synth_rmsnorm: rmsnorm.sv synth_rmsnorm.ys
	yosys -q -l synth_rmsnorm.log -s synth_rmsnorm.ys
	@echo "rmsnorm synth OK (no latches) — see synth_rmsnorm.log"

clean:
	rm -f $(SIM_ROPE_OUT) $(SIM_RMS_OUT) synth_rope.log synth_rmsnorm.log
	rm -f $(TB_DIR)/rope_vectors.txt $(TB_DIR)/rmsnorm_vectors.txt
