# rtl/Makefile — sim, real-data replay, and Yosys synthesis flows.
#
# Run all targets from this directory (rtl/) so $readmemh paths resolve.

PYTHON       ?= python3
RTL          := precision_controller.sv
MATE_RTL     := mate_pv.sv
MATE_FP16_RTL := mate_pv_fp16.sv
MATE_QKT_RTL := mate_qkt.sv
VECU_SM_RTL  := vecu_softmax.sv
TB_DIR       := tb
TV_DIR       := $(TB_DIR)/testvectors
GEN_SCRIPT   := ../analysis/gen_rtl_testvectors.py

SIM_OUT          := sim.out
SIM_REAL_OUT     := sim_realdata.out
SIM_MATE_OUT     := sim_mate.out
SIM_MATE_FP16_OUT := sim_mate_fp16.out
SIM_MATE_QKT_OUT := sim_mate_qkt.out
SIM_VECU_SM_OUT  := sim_vecu_softmax.out
SYNTH_LOG        := synth.log

.PHONY: all sim sim_realdata sim_mate sim_mate_fp16 sim_mate_qkt sim_vecu_softmax testvectors synth clean help

all: sim sim_realdata sim_mate sim_mate_fp16 sim_mate_qkt sim_vecu_softmax

help:
	@echo "Targets:"
	@echo "  sim           — precision_controller directed self-checking TB (110 tests)"
	@echo "  sim_realdata  — replay TB on quantized score tiles + boundary cases"
	@echo "  sim_mate      — mate_pv INT8 P.V MAC bit-exact vs matmul_int8 reference"
	@echo "  sim_mate_fp16 — mate_pv_fp16 FP16 P.V MAC bit-exact vs matmul_fp16 (fp32 accum)"
	@echo "  sim_mate_qkt  — mate_qkt Q.Kt decode-scoring bit-exact vs matmul_fp16 (fp32 accum)"
	@echo "  sim_vecu_softmax — vecu_softmax decode online-softmax bit-exact vs LUT golden"
	@echo "  testvectors   — regenerate scores.hex / expected.hex / labels.txt"
	@echo "  synth         — Yosys area report (FFs + NAND2-equivalent gate count)"
	@echo "  clean         — remove sim binaries, hex vectors, synth logs"

# ---------------------------------------------------------------------------
# mate_pv — INT8 P.V MAC tile (INT32 accumulator), bit-exact to mac_array_ref
# ---------------------------------------------------------------------------

sim_mate: $(SIM_MATE_OUT) $(TB_DIR)/mate_pv_vectors.txt
	cd $(TB_DIR) && ./../$(SIM_MATE_OUT)

$(TB_DIR)/mate_pv_vectors.txt: $(TB_DIR)/gen_mate_pv_vectors.py
	cd $(TB_DIR) && $(PYTHON) gen_mate_pv_vectors.py

$(SIM_MATE_OUT): $(MATE_RTL) $(TB_DIR)/tb_mate_pv.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# mate_pv_fp16 — FP16 P.V MAC tile (fp32 accumulator), bit-exact to matmul_fp16
# ---------------------------------------------------------------------------

sim_mate_fp16: $(SIM_MATE_FP16_OUT) $(TB_DIR)/mate_pv_fp16_vectors.txt
	cd $(TB_DIR) && ./../$(SIM_MATE_FP16_OUT)

$(TB_DIR)/mate_pv_fp16_vectors.txt: $(TB_DIR)/gen_mate_pv_fp16_vectors.py
	cd $(TB_DIR) && $(PYTHON) gen_mate_pv_fp16_vectors.py

$(SIM_MATE_FP16_OUT): $(MATE_FP16_RTL) $(TB_DIR)/tb_mate_pv_fp16.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# mate_qkt — Q.Kt decode-scoring engine (fp32 accumulator), bit-exact to matmul_fp16
# ---------------------------------------------------------------------------

# Runs the decode-scoring TB across three key-counts: L=1 (single key), L=8
# (rich D corner cases incl. the D=520 long reduction), and L=520 (long context /
# wide-lane). Each L regenerates its own budget-bounded vectors and elaborates the
# DUT with a matching lane count (parameter override via iverilog -P).
sim_mate_qkt: $(MATE_QKT_RTL) $(TB_DIR)/tb_mate_qkt.sv $(TB_DIR)/gen_mate_qkt_vectors.py
	@for LN in 1 8 520; do \
	  echo "--- mate_qkt L=$$LN ---"; \
	  (cd $(TB_DIR) && MATE_QKT_L=$$LN $(PYTHON) gen_mate_qkt_vectors.py); \
	  iverilog -g2012 -P tb_mate_qkt.L=$$LN -o $(SIM_MATE_QKT_OUT) $(MATE_QKT_RTL) $(TB_DIR)/tb_mate_qkt.sv; \
	  (cd $(TB_DIR) && ./../$(SIM_MATE_QKT_OUT)) || exit 1; \
	done

# ---------------------------------------------------------------------------
# vecu_softmax — decode online-softmax slice (LUT exp), bit-exact to the golden
# ---------------------------------------------------------------------------

sim_vecu_softmax: $(SIM_VECU_SM_OUT) $(TB_DIR)/vecu_softmax_vectors.txt
	cd $(TB_DIR) && ./../$(SIM_VECU_SM_OUT)

$(TB_DIR)/vecu_softmax_vectors.txt: $(TB_DIR)/gen_vecu_softmax_vectors.py ../sw/reference_model/vecu_softmax_ref.py
	cd $(TB_DIR) && $(PYTHON) gen_vecu_softmax_vectors.py

$(SIM_VECU_SM_OUT): $(VECU_SM_RTL) $(TB_DIR)/tb_vecu_softmax.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Functional simulation (existing testbench)
# ---------------------------------------------------------------------------

sim: $(SIM_OUT)
	./$(SIM_OUT)

$(SIM_OUT): $(RTL) $(TB_DIR)/tb_precision_controller.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Real-data replay
# ---------------------------------------------------------------------------

testvectors: $(TV_DIR)/scores.hex

$(TV_DIR)/scores.hex $(TV_DIR)/expected.hex: $(GEN_SCRIPT)
	$(PYTHON) $(GEN_SCRIPT)

sim_realdata: $(SIM_REAL_OUT) $(TV_DIR)/scores.hex $(TV_DIR)/expected.hex
	./$(SIM_REAL_OUT)

$(SIM_REAL_OUT): $(RTL) $(TB_DIR)/tb_realdata.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Yosys synthesis (area sanity check)
# ---------------------------------------------------------------------------

synth: $(RTL) synth.ys
	yosys -q -l $(SYNTH_LOG) -s synth.ys
	@echo ""
	@echo "Full log: $(SYNTH_LOG)"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	rm -f $(SIM_OUT) $(SIM_REAL_OUT) $(SIM_MATE_OUT) $(SIM_MATE_FP16_OUT) $(SIM_MATE_QKT_OUT) $(SYNTH_LOG)
	rm -f $(TV_DIR)/scores.hex $(TV_DIR)/expected.hex $(TV_DIR)/labels.txt
	rm -f $(TB_DIR)/mate_pv_vectors.txt $(TB_DIR)/mate_pv_fp16_vectors.txt $(TB_DIR)/mate_qkt_vectors.txt $(TB_DIR)/vecu_softmax_vectors.txt
	rm -f $(SIM_MATE_QKT_OUT) $(SIM_VECU_SM_OUT)
