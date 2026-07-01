IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

# ChannelQuant datapath: top + SRAM shell + the verified compute cores.
# (TurboQuant+ codec — rotation_unit/qjl_unit/quantizer/packer/decompressor/
#  norm_unit — was deleted in the revamp; see legacy/turboquant-plus branch.)
RTL_SRC := kv_cache_engine.sv sram_controller.sv cq_units.sv

TB_DIR  := tb
TV_DIR  := $(TB_DIR)/testvectors

SIM_BIN     := $(TB_DIR)/tb_kv_cache_engine.out
SIM_REAL_BIN := $(TB_DIR)/tb_realdata.out
SIM_CQ_BIN  := $(TB_DIR)/tb_channelquant.out
SIM_AMAX_BIN := $(TB_DIR)/tb_amax_unit.out
SIM_VPATH_BIN := $(TB_DIR)/tb_value_path.out

.PHONY: all sim sim_realdata sim_cq sim_amax sim_vpath testvectors clean

all: sim

sim: $(SIM_BIN)
	$(VVP) $(SIM_BIN)

# ChannelQuant golden-vector parity (contract §8). Self-contained: the TB pulls
# the vendored hex images under tb/testvectors/channelquant/. No RTL_SRC deps yet
# (the datapath modules are wired in as the teardown progresses).
sim_cq: $(SIM_CQ_BIN)
	$(VVP) $(SIM_CQ_BIN)

$(SIM_CQ_BIN): $(TB_DIR)/tb_channelquant.sv cq_fp_pkg.sv cq_units.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_channelquant.sv cq_units.sv

# Streaming amax_unit scale parity vs the golden scales (P2 front-end). Drives
# the synthesizable amax reduction; amax->scale via the proven cq_scale_unit.
sim_amax: $(SIM_AMAX_BIN)
	$(VVP) $(SIM_AMAX_BIN)

$(SIM_AMAX_BIN): $(TB_DIR)/tb_amax_unit.sv amax_unit.sv cq_fp_pkg.sv cq_units.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_amax_unit.sv amax_unit.sv cq_units.sv

# Streaming VALUE-path parity vs golden (scale + payload + V_hat). Integration
# of amax_unit + cq cores as the reusable per-token value datapath.
sim_vpath: $(SIM_VPATH_BIN)
	$(VVP) $(SIM_VPATH_BIN)

$(SIM_VPATH_BIN): $(TB_DIR)/tb_value_path.sv cq_value_path.sv amax_unit.sv cq_fp_pkg.sv cq_units.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_value_path.sv cq_value_path.sv amax_unit.sv cq_units.sv

$(SIM_BIN): $(RTL_SRC) $(TB_DIR)/tb_kv_cache_engine.sv
	$(IVERILOG) -g2012 -o $@ $(TB_DIR)/tb_kv_cache_engine.sv $(RTL_SRC)

sim_realdata: $(SIM_REAL_BIN) testvectors
	cd $(TB_DIR) && $(VVP) tb_realdata.out

$(SIM_REAL_BIN): $(RTL_SRC) $(TB_DIR)/tb_realdata.sv
	$(IVERILOG) -g2012 -o $@ $(TB_DIR)/tb_realdata.sv $(RTL_SRC)

testvectors:
	@if [ ! -f $(TV_DIR)/input_vectors.hex ]; then \
		echo "Generating test vectors..."; \
		$(PYTHON) ../analysis/gen_kv_testvectors.py; \
	fi

synth: synth.ys $(RTL_SRC)
	yosys -s synth.ys

clean:
	rm -f $(TB_DIR)/*.out $(TB_DIR)/*.vcd
