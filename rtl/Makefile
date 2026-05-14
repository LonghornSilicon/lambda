IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

RTL_SRC := kv_cache_engine.sv norm_unit.sv rotation_unit.sv quantizer.sv \
           qjl_unit.sv packer.sv sram_controller.sv decompressor.sv

TB_DIR  := tb
TV_DIR  := $(TB_DIR)/testvectors

SIM_BIN     := $(TB_DIR)/tb_kv_cache_engine.out
SIM_REAL_BIN := $(TB_DIR)/tb_realdata.out

.PHONY: all sim sim_realdata testvectors clean

all: sim

sim: $(SIM_BIN)
	$(VVP) $(SIM_BIN)

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
