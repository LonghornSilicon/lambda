# Token Importance Unit — RTL build/sim
IVERILOG = iverilog
VVP      = vvp
YOSYS    = yosys

RTL      = token_importance_unit.sv
SIM_BIN  = build/tb_tiu

.PHONY: all sim synth clean
all: sim

build:
	mkdir -p build

sim: build
	$(IVERILOG) -g2012 -o $(SIM_BIN) $(RTL) tb/tb_token_importance_unit.sv
	$(VVP) $(SIM_BIN)

# Yosys synth + FF-count report (matches the closed-form header comment)
synth: build
	$(YOSYS) -p "read_verilog -sv $(RTL); \
	             synth -top token_importance_unit -flatten; \
	             stat" 2>&1 | tee build/synth.log

clean:
	rm -rf build
