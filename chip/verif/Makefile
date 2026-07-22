# Cross-block RTL cosim (architecture, rtl branch).
# Vendored block RTL lives under blocks/{kve,acu,tiu}/; real-Qwen vectors under vectors/.
# 130nm Sky130 proxy RTL — used for 16nm estimates (see README.md).
IV     = iverilog -g2012
VVP    = vvp
KVE    = blocks/kve/cq_fp_pkg.sv blocks/kve/cq_units.sv blocks/kve/wht_unit.sv \
         blocks/kve/wht_inverse_out.sv blocks/kve/cq_value_path_wht.sv
ACU    = blocks/acu/precision_controller.sv blocks/acu/mate_pv.sv blocks/acu/mate_pv_fp16.sv blocks/acu/mate_qkt.sv blocks/acu/vecu_softmax.sv
TIU    = blocks/tiu/token_importance_unit.sv

.PHONY: cosim clean
cosim:
	$(IV) -I blocks/kve -o build/tb_cosim $(KVE) $(ACU) $(TIU) tb/tb_chip_cosim.sv
	$(VVP) build/tb_cosim

clean:
	rm -rf build *.out *.vcd
