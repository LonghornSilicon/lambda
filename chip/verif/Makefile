# Cross-block RTL cosim (architecture).
# KVE/TIU RTL is still vendored under blocks/{kve,tiu}/; ACU RTL is pulled directly from the
# canonical block dirs (acu/*/rtl/) — the monorepo keeps ONE copy, so no drift (the vendored
# ACU copies were removed at the 2026-07-22 ACU import; they were byte-identical to canonical).
# 130nm Sky130 proxy RTL — used for 16nm estimates (see README.md).
ACUDIR = ../../acu
IV     = iverilog -g2012
VVP    = vvp
KVE    = blocks/kve/cq_fp_pkg.sv blocks/kve/cq_units.sv blocks/kve/wht_unit.sv \
         blocks/kve/wht_inverse_out.sv blocks/kve/cq_value_path_wht.sv
ACU    = $(ACUDIR)/precision_controller/rtl/precision_controller.sv $(ACUDIR)/mate/rtl/mate_pv.sv \
         $(ACUDIR)/mate/rtl/mate_pv_fp16.sv $(ACUDIR)/mate/rtl/mate_qkt.sv $(ACUDIR)/vecu/rtl/vecu_softmax.sv
TIU    = blocks/tiu/token_importance_unit.sv

.PHONY: cosim clean
cosim:
	$(IV) -I blocks/kve -o build/tb_cosim $(KVE) $(ACU) $(TIU) tb/tb_chip_cosim.sv
	$(VVP) build/tb_cosim

clean:
	rm -rf build *.out *.vcd
