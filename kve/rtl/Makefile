IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

# ChannelQuant datapath: top + SRAM shell + the verified compute cores.
# (TurboQuant+ codec — rotation_unit/qjl_unit/quantizer/packer/decompressor/
#  norm_unit — was deleted in the revamp; see legacy/turboquant-plus branch.)
RTL_SRC := kv_cache_engine.sv sram_controller.sv kv_sram.sv cq_units.sv cq_units_syn.sv cq_value_path.sv cq_key_path.sv residual_buffer.sv scale_bank.sv amax_unit.sv

TB_DIR  := tb
TV_DIR  := $(TB_DIR)/testvectors

SIM_BIN     := $(TB_DIR)/tb_kv_cache_engine.out
SIM_REAL_BIN := $(TB_DIR)/tb_realdata.out
SIM_CQ_BIN  := $(TB_DIR)/tb_channelquant.out
SIM_AMAX_BIN := $(TB_DIR)/tb_amax_unit.out
SIM_VPATH_BIN := $(TB_DIR)/tb_value_path.out
SIM_KPATH_BIN := $(TB_DIR)/tb_key_path.out
SIM_TOP_BIN := $(TB_DIR)/tb_top_stream.out
SIM_SYN_BIN := $(TB_DIR)/tb_cq_syn.out

SIM_QWEN_BIN := $(TB_DIR)/tb_qwen_validate.out

.PHONY: all sim sim_realdata sim_cq sim_amax sim_vpath sim_kpath sim_top sim_syn sim_qwen sim_qwen_grid sim_wht sim_wht_syn sim_wht_pathb sim_wht_pathb_syn testvectors clean

# WHT-rotated INT3 value path (branch wht-turboquant-values): cq_wht_value (forward WHT +
# amax + INT3 + dequant + inverse WHT + x1/D) vs the C++/numpy reference V̂ on real Qwen,
# both head dims. Reference vectors: rtl/tb/testvectors/qwen/{g05b,g15b}/multi/vhatwht_*.hex
# (regen: sw/reference_model/test_wht_value_ref for each val_*.hex).
sim_wht:
	$(IVERILOG) -g2012 -DQD=64  -I. -o $(TB_DIR)/tb_wht64.out  cq_fp_pkg.sv cq_units.sv wht_unit.sv cq_wht_value.sv $(TB_DIR)/tb_wht_value.sv
	$(VVP) $(TB_DIR)/tb_wht64.out  +TVDIR=tb/testvectors/qwen/g05b/multi
	$(IVERILOG) -g2012 -DQD=128 -I. -o $(TB_DIR)/tb_wht128.out cq_fp_pkg.sv cq_units.sv wht_unit.sv cq_wht_value.sv $(TB_DIR)/tb_wht_value.sv
	$(VVP) $(TB_DIR)/tb_wht128.out +TVDIR=tb/testvectors/qwen/g15b/multi

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

$(SIM_VPATH_BIN): $(TB_DIR)/tb_value_path.sv cq_value_path.sv amax_unit.sv cq_fp_pkg.sv cq_units.sv cq_units_syn.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_value_path.sv cq_value_path.sv amax_unit.sv cq_units.sv cq_units_syn.sv

# Validate the value-path RTL against a REAL Qwen2 slice: replays tb/testvectors/
# qwen_v.hex through cq_value_path and checks V̂ bit-for-bit vs qwen_vhat_hw.hex (the
# fp16-exact codec, analysis/channelquant_hw.py). Regenerate the slice with:
#   python analysis/channelquant_hw.py --mode dump-slice
sim_qwen: $(SIM_QWEN_BIN)
	$(VVP) $(SIM_QWEN_BIN)

$(SIM_QWEN_BIN): $(TB_DIR)/tb_qwen_validate.sv cq_value_path.sv amax_unit.sv cq_fp_pkg.sv cq_units.sv cq_units_syn.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_qwen_validate.sv cq_value_path.sv amax_unit.sv cq_units.sv cq_units_syn.sv

# Full real-Qwen GRID: value AND key path, D=64 (0.5B) and D=128 (1.5B) — 696,320
# elements bit-exact. Regenerate the vectors with:
#   python analysis/channelquant_hw.py --mode dump-multi --model Qwen/Qwen2-0.5B --outdir ../rtl/tb/testvectors/qwen/g05b
#   python analysis/channelquant_hw.py --mode dump-multi --model Qwen/Qwen2-1.5B --outdir ../rtl/tb/testvectors/qwen/g15b
sim_qwen_grid:
	$(IVERILOG) -g2012 -DQD=64  -I. -o $(TB_DIR)/tb_qwen_v64.out  $(TB_DIR)/tb_qwen_multi.sv cq_value_path.sv amax_unit.sv cq_units.sv cq_units_syn.sv
	$(VVP) $(TB_DIR)/tb_qwen_v64.out  +TVDIR=tb/testvectors/qwen/g05b/multi
	$(IVERILOG) -g2012 -DQD=128 -I. -o $(TB_DIR)/tb_qwen_v128.out $(TB_DIR)/tb_qwen_multi.sv cq_value_path.sv amax_unit.sv cq_units.sv cq_units_syn.sv
	$(VVP) $(TB_DIR)/tb_qwen_v128.out +TVDIR=tb/testvectors/qwen/g15b/multi
	$(IVERILOG) -g2012 -I. -o $(TB_DIR)/tb_qwen_key.out $(TB_DIR)/tb_qwen_key.sv cq_key_path.sv residual_buffer.sv scale_bank.sv amax_unit.sv cq_units.sv cq_units_syn.sv
	$(VVP) $(TB_DIR)/tb_qwen_key.out +TVDIR=tb/testvectors/qwen/g05b/multi
	$(VVP) $(TB_DIR)/tb_qwen_key.out +TVDIR=tb/testvectors/qwen/g15b/multi


# Synthesis-tier WHT: fp16_addsub_syn (fp16 add/sub, no `real`) vs the behavioral oracle
# (cq_fp_pkg::fp16_add/sub) on a 375k-pair sweep, then wht_unit_syn vs behavioral wht_unit
# on real Qwen rows. The synthesizable butterfly for tape-out (cf. cq_units_syn.sv).
sim_wht_syn:
	$(IVERILOG) -g2012 -I. -o $(TB_DIR)/tb_fp16as.out cq_fp_pkg.sv fp16_addsub_syn.sv $(TB_DIR)/tb_fp16_addsub.sv
	$(VVP) $(TB_DIR)/tb_fp16as.out
	$(IVERILOG) -g2012 -DQD=64  -I. -o $(TB_DIR)/tb_whtsyn64.out  cq_fp_pkg.sv fp16_addsub_syn.sv wht_unit.sv wht_unit_syn.sv $(TB_DIR)/tb_wht_syn.sv
	$(VVP) $(TB_DIR)/tb_whtsyn64.out  +VAL=tb/testvectors/qwen/g05b/multi/val_0.hex
	$(IVERILOG) -g2012 -DQD=128 -I. -o $(TB_DIR)/tb_whtsyn128.out cq_fp_pkg.sv fp16_addsub_syn.sv wht_unit.sv wht_unit_syn.sv $(TB_DIR)/tb_wht_syn.sv
	$(VVP) $(TB_DIR)/tb_whtsyn128.out +VAL=tb/testvectors/qwen/g15b/multi/val_0.hex


# Path B (CQ-3-rot): cq_value_path_wht (KVE emits ROTATED fp16) -> wht_inverse_out (inverse
# WHT + x1/D on the P·V output) == the full reference V̂ on real Qwen. Proves the once-per-
# decode inverse split (store rotated, unspin once) is bit-exact to the full codec.
sim_wht_pathb:
	$(IVERILOG) -g2012 -DQD=64  -I. -o $(TB_DIR)/tb_pathb64.out  cq_fp_pkg.sv cq_units.sv wht_unit.sv wht_inverse_out.sv cq_value_path_wht.sv $(TB_DIR)/tb_wht_pathb.sv
	$(VVP) $(TB_DIR)/tb_pathb64.out  +TVDIR=tb/testvectors/qwen/g05b/multi
	$(IVERILOG) -g2012 -DQD=128 -I. -o $(TB_DIR)/tb_pathb128.out cq_fp_pkg.sv cq_units.sv wht_unit.sv wht_inverse_out.sv cq_value_path_wht.sv $(TB_DIR)/tb_wht_pathb.sv
	$(VVP) $(TB_DIR)/tb_pathb128.out +TVDIR=tb/testvectors/qwen/g15b/multi

# Path B, SYNTHESIZABLE lowering: cq_value_path_wht_syn -> wht_inverse_out_syn (no `real`,
# on wht_unit_syn/cq_units_syn/fp16_addsub_syn) == the reference V̂, bit-exact. This is the
# hardware the GF180 full-chip build synthesizes (lambda_acu `ifdef LAMBDA_SYN_KVE). The
# behavioral cq_value_path_wht/wht_inverse_out are compiled in only as the sim oracle deps.
# NOTE: the fp16 bit-emulation makes this ~100x slower than the behavioral run in Icarus —
# D=64 (80 tokens, 5120 elems) takes ~10 min; expect it. Result: 5120/5120 bit-exact.
sim_wht_pathb_syn:
	$(IVERILOG) -g2012 -DQD=64  -I. -o $(TB_DIR)/tb_pathbsyn64.out  cq_fp_pkg.sv cq_units.sv cq_units_syn.sv wht_unit.sv wht_unit_syn.sv fp16_addsub_syn.sv wht_inverse_out.sv cq_value_path_wht.sv cq_value_path_wht_syn.sv wht_inverse_out_syn.sv $(TB_DIR)/tb_wht_pathb_syn.sv
	$(VVP) $(TB_DIR)/tb_pathbsyn64.out  +TVDIR=tb/testvectors/qwen/g05b/multi
	$(IVERILOG) -g2012 -DQD=128 -I. -o $(TB_DIR)/tb_pathbsyn128.out cq_fp_pkg.sv cq_units.sv cq_units_syn.sv wht_unit.sv wht_unit_syn.sv fp16_addsub_syn.sv wht_inverse_out.sv cq_value_path_wht.sv cq_value_path_wht_syn.sv wht_inverse_out_syn.sv $(TB_DIR)/tb_wht_pathb_syn.sv
	$(VVP) $(TB_DIR)/tb_pathbsyn128.out +TVDIR=tb/testvectors/qwen/g15b/multi

# Streaming KEY-path parity vs golden (per-channel scales + INT4 payload + K_hat
# + sidecar). Exercises residual_buffer + amax_unit + scale_bank + cq cores.
sim_kpath: $(SIM_KPATH_BIN)
	$(VVP) $(SIM_KPATH_BIN)

$(SIM_KPATH_BIN): $(TB_DIR)/tb_key_path.sv cq_key_path.sv residual_buffer.sv scale_bank.sv amax_unit.sv cq_fp_pkg.sv cq_units.sv cq_units_syn.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_key_path.sv cq_key_path.sv residual_buffer.sv scale_bank.sv amax_unit.sv cq_units.sv cq_units_syn.sv

# P4b equivalence: synthesizable cores (cq_units_syn.sv) vs behavioral oracle
# (cq_units.sv), bit-exact over an fp16 sweep. See rtl/P4B_PLAN.md.
sim_syn: $(SIM_SYN_BIN)
	$(VVP) $(SIM_SYN_BIN)

$(SIM_SYN_BIN): $(TB_DIR)/tb_cq_syn.sv cq_units_syn.sv cq_fp_pkg.sv cq_units.sv
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_cq_syn.sv cq_units_syn.sv cq_units.sv

# Top-level end-to-end parity through the AXI interfaces (per-token codec in the
# top FSM + SRAM). CQ-8 K+V vs golden expected_*_hat.
sim_top: $(SIM_TOP_BIN)
	$(VVP) $(SIM_TOP_BIN)

$(SIM_TOP_BIN): $(TB_DIR)/tb_top_stream.sv $(RTL_SRC)
	$(IVERILOG) -g2012 -I. -o $@ $(TB_DIR)/tb_top_stream.sv $(RTL_SRC)

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
