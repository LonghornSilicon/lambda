# VecU — Vector Unit (ACU)

Vector-unit slices of the ACU. Three synthesizable blocks today: `vecu_softmax`
(single-row decode online softmax, 64-entry exp-LUT + fp32 accumulator), `rope`
(rotary position embedding, sin/cos LUT + fp16 rotation), and `rmsnorm` (RMSNorm,
fp32 sum-of-squares + rsqrt LUT + per-element gain). `rope`/`rmsnorm` are the raw
Q/K + hidden-state pre-processing the chip-top from-scratch path needs (the cosim's
loaded Qwen tiles are already RoPE'd/normed). A full programmable VecU comes later.

## Layout
- `rtl/vecu_softmax.sv` + `rtl/tb/` (`tb_vecu_softmax.sv`, `gen_vecu_softmax_vectors.py`) — shares the MatE harness (`make -C ../../mate/rtl sim_vecu_softmax`).
- `rtl/rope.sv`, `rtl/rmsnorm.sv` + `rtl/tb/` (`tb_rope.sv`, `tb_rmsnorm.sv`, `gen_*_vectors.py`) + `rtl/Makefile` (`make sim`, `make synth`).
- `sw/reference_model/` — the golden models (`vecu_softmax_ref.py`, `rope_ref.py`, `rmsnorm_ref.py`; the RoPE/RMSNorm goldens reuse the softmax fp16/fp32 bit helpers).
- `pdk/sky130/openlane/{vecu_softmax,rope,rmsnorm}/`, `pdk/gf180/librelane/{vecu_softmax.yaml,rope.yaml,rmsnorm.yaml}` — harden configs (vecu_softmax signed off; rope/rmsnorm staged, P&R run pending).
- `docs/{vecu_softmax,rope,rmsnorm}_rtl.md` — RTL design notes.

## Known gotchas
- **The exp-LUT carries ~2% error** vs exact softmax (64-entry linear interp over [-16,0]) — cosim
  tolerances are set FROM it, not tighter.
- **RoPE/RMSNorm LUTs carry ~7e-4 rel-err** vs exact (cos/sin and rsqrt fp16 quantization) — well
  under the `rel_err < 5e-3` bar, but real error, not zero.
- **All three datapaths are fp32-internal with one fp16 rounding on emit** (no SystemVerilog `real`
  — non-synthesizable). Reuse the shared `fp16↔fp32` / `fp32_add/sub/mul` / `fp32_to_fp16` funcs.
- **The exp/rescale/accumulate (and rope/rmsnorm) chains won't close at the GF180 ss corner unless
  micro-sequenced** — they run one fp32-op/cycle (decode is latency-tolerant, so extra cycles are free).
- **RoPE `pos` is sampled on the FIRST channel** of a vector (constant for the vector); **RMSNorm
  folds `1/D` as a constant** (`INV_D32`, exact for D=16) — regenerate it if `D` changes.
- **The Sky130 GDS is checked in gzipped** (`results/vecu_softmax.gds.gz`) — decompress before use.

See `DECISIONS.md` and `AGENTS.md`.
