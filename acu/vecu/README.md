# VecU — Vector Unit (ACU)

The decode online-softmax slice of the ACU. Today it is exactly `vecu_softmax` (single-row online
softmax with a 64-entry exp-LUT and an fp32 accumulator); a full programmable VecU + RoPE/RMSNorm
come later.

## Layout
- `rtl/vecu_softmax.sv` + `rtl/tb/` (`tb_vecu_softmax.sv`, `gen_vecu_softmax_vectors.py`).
- `sw/reference_model/vecu_softmax_ref.py` — the golden model.
- `pdk/sky130/openlane/vecu_softmax/` — Sky130A sign-off (GDS.gz + metrics).
- `pdk/gf180/librelane/vecu_softmax.yaml` — GF180 tape-out hardening config.
- `docs/vecu_softmax_rtl.md` — RTL design note.

## Known gotchas
- **The exp-LUT carries ~2% error** vs exact softmax (64-entry linear interp over [-16,0]) — cosim
  tolerances are set FROM it, not tighter.
- **The exp/rescale/accumulate chain won't close at the GF180 ss corner unless pipelined** — it is
  pipelined to one fp32-op/cycle (decode is latency-tolerant, so the extra stages are free).
- **The Sky130 GDS is checked in gzipped** (`results/vecu_softmax.gds.gz`) — decompress before use.

See `DECISIONS.md` and `AGENTS.md`.
