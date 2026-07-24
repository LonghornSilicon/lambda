# VecU — Vector Unit (ACU)

Vector-unit slices of the ACU. Three synthesizable blocks today:
- `vecu_softmax` — single-row decode online softmax (64-entry exp-LUT + fp32 accumulator).
- `rope` — rotary position embedding (sin/cos LUT + fp16 rotation).
- `rmsnorm` — RMSNorm (fp32 sum-of-squares + rsqrt LUT + per-element gain).

`rope`/`rmsnorm` are the raw Q/K + hidden-state pre-processing the chip-top from-scratch path needs
(the cosim's loaded Qwen tiles are already RoPE'd/normed). A full programmable VecU comes later.

All three datapaths are fp32-internal with one fp16 rounding on emit, reusing the shared
`fp16↔fp32` / `fp32_add/sub/mul` primitives (no SystemVerilog `real` — non-synthesizable). They are
**bit-exact to their Python LUT goldens** (`sw/reference_model/*_ref.py`), but note the reference
models are **Python-only with no committed parity test** — the RTL tb self-checks against
`gen_*_vectors.py` output; there is no checked-in pytest driving the `_ref.py` models.

## Branch model
`main` is a clean scaffold — **no `.sv`/`.v` RTL**; the RTL lives on the `rev0` revision branch
(PR into `rev0`; leads bless → merge to `main`). To view/work on the blocks below run
**`git checkout rev0`**. Full model: `docs/REVISION_SYNC_SOP.md` §6a.

## Layout — canonical block layout `sw/ rtl/ pdk/ docs/ research/`
- `rtl/vecu_softmax.sv` + `rtl/tb/` (`tb_vecu_softmax.sv`, `gen_vecu_softmax_vectors.py`) — shares the
  MatE harness (`make -C ../../mate/rtl sim_vecu_softmax`).
- `rtl/rope.sv`, `rtl/rmsnorm.sv` + `rtl/tb/` (`tb_rope.sv`, `tb_rmsnorm.sv`, `gen_*_vectors.py`) +
  `rtl/Makefile` (`make sim`, `make synth`).
- `sw/reference_model/` — the golden models (`vecu_softmax_ref.py`, `rope_ref.py`, `rmsnorm_ref.py`;
  the RoPE/RMSNorm goldens reuse the softmax fp16/fp32 bit helpers).
- `pdk/sky130/openlane/{vecu_softmax,rope,rmsnorm}/`, `pdk/gf180/librelane/{vecu_softmax,rope,rmsnorm}.yaml`
  — harden configs + committed sign-off `results/` (`SIGNOFF.md` + metrics JSON + gzipped GDS).
- `docs/{vecu_softmax,rope,rmsnorm}_rtl.md` — RTL design notes.

## Status
Per-block sign-off per PDK. Source: `docs/PROGRESS.md` (generated from committed metrics JSON);
sign-off definitions: `docs/REVISION_SYNC_SOP.md` §5.2.

| Block | Sky130 | GF180 |
|---|---|---|
| `vecu_softmax` | **signed-off** · 9.5 MHz · 726k µm² | config-only |
| `rope` | **signed-off** · 9.5 MHz · 278k µm² | **signed-off** · 3.85 MHz · 432k µm² |
| `rmsnorm` | config-only | **signed-off** · 3.85 MHz · 1.095M µm² |

- **signed-off** — the 5 hard checks (setup/hold/DRC/LVS/antenna) all 0, with a GDS. The loose-clock
  ss-corner register-array max-slew / max-cap residuals are disclosed per-block in `results/SIGNOFF.md`
  (not gating; absorbed by the deliberately loose single-fp32-op-per-cycle clock).
- **config-only** — harden config committed, flow not yet run.

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
- **All committed VecU GDS are checked in gzipped** (`results/*.gds.gz`) — decompress before use.

See `DECISIONS.md` and `AGENTS.md`.
