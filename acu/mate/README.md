# MatE — Matmul Engine (ACU)

The matmul tiles of the ACU. Three signed-off logic tiles plus the MAC-array reference model.

| Tile | File | Role |
|---|---|---|
| `mate_qkt` | `rtl/mate_qkt.sv` | Q·Kᵀ decode-scoring (fp16 in, fp32 accumulator) |
| `mate_pv` | `rtl/mate_pv.sv` | INT8 P·V MAC tile |
| `mate_pv_fp16` | `rtl/mate_pv_fp16.sv` | FP16 P·V MAC tile (the FP16 escape path) |

## Layout
- `rtl/` — the three tiles + `tb/` (self-checking tb + `gen_*_vectors.py` golden generators) +
  the **shared multi-tile synth/EDA harness** (`Makefile`, `synth.ys`, `sweep_synth.py`,
  `eda/*.tcl` Cadence stubs, `constraints/`, `tb/tb_realdata.sv`). The harness is parked here because
  MatE is the engine hub; it also drives the vecu/pc tiles.
- `sw/reference_model/` — `mac_array_ref.{py,cpp,hpp}` (8×8 = 64-PE INT8+FP16 matmul) + the shared
  reference-model `Makefile`/`README`/examples + the `sparsity_controller_ref` (a research controller
  that rides with the reference models).
- `pdk/sky130/openlane/{mate_pv,mate_pv_fp16,mate_qkt}` — Sky130A sign-offs (GDS + metrics).
- `pdk/asap7/orfs/asap7/{mate_pv,mate_pv_fp16}` — predictive 7nm bracket (+ the shared `README`/`run_asap7.sh`).
- `pdk/gf180/librelane/{mate_pv,mate_pv_fp16,mate_qkt}.yaml` — GF180 tape-out hardening configs.
- `docs/` — per-tile RTL notes + `mac_array_design.{tex,pdf}`.

## Known gotchas
- **P·V accumulator is INT32, not INT24** — P·V reduces over the token axis; INT24 overflows past
  ~520 flat tokens, INT32 covers ~133k.
- **The 8×8 grid = 64 PEs = 128 GOPS** — an old ref-model default said 16×16/256; that was stale.
- **FP16 can't be bit-exact to numpy `@`** — verify FP16 tiles vs a sequential-fp32 golden (`rel_err < 5e-3`).
- **ASAP7 ORFS is 4×-drawn** — areas read 16× too large unless de-scaled, but this ORFS platform
  already ships the de-scaled 1× LEFs (SITE `0.054×0.270` is the real dimension), so **confirm the
  SITE, don't re-apply /16** (`../docs/pdk_bracket_asap7.md` — de-scale is RESOLVED).

See `DECISIONS.md` and `AGENTS.md`.
