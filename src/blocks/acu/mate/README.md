# MatE — Matmul Engine (ACU)

The matmul tiles of the ACU: the 8×8 systolic MAC array (64 PEs = 128 GOPS) that runs decode
attention's Q·Kᵀ and P·V reductions, plus the MAC-array reference model. Three tiles today.

| Tile | File | What it computes |
|---|---|---|
| `mate_qkt` | `rtl/mate_qkt.sv` | Q·Kᵀ decode scoring — Q(int8→fp16)·K(fp16), **fp32** accumulate, one fp16 round on emit |
| `mate_pv` | `rtl/mate_pv.sv` | INT8 P·V MAC tile — int8×int8, **INT32** accumulator (token-axis reduction) |
| `mate_pv_fp16` | `rtl/mate_pv_fp16.sv` | FP16 P·V MAC tile — the precision-controller FP16 escape path |

INT paths (`mate_pv`, and the int8→fp16 promotion in `mate_qkt`) are **bit-exact** to the reference;
the FP16 paths are **tolerance-checked** (`rel_err < 5e-3`), not bit-exact — see gotchas. Weight/FFN
GEMMs on the same array run INT8×INT4 (W4A8) with INT24 accumulators (hidden-dim reductions fit INT24;
only P·V's token-axis reduction needs INT32 — see `DECISIONS.md`).

## Branch model
`main` is a clean scaffold — **no `.sv`/`.v` RTL**. The RTL lives on the `rev0` revision branch;
contributors PR into `rev0`, leads bless and merge to `main`. To view/work on the tiles below run
**`git checkout rev0`**. Full model: `docs/REVISION_SYNC_SOP.md` §6a.

## Layout — canonical block layout `sw/ rtl/ pdk/ docs/ research/`
- `rtl/` — the three tiles + `tb/` (self-checking tb + `gen_*_vectors.py` golden generators) and the
  **shared multi-tile synth/EDA harness** (`Makefile`, `synth.ys`, `sweep_synth.py`, `eda/*.tcl`
  Cadence stubs, `constraints/`, `tb/tb_realdata.sv`). Parked here because MatE is the engine hub — it
  also drives the vecu tiles (`make -C ../../mate/rtl sim_vecu_softmax`).
- `sw/reference_model/` — `mac_array_ref.{py,cpp,hpp}` (8×8=64-PE INT8+FP16 matmul) + committed parity
  tests (`test_mac_array_ref.{py,cpp}`) + the shared reference-model `Makefile`/`README`/examples + the
  `sparsity_controller_ref` research controller.
- `pdk/sky130/openlane/{mate_pv,mate_pv_fp16,mate_qkt}/` — Sky130A sign-offs (GDS + metrics).
- `pdk/asap7/orfs/asap7/{mate_pv,mate_pv_fp16}/` — predictive 7nm bracket (+ shared `README`/`run_asap7.sh`).
- `pdk/gf180/librelane/{mate_pv,mate_pv_fp16,mate_qkt}.yaml` — GF180 harden configs (declared, not run).
- `docs/` — per-tile RTL notes + `mac_array_design.{tex,pdf}`.

## Status
Per-tile sign-off per PDK. Source: `docs/PROGRESS.md` (generated from committed metrics JSON);
sign-off definitions: `docs/REVISION_SYNC_SOP.md` §5.2.

| Tile | Sky130 | ASAP7 | GF180 |
|---|---|---|---|
| `mate_pv` | **signed-off** · 71 MHz · 75.7k µm² | **route-clean** · 2.0 GHz | config-only |
| `mate_pv_fp16` | **signed-off** · 11.8 MHz · 222k µm² | **route-clean** · 286 MHz | config-only |
| `mate_qkt` | **signed-off** · 12.5 MHz · 199k µm² | — | config-only |

- **signed-off** — Magic-DRC / KLayout-DRC / LVS / antenna / setup / hold all 0, with a GDS.
- **route-clean** — ORFS routed and timing-clean, but the ASAP7 flow runs **no Magic-DRC and no LVS**;
  not a full sign-off, never credited as one.
- **config-only** — harden config committed, flow not yet run.

## Known gotchas
- **P·V accumulator is INT32, not INT24** — P·V reduces over the token axis; INT24 overflows past
  ~520 flat tokens, INT32 covers ~133k.
- **The 8×8 grid = 64 PEs = 128 GOPS** — an old ref-model default said 16×16/256; that was stale.
- **FP16 can't be bit-exact to numpy `@`** — verify FP16 tiles vs a sequential-fp32 golden
  (`rel_err < 5e-3`); BLAS pairwise summation reorders the MACs, so bit-exactness is impossible.
- **ASAP7 ORFS is 4×-drawn** — areas read 16× too large unless de-scaled, but this ORFS platform
  already ships the de-scaled 1× LEFs (SITE `0.054×0.270` is the real dimension), so **confirm the
  SITE, don't re-apply /16** (`../docs/pdk_bracket_asap7.md` — de-scale is RESOLVED).

See `DECISIONS.md` and `AGENTS.md`.
