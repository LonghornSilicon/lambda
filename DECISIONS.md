<!-- Chip-wide settled calls. Append-only; never delete, mark superseded. Format: what · why · date.
     Per-block decisions live in <block>/DECISIONS.md. Seeded from docs/prototypes/DECISIONS.seed.md
     (the 2026-07 work) at monorepo creation 2026-07-22. -->

# DECISIONS — chip-wide (do not re-litigate unless the premise changed)

- **Tapeout = SSCS Chipathon 2026 on GF180MCU** (LibreLane multi-macro padring), NOT Sky130 · the
  2026 shuttle PDK is GF180 · 2026-07-21. Sky130 stays as the flagship/dev-vehicle proof.
- **Tapeout boundary = the decode attention datapath** (Q·Kᵀ→softmax→P·V + KVE + TIU + ACU gate) ·
  it's the coherent completion of the RTL we actually built; projections/FFN GEMMs were never RTL
  and run off-chip for the shuttle · 2026-07-21.
- **Repo structure = monorepo `lambda`, block-major, auto-mirror per block** · one source of truth,
  self-contained blocks, standalone mirror repos with no drift · 2026-07-22. See
  `docs/repo_reorg_plan.md`.
- **Online-softmax citation = Milakov & Gimelshein 2018**, not FlashAttention-3 · the recurrence
  predates FA; FA added the tiling · 2026-07-21.
- **Commit identity = `themoddedcube@gmail.com`** via `git -c` · other emails don't link to the
  GitHub account · 2026-07-20.

## Migration log (this monorepo)
- **kve/ and tiu/ imported with history via `git subtree add`** (rtl+master unioned per block) ·
  2026-07-22. rtl was authoritative (all RTL/openlane + current docs); master contributed its
  unique docs/analysis files; shared-doc conflicts resolved in favor of rtl.
- **ACU import HELD** · `attention-compute-unit` (Sky130 sign-offs) and `chipathon-lambda-acu`
  (vecu_softmax re-harden) were mid-flight · `acu/` and `chip/pdk/gf180/` are documented
  placeholders with TODOs; ACU-level decisions parked in `acu/DECISIONS.md` · 2026-07-22.
  · **SUPERSEDED — ACU import DONE 2026-07-22** (`edaaa41`): `acu/{mate,vecu,precision_controller}`
  imported block-major with real `rtl/ sw/ pdk/` sign-off results (all 5 tiles Sky130-signed);
  `chip/pdk/gf180/` holds real content (LibreLane configs, real `gf180mcu_fd_ip_sram` macro, GLS
  report), not placeholders; `acu/DECISIONS.md` now holds settled calls. Root README/AGENTS state
  "Imported; all 5 tiles Sky130-signed."
- **chip-level content from `architecture` brought as a curated copy** (partial/selective import —
  cosim → `chip/verif/`, key docs → `docs/`), not a subtree · subtree add is whole-tree and this
  was a selective slice · 2026-07-22.
- **kve/ and tiu/ PDK dirs normalized to the block-major `pdk/{sky130,gf180}/` convention** (match
  ACU) · every block now has an identical `sw/ rtl/ pdk/ docs/ research/` shape · 2026-07-23.
  `kve/openlane`→`kve/pdk/sky130/openlane`, `tiu/openlane`→`tiu/pdk/sky130/openlane`; the per-block
  GF180 configs left in `chip/pdk/gf180/librelane/` moved into their blocks
  (`kve/pdk/gf180/librelane/{kve,kve_store_gf180}.yaml`+`pdn_cfg_sram.tcl`, the `kve_gf180_sram`
  real-SRAM wrapper → `kve/pdk/gf180/`, `token_importance_unit.yaml`→`tiu/pdk/gf180/librelane/`).
  `chip/pdk/gf180/` keeps only integration/chip assets (PROVENANCE/README/SUBMISSION/docs/scripts/tb
  + `chip/rtl/`). All history-preserving `git mv`; moved-config `VERILOG_FILES` now resolve to the
  block's own `rtl/` (this also fixed a latent bug: `kve.yaml` had pointed at `chip/verif/blocks/kve/`
  which was missing kv_cache_engine/sram_controller/kv_sram.sv); `scripts/harden.sh` now locates a
  block's config by macro name.
