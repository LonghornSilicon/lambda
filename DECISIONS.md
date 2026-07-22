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
- **chip-level content from `architecture` brought as a curated copy** (partial/selective import —
  cosim → `chip/verif/`, key docs → `docs/`), not a subtree · subtree add is whole-tree and this
  was a selective slice · 2026-07-22.
