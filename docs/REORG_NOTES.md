# Reorg notes — block-major → `src/blocks/` taxonomy (2026-07)

**What happened:** the monorepo was restructured from the flat block-major layout
(`kve/ tiu/ acu/ chip/`) to the **`architecture`-repo taxonomy** (`src/blocks/<block>/`), for a
cleaner, more legible root that names every block — including the not-yet-built ones. This is the
structure prototyped in `LonghornSilicon/lambda-test-rebase`, applied to the *real* repo with the
current-latest content and the fixes that probe deliberately skipped.

## Content mapping (all via `git mv` — history preserved, 760 renames)

| Old path | New path | Note |
|---|---|---|
| `kve/` | `src/blocks/kve/` | **kept `kve/`, NOT `kce/`** — see deviation 1 |
| `tiu/` | `src/blocks/tiu/` | |
| `acu/` (mate, vecu, precision_controller, docs, research) | `src/blocks/acu/` | **umbrella nesting preserved** — see deviation 2 |
| `chip/` | `chip/` (unchanged, top level) | integration block |
| — | `src/blocks/{msc,lsu,hif}/` | NEW spec-only stubs (README + arch.yml spec) |
| — | `src/isa/`, `src/golden/` | NEW chip-level index scaffolds |
| — | `tools/`, `tests/integration/` | NEW scaffolds (README stubs) |

## Deviations from the `lambda-test-rebase` probe (deliberate, for correctness)

1. **`kve/` kept, not renamed to `kce/`.** The probe kept `kce` "for HLS continuity" per the
   architecture repo. But lambda's KVE is *RTL*, not HLS; the block *is* the KV Cache Engine; and
   the mirror is `lambda-kve`. `kce` would re-import a name the project already retired. Correct
   name wins over fidelity to the arch stub.
2. **ACU umbrella nesting preserved** (`src/blocks/acu/{mate,vecu,precision_controller}/`) instead
   of flattening mate/vecu to top-level blocks. The probe's flat layout *breaks the `lambda-acu`
   umbrella mirror* — `git subtree split` needs one prefix per mirror, and a flat layout has no
   single prefix covering all three ACU pieces. Nesting keeps all 6 mirrors working with only a
   prefix update, and matches design reality (PC drives per-tile precision across MatE+VecU).
3. **No `SPEC.md` import.** The probe added architecture's per-block spec as `SPEC.md` beside each
   README. That crosses the `architecture`→`lambda` boundary (out of this reorg's scope); deferred
   to the architecture reconcile. Each block keeps its real `README.md`.

## Path-reference integrity (functional correctness)

- **Within-block relative refs are safe.** Each block moved atomically, so a block's
  `pdk/.../config` → `../../../rtl/*.sv` (and the 4-deep sky130 `../../../../rtl/`) still resolve.
  Verified for mate/vecu/kve librelane + openlane configs.
- **Cross-block ref fixed:** `chip/verif/Makefile` `ACUDIR = ../../acu` → `../../src/blocks/acu`
  (the one build-breaking reach).
- **Stale doc/comment paths repointed:** `kve/pdk`, `kve/rtl`, `tiu/pdk`, `acu/mate`, … → `src/blocks/…`
  across `chip/pdk/gf180/{docs,README,PROVENANCE,SUBMISSION}` + `chip/rtl/*.sv` comments
  (guarded replace; verified no corruption).
- **Mirror prefixes updated** in `.github/workflows/mirror-blocks.yml` (`acu`→`src/blocks/acu`, etc.);
  mirror repo names unchanged.
- **Tooling updated** to the new paths (`scripts/*.py` block maps) — `gen_progress.py` and
  `check_block_structure.py` verified green (same 23 macro×PDK rows, same findings).

## Honest follow-ups (NOT done here — tracked, not silently skipped)

1. **Build-green not verified.** The open-PDK flows (OpenLane/LibreLane) and the Cadence chamber
   flow are not runnable in this environment. Structure + path-ref integrity are verified; an actual
   `librelane`/cosim run on the chamber is the real build-green check.
2. **`chip/verif/blocks/{kve,tiu}` still vendors RTL copies** (flagged in `chip/verif/Makefile` +
   `chip/README.md`) — re-point to `src/blocks/{kve,tiu}/rtl` for single-source-of-truth.
3. **`tools/`, `src/isa/`, `src/golden/` are stubs** — populate `tools/` from the architecture repo's
   validated chamber launchers (on branch `rescue/2026-06-audit-uncommitted`); wire chip-level ISA/golden.
4. **`msc/lsu/hif`** are spec-only until their RTL work begins.

## Preserved state

The pre-reorg tree is preserved on branch **`rev-R1`** (at `cb08d1b`) — the R1 baseline snapshot.
This reorg lands on `reorg/arch-structure`; `main` becomes the clean `src/blocks/` scaffolding.
See `docs/REVISION_SYNC_SOP.md` for the revision model.
