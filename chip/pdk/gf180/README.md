# chip/pdk/gf180 — full-chip GF180MCU padring (PLACEHOLDER — import held)

> **TODO / HELD (2026-07-22).** The full-chip GF180 padring assembly, floorplan, and submit package
> are **not imported yet**. They come from **`chipathon-lambda-acu`**, where an agent is currently
> re-hardening `vecu_softmax`. Importing the PDK/padring bits mid-flight would race that work and
> risk pulling in hand-synced `.sv` copies that are about to change.
>
> Import this once the `chipathon-lambda-acu` agent finishes, alongside the ACU block import
> (see `acu/README.md`).

## What will land here (per `docs/repo_reorg_plan.md`)
- Multi-macro **LibreLane padring** assembly for the SSCS Chipathon 2026 GF180MCU shuttle.
- Full-chip floorplan + the submit package.
- References the block RTL/macros **by path** in the monorepo (no hand-synced copies → no drift).
  In `chipathon-lambda-acu` today these are copied `.sv` files tracked in `PROVENANCE.md`; the
  monorepo keeps ONE authoritative copy per block and this padring points at it.

## Chip-wide GF180 gotchas (already learned — see also root README)
- **gf180 SRAM macro power connects on Metal3.** Route macro power to the M4 straps with a legal
  **Via3** — a Metal1/Metal2 route forces illegal Via1/Via2 stacks (7000+ DRC).
- **The gf180 SRAM abstract has ONE sub-min-width pin** (0.11 µm vs 0.28 µm) — an abstract artifact;
  the vendor GDS is clean. DRC-view-only maglef that widens just that pin; LVS on the real device
  view. (Re-DRC'ing the real GDS throws ~38k false bitcell errors.)
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` disables slew repair** (passes `-slew_margin 0`) — restore ~20%.
