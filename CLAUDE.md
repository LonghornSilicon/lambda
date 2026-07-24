# AGENTS.md — Lambda (monorepo root)

> **Read this before touching the repo.** Also read `CLAUDE.md` (same content, for Claude Code).
> This file is the front door: it routes you to context, gives the runbook, and states the
> lab-notebook rules you MUST follow. Following them is how we stop re-running experiments,
> re-building blocks, and re-hitting the same walls.

## What this is
**Lambda** — the LonghornSilicon decode-attention accelerator, as a **monorepo with a `src/blocks/`
taxonomy** (reorg 2026-07; see `docs/REORG_NOTES.md`). Each functional block is a self-contained
folder under `src/blocks/<block>/` holding all its aspects (`sw/ rtl/ pdk/ docs/ research/`).
Cross-block integration lives in `chip/`. The architecture spec / ISA is the standalone
[`lambda-arch`](https://github.com/LonghornSilicon/lambda-arch) repo (see `src/arch/`); the Cadence
chamber tooling lives in `tools/`.

**Branch model — `main` is a clean scaffold; RTL lives on revision branches.** `main` carries the
structure, docs, `pdk/` configs, Python reference-models (the golden spec), tooling, and the proven
`results/` record — but **no `.sv`/`.v` RTL**. The RTL is developed by contributors on the `rev0`
revision branch (PR into `rev0`); a lead **blesses** it (reviewed, sign-off reproduced) and merges
it upstream into `main`. This is deliberate — Longhorn Silicon is a talent-development tapeout, so
students and leads write the RTL. Full model: `docs/REVISION_SYNC_SOP.md` §6a. **To see/work on RTL:
`git checkout rev0`.**

**Targets — one RTL, multiple PDKs.** The **product target is TSMC 16nm (N16FFC)**. That PDK is
under NDA, so we prove/estimate on open PDKs: **GF180MCU** is the near-term *chipathon shuttle*
(SSCS Chipathon 2026 — a real open-silicon tapeout of the decode attention datapath
Q·Kᵀ→softmax→P·V + KVE + TIU + ACU gate); **Sky130** is the flagship dev proof; **ASAP7 (7nm
FinFET)** is the research bracket closest to 16nm. Open-PDK hardening lives in each block's
`pdk/{sky130,gf180,asap7}/` **folders** — coexisting, referencing the one shared RTL, **never
branches**. The **16nm hardening is a separate PRIVATE overlay** (NDA — TSMC PDK files can't live
in a public repo); it points at this same RTL once we have PDK access. GF180/Sky130 are proxies,
not the destination.

## Repo map (src/blocks/ taxonomy — reorg 2026-07; see docs/REORG_NOTES.md)
```
lambda/
├── src/blocks/
│   ├── acu/     ← Attention Compute Unit umbrella (mate + vecu + precision_controller). Mirrors lambda-acu/-mate/-vecu/-precision-controller.
│   │   ├── mate/                 Q·Kᵀ + P·V matmul PEs
│   │   ├── vecu/                 decode softmax / RoPE / RMSNorm
│   │   └── precision_controller/ INT8/FP16 per-tile gate
│   ├── kve/     ← KV Cache Engine (ChannelQuant codec). Complete block: sw/ rtl/ pdk/ docs/ research/. Mirrors lambda-kve.
│   ├── tiu/     ← Token Importance Unit (H2O keep/demote/evict). Complete block. Mirrors lambda-tiu.
│   └── msc/ lsu/ hif/  ← spec-only stubs (not-yet-built blocks; README + arch.yml spec).
├── src/arch/    ← chip-level ISA (LSU opcodes, CSR map). Per-block ISA lives in each block's docs/isa/.
├── src/golden/ ← chip-level golden reference + index of per-block reference models.
├── chip/       ← cross-block integration: verif/ (tb_chip_cosim), pdk/gf180/ (full-chip padring assembly).
├── tools/      ← Cadence-chamber launcher framework (reconciled from the architecture repo).
├── tests/integration/ ← cross-block cosim + end-to-end decode traces.
├── docs/       ← chip-wide: STATUS, REVISION_SYNC_SOP, REVISIONS, ROADMAP, PROGRESS (generated), standard, audits, paper/.
├── research/   ← chip-wide research (APA RL project + exploration notes).
├── scripts/    ← Revision-Sync tooling (gen_progress, check_block_structure, cut_revision, rtl_doc_gate).
├── arch.yml    ← the machine-readable architecture (blocks, tiles, dataflow).
└── .github/workflows/mirror-blocks.yml ← auto-mirror each src/blocks/<block> to its standalone lambda-<block> repo.
```

## Before you start — read these (don't skip; they exist so you don't repeat work)
- **`<block>/research/`** — the "why": design rationale, dead ends, experiments already run.
- **`DECISIONS.md`** (root = chip-wide; per block = `<block>/DECISIONS.md`) — settled calls + why +
  date. **Do not re-litigate a settled decision unless its stated premise changed.**
- **`## Known gotchas`** in each `README.md` — pitfalls that cost someone time. Check before debugging.
- **`docs/`** — chip-wide spec / audit / plans. `arch.yml` is the machine-readable arch.

## Runbook (exact commands — don't re-derive the flow)
```
# RTL work happens on the rev0 revision branch (main is scaffold, no .sv):
git checkout rev0
# cross-block cosim (the integration harness)
make -C chip/verif cosim
# per-block sim / parity (from inside the block, e.g. src/blocks/kve/rtl or src/blocks/tiu/rtl)
make -C src/blocks/kve/rtl sim   # (see the block's own AGENTS.md for its exact targets)
make -C src/blocks/tiu/rtl sim
# harden a block for a PDK (block-local pdk/ config)
#   librelane src/blocks/<block>/pdk/<pdk>/<block>.yaml   (see block AGENTS.md)
```
Environment note: the LHS box venv is read-only (no pip/numpy). Reinstall `iverilog`/`yosys`
each session; use `/home/shadeform/cuda_advisor/.venv/bin/python` for numpy. See root
`## Known gotchas` in `README.md`.

## Lab-notebook standard — MANDATORY (this is the rule everyone follows)
Every change carries its own record. In the **same commit/PR** as your work:
1. **Docs travel with code.** Touch `rtl/` → update the block `README`/`docs/`. Never leave a repo
   describing something that's no longer true.
2. **Log the decision.** Made a real design/build call? One line in `DECISIONS.md`: *what · why · date*.
3. **Log the gotcha.** Lost time to something surprising? Add it to `## Known gotchas`.
4. **Record the experiment.** Ran a measurement? Record *result · n · artifact · script* in
   `research/` (or a `CLAIMS.md`-style ledger) so it's never re-run to re-learn the answer.
5. **Report honestly.** If something didn't close / a corner failed / a check is waived, say so with
   numbers. A documented near-miss beats a faked pass.

Full standard: `docs/documentation_standard.md`. A PR that changes `rtl/` without touching
`docs/`/`DECISIONS.md` should be flagged by CI.

## Monorepo + mirrors
Develop **here** (atomic cross-block commits, one CI). On push to `main` **and** `rev0`,
`.github/workflows/mirror-blocks.yml` `subtree split`s each `src/blocks/<block>` dir and
force-pushes it to the same-named branch of a read-only `lambda-<block>` mirror — so each mirror has
a `main` (scaffold) and a `rev0` (RTL), matching this monorepo. Mirrors are **read-only** — PRs land
here and propagate out. The mirror job needs the repo secret `MIRROR_PAT` (configured).

## Commit conventions
- **Author as yourself** — your own name + GitHub email, via `git -c user.name=... -c user.email=...`
  or your git config. Attribution matters: contributors own the work they do.
- RTL work + block commits go on `rev0` (the revision branch); the SOP §6a describes how a blessed
  revision merges to `main`. Never merge `main` → a revision branch (it would delete the RTL).
