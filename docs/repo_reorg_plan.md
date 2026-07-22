# Repo Reorg Plan — Monorepo + Auto-Mirror

**Drafted 2026-07-22.** Goal: one **monorepo** for build/integration ergonomics, PLUS a
**standalone browsable/cloneable repo per block** (read-only auto-mirror) so we keep the
"open it up and see it" property. Prototype mirror workflow: `docs/prototypes/mirror-blocks.yml`.

## Why this shape (the "have both")

- **Monorepo** = one clone, atomic cross-block commits, one CI, trivial cosim/integration
  (`lambda_acu` top pulls MatE+VecU+KVE+PC from sibling dirs, not from pinned submodule SHAs).
- **Auto-mirror** = a CI job `git subtree split`s each block dir and force-pushes it to a
  standalone `lambda-<block>` repo → each block still has its own GitHub URL + README + clone,
  just read-only. PRs land on the monorepo; mirrors follow.
- Net: the developer experience is a monorepo; the *public face* is still per-block repos.

## Step 0 — Clean up the ACU repo FIRST (it's the messy one)

KVE and TIU are clean (born as block repos). The `attention-compute-unit` repo is messy because
it's still the original **"Adaptive Precision Attention" RL research project** with hardware
grafted on. Root-level clutter to resolve:

| Item | What it is | Disposition |
|---|---|---|
| `phase1_policy/`, `phase2_kernel/` | RL policy evolution + evolved kernel (research) | → `research/` (or a separate `adaptive-precision-attention-research` repo) |
| `kv_cache/`, `common/` | Python KV-cache + shared prototypes (research) | → `research/` |
| `run_phase1.sh`, `run_phase2.sh`, `requirements.txt` | research runners | → `research/` |
| `analysis/` (48 files) | benchmarks / sweeps (FA-2 compare, entropy) | → `research/analysis/` (keep — it backs the paper) |
| `paper/` | the APA method paper | keep (research artifact) |
| `rtl/`, `openlane/`, `orfs/`, `sw/reference_model/`, `docs/isa/` | **the actual hardware** | **keep — this is the ACU** |

**Target (matches KVE/TIU neatness):** repo root = `README.md`, `rtl/`, `sw/`, `openlane/`,
`orfs/`, `docs/`, `analysis/` (light), `research/` (the archived RL project), `.github/`.
Also: rename the README from "Adaptive Precision Attention" → "ACU — Attention Compute Unit"
and flatten the stray EDA tcl (`genus*.tcl`, `innovus.tcl`, `mmmc.tcl`) into `rtl/eda/` or drop
the unused Cadence stubs (we sign off with LibreLane/OpenLane, not Genus/Innovus).

**⚠️ Sequencing:** a Sky130-sign-off agent is *live in this repo right now* (adding
`openlane/mate_qkt/` + `openlane/vecu_softmax/`). **Do the ACU cleanup only after it finishes** —
concurrent file moves + its commits = merge pain. Cleanup is the first executed step, but it
waits on that agent.

## Target monorepo layout (`lambda`)

**BLOCK-MAJOR** — each block is a self-contained top-level folder holding *all* its aspects
(python sw, rtl, pdk, docs, research). This is the OpenTitan pattern (`hw/ip/<block>/{rtl,dv,doc}`),
and it makes each block's mirror a *complete* repo. A `chip/` folder holds the cross-block work that
belongs to no single block.

```
lambda/
├── kve/                          # ← kv-cache-engine, wholesale (one self-contained block)
│   ├── sw/                       #   Python reference / golden models
│   ├── rtl/                      #   SystemVerilog + tb
│   ├── pdk/{sky130,gf180}/       #   hardening configs + results for THIS block
│   ├── docs/
│   └── research/                 #   design notes / dead ends / benchmarks (LLM/agent context)
├── tiu/                          # ← token-importance-unit, wholesale
│   └── sw/ rtl/ pdk/ docs/ research/
├── acu/                          # multi-block unit → sub-block folders (each self-contained)
│   ├── mate/                     #   sw/ rtl/ pdk/ docs/ research/   (mate_pv, _fp16, _qkt)
│   ├── vecu/                     #   sw/ rtl/ pdk/ docs/ research/   (vecu_softmax, +rope/rmsnorm)
│   ├── precision_controller/     #   sw/ rtl/ pdk/ docs/ research/
│   ├── docs/  research/          #   ACU-level
│   └── README.md
├── chip/                         # cross-block integration — belongs to NO single block
│   ├── rtl/                      #   lambda_acu top, chip_core, spi_loader
│   ├── verif/                    #   tb_chip_cosim (the cross-block cosim)
│   ├── pdk/gf180/                #   padring assembly + full-chip floorplan + submit package
│   └── docs/
├── docs/                         # chip-wide: arch.yml, papers, the audit + these plans
├── research/                     # chip-wide research (the APA RL project)
├── README.md
└── .github/workflows/            # CI + mirror-blocks.yml
```

Why block-major here (vs aspect-major `rtl/ pdk/ sw/`): (1) **mirrors become complete blocks** —
`kve/ → lambda-kve` ships rtl + python + pdk + docs + research together, exactly the "open it and
see it" + LLM-context goal; (2) **migration is trivial** — each current repo already *is* a block
folder (`sw/ rtl/ docs/`), so it drops in wholesale; (3) it's the proven chip-repo pattern.

**Mirror map** (per functional block — every block, incl. TIU; extend the row list as new blocks land):

| monorepo path | mirror repo | level |
|---|---|---|
| `acu` | `lambda-acu` | **umbrella** — assembled ACU (mate + vecu + pc) |
| `acu/mate` | `lambda-mate` | piece (complete: sw+rtl+pdk+docs+research) |
| `acu/vecu` | `lambda-vecu` | piece |
| `acu/precision_controller` | `lambda-precision-controller` | piece |
| `kve` | `lambda-kve` | block (complete) |
| `tiu` | `lambda-tiu` | block (complete) |
| `chip` | `lambda-chip` | integration (optional) |
| *(future)* `msc`, `lsu`, `hif` | `lambda-msc`, … | block |

Because it's block-major, each mirror carries the block's **whole** story (python + rtl + pdk +
docs + research), not just its RTL.

**Nested mirrors are fine and drift-free.** `subtree split` is per-prefix, so `lambda-acu` (the
whole `rtl/acu/`) and `lambda-mate` (`rtl/acu/mate/`) are both read-only projections of the *same*
source tree. MatE's files appearing in both is not drift — it's one authoritative copy seen through
two windows. The umbrella "shows the assembled ACU"; the pieces show focused blocks.

**Copy-drift elimination (a real benefit, not just tidiness):** the `chipathon-lambda-acu` repo today
holds **hand-synced `.sv` copies** of every block (tracked in `PROVENANCE.md`) — they can silently
drift from the source repos. In the monorepo there is **one** copy of each block; `pdk/gf180/` and
`pdk/sky130/` reference it by path. The drift hazard goes away entirely.

## Agent context & lab-notebook layer (decision #6, 2026-07-22)

Goal: stop agents/people re-running experiments, re-building blocks, and re-hitting the same walls.
`research/` + lab-notebook are the foundation but don't produce a high-signal "don't repeat" surface
on their own. Add a **thin, always-read layer** (NOT a parallel system — sprawl rots):

- **`AGENTS.md`** (root + per block; template: `docs/prototypes/AGENTS.md`) — the front door every
  human/agent reads first: routes to `research/`/`DECISIONS.md`/gotchas, gives the exact runbook,
  and states the lab-notebook rules. Mirror to `CLAUDE.md` for Claude Code.
- **`DECISIONS.md`** (per block + chip-wide, append-only) — settled calls + why + date, so they
  aren't re-litigated. **Seeded** from this session: `docs/prototypes/DECISIONS.seed.md`.
- **`## Known gotchas`** section in each block README — pitfalls that cost time (seeded above).
- **Experiment ledger** in `research/` (or `CLAIMS.md`-style) — result · n · artifact · script, so
  measurements aren't re-run. ChannelQuant's `CLAIMS.md` is the precedent.

Three genres, three kinds of repeated work prevented: **decision log** (re-litigation), **gotchas**
(re-hitting walls), **experiment ledger** (re-running). Lab-notebook is the *discipline* that keeps
all three current; `AGENTS.md` is the *door* that makes them discoverable; formalize the rule in
`docs/documentation_standard.md` and back it with a light CI check (PR touching `rtl/` must touch
`docs/`/`DECISIONS.md`).

## Migration — least-friction, history-preserving

**Principle:** never copy-paste files (loses history + blame). Import each repo *with history*
into its monorepo path using `git subtree add` or (cleaner) `git filter-repo --to-subdirectory-filter`.

1. **Freeze** — land the in-flight work first (KVE PDN sign-off, mate_qkt/vecu_softmax Sky130,
   vecu_softmax GF180 re-harden). Migrating mid-flight multiplies conflicts.
2. **Clean the ACU repo** (Step 0) on its `rtl` branch; commit.
3. **Seed the monorepo** from `architecture` (it already holds docs/arch/cosim + is the doc hub) —
   or a fresh `lambda` repo. Move its own content into the target dirs.
4. **Import each block with history** into its path:
   `git filter-repo --to-subdirectory-filter rtl/kve` on a clone of kv-cache-engine, then pull it
   in; repeat for tiu, acu, and the GF180 PDK work. (Each block's full commit history is preserved
   under its new prefix.)
5. **Re-point flows** — the cosim `Makefile` include paths, the OpenLane/LibreLane `VERILOG_FILES`
   (`dir::...`), the ASAP7 ORFS `config.mk`, and CI. This is the bulk of the mechanical work.
6. **Stand up CI + the mirror** — port each repo's `.github` CI into the monorepo; add
   `mirror-blocks.yml` + create the empty `lambda-<block>` mirror repos + the `MIRROR_PAT` secret.
   First mirror push seeds them.
7. **Retire the old repos** — archive them (GitHub "Archive") with a README pointer to the monorepo,
   OR let the mirrors *become* them (point people at the read-only mirrors). Keep the git history —
   don't delete.

## Decisions — CONFIRMED 2026-07-22

1. **Monorepo home:** a **fresh repo named `lambda`** (final). Family symmetry with the per-block
   mirror repos (`lambda-mate`, `lambda-vecu`, `lambda-kve`, `lambda-tiu`, …).
2. **Research:** keep it **as a `research/` subdir** (NOT archived, NOT a branch, NOT its own repo).
   Two levels: a top-level `research/` in `lambda` (the APA RL project + chip-wide research), AND a
   **`research/` subdir inside each block dir**. Rationale (Chaithu): the per-block `research/` is
   **context for future LLM/agents** — design rationale, dead ends, benchmarks, exploration notes —
   so that anyone (human or agent) starting new work on that block, or a new related project, inherits
   the "why," not just the RTL. It rides along in the block's mirror repo, so the context is wherever
   the block is.
3. **`rtl` layout:** **subdirs for multi-block units, flat for single blocks.** `rtl/acu/` gets
   `mate/` + `vecu/` + `precision_controller/`; `rtl/kve/` and `rtl/tiu/` stay flat (one block each,
   even if many files). A unit gains subdirs only if it later holds multiple distinct blocks.
4. **Structure = BLOCK-MAJOR** (revised 2026-07-22 per Chaithu): each block is a self-contained
   top-level folder holding all its aspects — `<block>/{sw, rtl, pdk, docs, research}` — *not* an
   aspect-major `rtl/ pdk/ sw/` top level. Within a block, `pdk/` splits per target
   (`pdk/sky130/`, `pdk/gf180/`, `pdk/asap7/` — we test on multiple PDKs, each its own folder).
   Cross-block work (the integration top, the cosim, the full-chip padring) lives in a top-level
   `chip/` folder. All on `main` (directories, not branches). This makes each block's mirror a
   *complete* repo and matches how the current repos are already laid out (near-zero-friction move).
5. **Mirror policy:** **every functional block gets its own mirror repo** — and every *new* block we
   make adds a mirror row. Granularity = the architecture's functional blocks (MatE, VecU, KVE, TIU,
   precision-controller, + future MSC/LSU/HIF), matching how we name blocks — not the flat leaf tiles
   (`mate_pv` etc. live *inside* `lambda-mate`). Each block dir carries its `research/` + `docs/` so
   the mirror is self-describing.

## Risks / notes

- `git subtree split` re-walks history each mirror run — fine here; swap to `josh` if it ever slows.
- Mirror repos are **read-only**; document that in each mirror's README so contributors go to the mono.
- The cutover should be **one atomic reorg**, not piecemeal, to avoid a long window of broken paths.
- Nothing is moved until the in-flight sign-offs land and the four decisions above are made.
