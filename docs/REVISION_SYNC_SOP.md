# Revision-Sync SOP — Lambda

**Status:** of-record · **Version:** sop-1.0 · **Created:** 2026-07-23 · **Owner:** architecture lead
**Scope:** the `lambda/` monorepo and its auto-generated `lambda-<block>` mirrors. **Nothing else.**

> This is the Standard Operating Procedure for **coordinated cross-block revisions**. It exists so
> team leads and contributors can organize versioned RTL changes and baselines across every design
> block *without drift* — one clean, succinct scheme everyone shares. Read `AGENTS.md` (front door +
> lab-notebook rules) and `docs/documentation_standard.md` (per-file doc standard) first; this SOP
> sits on top of them and governs how their outputs get *versioned and synced across blocks*.

---

## 0. Why this exists (the problem it solves)

Before this SOP, the repo had **zero git tags** and **no chip-wide revision object**. Blocks
versioned independently (per-block ISA, and only 3 of 5 blocks even had one), referenced each other
by loose commit SHAs and prose, and chip-level status docs drifted from block reality (e.g. RoPE/
RMSNorm were GDSII-signed in `src/blocks/acu/vecu/DECISIONS.md` while three chip-level docs still said "no
RTL"). There was no defined moment to say "**all blocks, as of here, are baseline N**."

**The key enabler:** `lambda/` is a *monorepo*, so **one annotated git tag atomically pins every
block's exact state at once**. Cross-block revision atomicity is essentially free — this SOP just
adds the discipline (when to tag), the legibility (a human-readable manifest), and the guardrails
(structure + mirror + publish rules) around it.

---

## 1. The revision model

A **revision** is a named, coordinated snapshot of the *entire chip* — all blocks together.

| Kind | Tag | Meaning | Mutability |
|---|---|---|---|
| **Baseline** `RN` (N = 1..4) | `rev-R1` … `rev-R4` | A **frozen** cross-block baseline. A real milestone/baseline was hit. Fully manifested. | **Immutable** once cut. |
| **Assembly** `RN.5` | `rev-R1.5` … `rev-R4.5` | The **in-progress integration** between `RN` and `R(N+1)` — where large changes land and get assembled ("baseline-assembly headroom"). A coherent integration checkpoint / release-candidate. | Re-cuttable (a `.5` may be re-tagged as assembly advances; the *integer* baselines never move). |

There are **four baselines** total (`R1`–`R4`), each with a `.5` assembly step — matching the
project's four coordinated milestones with buffer between each. The `.5` is the safety headroom:
you assemble and stabilize at `RN.5`, then *freeze* into `R(N+1)`.

### 1.1 Baseline → milestone mapping (canonical; see `docs/ROADMAP.md`)

| Rev | Milestone | Track | Target |
|---|---|---|---|
| **R1** | Proxy-PDK **block baseline** — compute + memory blocks signed off on open PDKs; GF180 Chipathon KV-coproc GDS closed | proxy | **now** (2026-07) |
| R1.5 | Full-chip **GF180 assembly** — integrate blocks into full-chip; close remaining per-block GF180 hardening | proxy | 2026 H2 |
| **R2** | Full-chip **GF180 baseline** + N16 design kickoff | proxy → product | ~2026 Q4 |
| R2.5 | **N16 RTL assembly** — port shared RTL onto the N16 flow (private overlay) | product | 2027 H1 |
| **R3** | **N16 RTL design freeze** | product | 2027 Spring |
| R3.5 | **N16 PD assembly** — floorplan freeze, signoff iteration | product | 2027 H2 |
| **R4** | **N16 tapeout** | product | **2027 Fall** (1+ semester buffer) |

The two tracks run **in parallel** — GF180/Chipathon proxy work does not block, and feeds, the
hallmark N16 product design. Mapping is a plan, not a contract: revise here when the roadmap shifts.

### 1.2 What a revision is NOT

- Not a replacement for **per-block ISA versions** (`kv-isa-0.2`, `pc-isa-0.2`, `tiu-isa-0.1`,
  unified `lh-isa-0.1`). Those keep tracking *interface compatibility per block*. A chip revision
  *records* them; it does not supersede them.
- Not the **`arch.yml` doc version** (`metadata.version`), which versions the spec *document*.
- Not a per-block thing. A revision is always **chip-wide** — every block is pinned, even unchanged
  ones. That is the whole point.

---

## 2. Triggers — when to cut a revision

Cut a revision when **any** of these fire (the trigger also sets the version bump):

| Trigger | Example | Cut |
|---|---|---|
| **Baseline hit** | A block reaches clean GDS sign-off on a target PDK; chip integration closes | → advance to next **`RN`** (freeze) |
| **Large change lands** | Codec-of-record change (e.g. the TurboQuant→ChannelQuant pivot), an ISA **MAJOR** bump, floorplan freeze | → **`RN.5`** at minimum; **`RN`** if it defines a baseline |
| **Timeline gate** | A `docs/ROADMAP.md` milestone date is reached | → the mapped **`RN`** |
| **Coordinated integration point** | Multiple blocks changed and need a shared reference for downstream work | → **`RN.5`** |

Routine single-block fixes do **not** cut a revision — they land as normal commits and are swept
into the next `.5`. Cut a revision when *cross-block coordination* or a *milestone* is at stake.

---

## 3. Roles

- **Contributor** (block RTL/sw/pdk author): works on `main` in the monorepo, follows the
  lab-notebook standard (`AGENTS.md` §Lab-notebook), never edits a `lambda-<block>` mirror. Runs
  `scripts/check_block_structure.py` and `scripts/gen_progress.py` before pushing.
- **Block lead**: owns a block's ISA version, DECISIONS log, and sign-off artifacts; declares when
  their block has "hit a baseline" for the next `RN`.
- **Architecture / revision lead**: the only role that **cuts revisions** (`scripts/cut_revision.py`),
  writes the `REVISIONS.md` entry, reconciles cross-block conflicts, and owns `arch.yml` +
  `docs/ROADMAP.md`. Decides `RN` vs `RN.5`.

---

## 4. The revision-cut procedure

Run from a clean `main` in `lambda/`:

```bash
# 0. Preconditions — no drift, no dirty tree
git status --porcelain            # must be empty
python3 scripts/check_block_structure.py     # must pass (or documented waivers)
python3 scripts/gen_progress.py --check       # STATUS progress matrix matches ground-truth artifacts

# 1. Cut — stamps the manifest entry + creates the annotated tag atomically
python3 scripts/cut_revision.py R1 \
    --milestone "Proxy-PDK block baseline; GF180 Chipathon KV-coproc GDS closed"
#   → appends a block to docs/REVISIONS.md (block SHAs, ISA versions, sign-off states)
#   → creates annotated tag rev-R1 on HEAD
#   → prints the manifest block for review

# 2. Review the manifest entry + tag, then push (tag + commit together)
git show rev-R1 --stat
git push origin main --follow-tags
```

The **monorepo tag is the source of truth** — it atomically pins all blocks. The
`lambda-<block>` mirrors inherit the pinned content on the next mirror push (§6); mirror-side tags
are optional and, if used, are stamped by the mirror workflow, never hand-created.

**Never** move an integer baseline tag. If a frozen `RN` was cut in error, cut a corrected `RN.5`
or the next baseline and note the correction in `REVISIONS.md` — the append-only rule from
`DECISIONS.md` applies to the manifest too.

---

## 5. Repo-structure standard (what "clean & succinct across blocks" means)

Every functional block (`kve/`, `tiu/`, `src/blocks/acu/mate/`, `src/blocks/acu/vecu/`, `src/blocks/acu/precision_controller/`)
conforms to the **canonical block template**. `scripts/check_block_structure.py` enforces it.

### 5.1 Canonical block layout
```
<block>/
├── README.md            REQUIRED — overview + ## Known gotchas
├── DECISIONS.md         REQUIRED — append-only, what · why · YYYY-MM-DD (full ISO)
├── AGENTS.md            REQUIRED — block front door (CLAUDE.md may symlink/duplicate it)
├── rtl/                 REQUIRED — *.sv/*.v + tb/  (single source of truth for RTL)
├── sw/reference_model/  REQUIRED — ref model + a parity test + a stated parity count
├── docs/                REQUIRED — incl docs/isa/ for blocks with a programmable interface
├── pdk/                 REQUIRED — pdk/<pdk>/<flow>/<macro>/...
└── research/            REQUIRED — the "why": rationale, dead ends, experiments
```
`chip/` is an **integration block** and is explicitly exempt: it uses `rtl/ verif/ pdk/` and needs
no `sw/ research/` — but it MUST carry a `README.md` and a `DECISIONS.md`.

### 5.2 Sign-off artifact conventions (normalized)
- **One metrics filename schema:** `pdk/<pdk>/<flow>/<macro>/results/<pdk>_signoff_metrics.json`.
  (Historically four conventions existed — `metrics.json`, `signoff_metrics.json`,
  `sky130_71MHz_signoff_metrics.json`, etc. Normalize to the one above; frequency goes *inside*
  the JSON, not the filename.)
- **Sign-off is per-flow.** A macro is **"signed off"** on a PDK only when its metrics JSON shows
  `magic__drc_error__count = 0`, `klayout__drc_error__count = 0`, `design__lvs_error__count = 0`,
  `antenna__violating__nets = 0`, `timing__{setup,hold}_vio__count = 0`, **and** a GDS exists.
  - **ORFS/ASAP7 runs are "route-clean", NOT signed off** — the ASAP7 flow runs no Magic-DRC and
    no LVS step. Report them as `route-clean` only; never credit them as full sign-off.
  - Corner near-misses (ss max-cap / max-slew) that are not in the headline JSON counts MUST be
    disclosed in a `results/SIGNOFF.md` beside the JSON (honest-report rule).
- **A `pdk/.../config.*` with no matching `results/` is "declared, not run"** and is flagged by the
  linter — no orphan build configs, no orphan sign-off *claims* (a gf180 number with no committed
  artifact in the block is a claim, not a result).
- **RTL single-source-of-truth:** the block's `rtl/` is authoritative. PDK flows and `chip/verif/`
  reference it by include path; they do not vendor divergent copies. (`chip/verif/blocks/` currently
  vendors copies — flagged for re-point; see `chip/README.md`.)

---

## 6. Mirror rules (one-directional, read-only)

- Develop **only** in the monorepo. On push to `main`, `.github/workflows/mirror-blocks.yml`
  `subtree split`s each block and **force-pushes** it to its read-only `lambda-<block>` mirror.
- **Never edit, and never clone-as-authoritative, a `lambda-<block>` mirror.** Mirrors trail the
  monorepo by design; a mirror clone can be stale (verified: local clones lagged ~24 commits) or
  carry pre-normalization content. PRs land on the monorepo and propagate out.
- Every **new block** adds a matrix row to `mirror-blocks.yml` (else it never mirrors).
- **Mirror health is monitored** (§8): the push is force-push and fire-and-forget, and has silently
  failed before (a downed self-hosted runner dropped a push). Treat mirror success as *verified*,
  not assumed.

---

## 7. Out-of-band repos (documented boundary — NOT governed by this SOP)

Two sibling repos feed content into `lambda/` but are **outside** the monorepo's atomic revisioning.
They are **reconcile-only** and explicitly out of this SOP's refactor scope:

- **`attention-compute-unit/`** — Chaithu's standalone `master`; upstream of `src/blocks/acu/precision_controller`
  (the `precision_controller.sv` is byte-identical). Already imported (2026-07-22), now overtaken by
  monorepo work. **Frozen upstream.** No auto-sync exists in either direction → any future edit on
  either side diverges silently. Rule: **future precision-controller changes originate in the
  monorepo.** If the standalone must change, re-import and note it in `src/blocks/acu/precision_controller/DECISIONS.md`.
- **`architecture/`** — the ancestor of `arch.yml`; a stale, divergent working fork. Its unique
  uncommitted audit work was **captured non-destructively** on branch `rescue/2026-06-audit-uncommitted`
  (2026-07-23). **Do not blind-sync it** — a fast-forward/re-clone would destroy that work. Any
  reconciliation is a deliberate three-way merge of its unique corrections into `lambda/arch.yml`,
  tracked as its own task. `architecture/src/blocks/{lsu,hif,msc,kce}` holds design content not yet
  in `lambda/` — capture before archiving.

---

## 8. Pre-publish gate — private / NDA research (learned the hard way)

A synthesis agent once auto-published **private KVE research (OliVe/OVP) to the public `lambda`
repo + mirror without authorization**; a security classifier caught it and it was reverted — but it
survives in public git history. To prevent recurrence:

- **Nothing NDA/private reaches `lambda/` or a mirror without an explicit human OK.** The N16 (TSMC)
  PDK, vendor-NDA PHY datasheet numbers, and unpublished research stay in private repos (e.g. the
  private `kv-cache-engine` for KVE research; the private N16 hardening overlay for the PDK).
- Before a push that adds research/data, scan for NDA markers (vendor names + datasheet figures,
  "CONFIDENTIAL", private-repo provenance). When in doubt, it does not go public.
- Public `arch.yml` marks NDA-gated numbers **TBD / estimate with bounds**, never the real vendor
  figure (`AGENTS.md` targets note).

---

## 9. Tooling reference (all local, no CI secrets)

| Tool | Purpose |
|---|---|
| `scripts/gen_progress.py` | Derives the progress matrix from every `*_signoff_metrics.json` → writes/checks the STATUS progress table. **Kills status drift at the root** (status is *computed*, not hand-copied). |
| `scripts/check_block_structure.py` | Lints each block against §5: filename schema, required doc set, ref-model minimum, dangling configs, per-flow sign-off definition. |
| `scripts/cut_revision.py` | Cuts a revision: gathers block SHAs + ISA versions + sign-off states → appends the `REVISIONS.md` entry → creates the annotated `rev-RN[.5]` tag. |
| `.pre-commit-config.yaml` | `rtl`-touched-without-`docs`/`DECISIONS` gate + structure lint on commit. |

CI (GitHub Actions) enforcement is **deliberately deferred** — these run locally + via pre-commit
so the team controls them with no repo secrets. Graduate to CI when the team wants org-wide gating.

---

## 10. Change log
- **2026-07-23** — sop-1.0. Initial SOP: R1–R4 + `.5` revision model, triggers, roles, cut
  procedure, block-structure standard, mirror rules, out-of-band boundary, NDA pre-publish gate.
  Seeded alongside `docs/REVISIONS.md` (R1) and `docs/ROADMAP.md`.
