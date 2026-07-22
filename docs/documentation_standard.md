# LonghornSilicon — Documentation & ISA Standard

`std-0.1`, 2026-07-18. The conventions every block repo and the architecture hub follow
so a compiler/verification team can build against the accelerator without reading RTL.
If you are standing up a new block, this sits alongside
[`acu/docs/new_block_blueprint.md`](../acu/docs/new_block_blueprint.md)
(which covers the RTL→GDS pipeline); this doc covers what to *write down*.

## 0. Lab-notebook principle (the one rule above all others)

**Treat every repo like a lab notebook: documentation travels with code, in the same
change.** When you alter code, a codec, a spec number, a block name, or a design decision,
you update the affected **README(s), `docs/`, and spec (`arch.yml` / `CLAIMS.md` /
`HW_CONTRACT.md` / paper) in the *same* commit or PR** — never as a "docs later" follow-up.
A README that lies about the current design is worse than none: teammates and the compiler
team act on it. Docs are part of *done*.

Concretely, before you call a substantive change complete: grep the repo **and its sibling
repos** for the fact you changed (old tier, old number, old block name, retired feature) and
update every hit — README, `docs/`, `CLAIMS.md`, `arch.yml`, `HW_CONTRACT.md`, the paper, and
the org profile README. A discovery that spans repos (e.g. a codec change touching both the
KVE and the standalone ChannelQuant repo) updates **both**. The nightly drift check (below)
is a backstop for what slips through, not a substitute for doing this by hand as you go.

## 1. The two-repo model

- **Per-block repos** (`attention-compute-unit`, `kv-cache-engine`,
  `token-importance-unit`, …) own the RTL of record, the block's own ISA/interface spec,
  its reference model, and its paper section.
- The monorepo **`docs/`** hub (this repo, "Lambda") owns the chip-level spec (`arch.yml`), the
  unified ISA (`src/isa/`), the compiler programming guide (`docs/`), the golden
  models index (`src/golden/`), and cross-block reconciliation (`STATUS.md`).

The `docs/` hub **links to** per-block specs; it does not fork them. When a
block-level fact and `arch.yml` disagree, that is a reconciliation item for `STATUS.md`
§7 — flag it, don't silently pick one.

**Automated drift check.** Because the hub restates block facts (codec, tiers, params,
sign-off numbers, precision path), it drifts as blocks change. The
`Architecture Drift Check` workflow (`.github/workflows/architecture-drift-check.yml`)
runs Claude nightly (and on demand) over the block repos vs the hub and opens/updates/
closes a single GitHub issue (`🔄 Architecture drift check`) listing concrete divergences
with suggested fixes. **Setup:** add the repo/org secret `ANTHROPIC_API_KEY`. Block-repo
pushes can additionally trigger it via `repository_dispatch` (`type: block-updated`) once
a cross-repo token is wired. The block repos remain the source of truth — the check
flags stale hub claims for a human to reconcile; it does not auto-edit the spec.

## 2. Required documents per block

Every block ships, in its repo, all of:

| Artifact | Path | Purpose |
|---|---|---|
| **README** | `README.md` | TL;DR table (what/why/how/verified/status), prior-art delta, chip-diagram placement, reproduce steps. |
| **ISA / interface spec** | `docs/isa/<block>_isa.{tex,md,pdf}` | Compiler-facing: block overview, op semantics + latency, AXI-Lite/CSR register map, synth params, change log. Versioned `<block>-isa-X.Y`. |
| **Reference model** | `sw/reference_model/<block>_ref.{py,cpp,hpp}` + `test_*` | Bit-exact vs RTL (and vs each other). The ground truth a compiler develops against. Ships a `test_*` proving parity on golden vectors. |
| **Compiler-use example** | `sw/reference_model/example_compiler_use.py` | Runnable walkthrough of the surface a backend targets. |
| **sw overview** | `docs/sw_overview.{tex,pdf}` | How RTL ⇄ reference model ⇄ test vectors ⇄ compiler entry point fit. |
| **Paper section** | `paper/<block>.{tex,pdf}` | Verification → functional → sweep → Sky130 sign-off → 16FFC projection. |
| **Findings** | `docs/findings/*.md` | Dated, provenance-bearing negative/positive results. |

## 3. Ground-truth principle

**The reference model is authoritative, not the prose.** Interfaces are pre-tape-out and
move; the versioned reference model tracks the RTL and is the contract a compiler tests
against. Every doc that states a numeric interface fact (register offset, format, latency)
should be derivable from, and consistent with, the reference model / `arch.yml`.

## 4. Versioning

- Per-block ISA: `<block>-isa-MAJOR.MINOR` (e.g. `kv-isa-0.2`, `pc-isa-0.1`, `tiu-isa-0.1`).
  MINOR = additive/clarifying; MAJOR = breaking interface change. Record in the doc's
  change log and in `INFO_VERSION`.
- Unified chip ISA: `lh-isa-MAJOR.MINOR` (compiler guide) with sub-versions `lsu-isa-*`,
  `csr-isa-*`, `vecu-isa-*` for the individual headers.
- Never invent hardware numbers (area/power/Fmax/accuracy). Cite the sign-off doc or the
  measurement; mark unmeasured values **TBD**.

## 5. Numbers & provenance

- Accuracy/area/power come from a committed run (Sky130 metrics.json, a HellaSwag/ppl
  harness). Link the artifact.
- Retired approaches stay cited as prior work with a `legacy/*` branch pointer; do not
  delete history, and do not present retired numbers as current.

## 6. Authorship / CI conventions

- Commit author identity and CI runner conventions are in `new_block_blueprint.md`
  (§Gotchas). CI gates (functional / synth FF-count / formal equivalence / Sky130
  sign-off) must be green before a block is "done"; the blueprint's final checklist is
  the bar.
