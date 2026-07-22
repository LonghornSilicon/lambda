# Lambda — LonghornSilicon decode-attention accelerator (monorepo)

**Lambda** is the LonghornSilicon LLM decode-attention accelerator. This is the **block-major
monorepo**: one clone, atomic cross-block commits, one CI. Each functional block is a
self-contained top-level folder (`sw/ rtl/ pdk/ docs/ research/`); cross-block integration lives
in `chip/`. Each block is also auto-mirrored to a standalone read-only `lambda-<block>` repo so it
keeps its own browsable/cloneable URL.

Tape-out target: **SSCS Chipathon 2026 on GF180MCU** — the decode attention datapath
(Q·Kᵀ → softmax → P·V + KVE + TIU + ACU gate). Sky130 is the flagship/dev-vehicle proof.

New here? Read **`AGENTS.md`** first (front door + runbook + lab-notebook rules), then
**`DECISIONS.md`** and `docs/`.

## Layout

```
lambda/
├── kve/      KV Cache Engine — ChannelQuant codec (per-channel INT4 K + per-token INT4 V + FP16 outlier lane).
│             Complete block: sw/ rtl/ docs/ research/. Imported with history. → mirror lambda-kve.
├── tiu/      Token Importance Unit — H2O accumulated-mass keep/demote/evict for the KV cache.
│             Complete block. Imported with history. → mirror lambda-tiu.
├── acu/      Attention Compute Unit (MatE + VecU + precision_controller). HELD — placeholder (see acu/README.md).
├── chip/     Cross-block integration:
│             ├── verif/     tb_chip_cosim + vendored block RTL + Makefile (the cross-block cosim harness)
│             └── pdk/gf180/  full-chip padring assembly — HELD placeholder (imports from chipathon-lambda-acu)
├── docs/     Chip-wide: STATUS.md, pdk_holes_audit.md, chipathon_rtl_closure_plan.md,
│             repo_reorg_plan.md, dataflow_walkthrough.md, documentation_standard.md, paper/.
├── research/ Chip-wide research (APA RL project + chip-wide exploration).
├── arch.yml  Machine-readable architecture (blocks, tiles, dataflow).
└── .github/workflows/mirror-blocks.yml  Auto-mirror each block to its standalone repo.
```

## Auto-mirror

On every push to `main`, `.github/workflows/mirror-blocks.yml` runs `git subtree split
--prefix=<block>` and force-pushes each block to its standalone **read-only** mirror repo:

| monorepo path | mirror repo | status |
|---|---|---|
| `kve` | [`LonghornSilicon/lambda-kve`](https://github.com/LonghornSilicon/lambda-kve) | active |
| `tiu` | [`LonghornSilicon/lambda-tiu`](https://github.com/LonghornSilicon/lambda-tiu) | active |
| `acu` | `LonghornSilicon/lambda-acu` | HELD (row commented out until ACU is imported) |
| `acu/mate` | `LonghornSilicon/lambda-mate` | HELD |
| `acu/vecu` | `LonghornSilicon/lambda-vecu` | HELD |
| `acu/precision_controller` | `LonghornSilicon/lambda-precision-controller` | HELD |

Mirrors are **read-only** — open PRs against this monorepo; they propagate out on the next push.

### ⚠️ Required user action — `MIRROR_PAT` secret (one-time)

The mirror workflow pushes to **other** repos, which the default `GITHUB_TOKEN` cannot do. It needs
a repo secret **`MIRROR_PAT`**: a fine-grained Personal Access Token with **`contents: write`** on
the mirror repos (`lambda-kve`, `lambda-tiu`, and the future ACU mirrors). Create it, then:

```
gh secret set MIRROR_PAT --repo LonghornSilicon/lambda   # paste the PAT when prompted
```

Until this secret exists, the auto-mirror job will fail at the push step. (The mirrors have already
been **seeded manually** once so they are non-empty and browsable — see the reorg plan.)

## Known gotchas (chip-wide) — check before debugging

**Environment / flow**
- **LHS box venv is read-only, no numpy/pip.** Use `/home/shadeform/cuda_advisor/.venv/bin/python`
  for numpy; reinstall `iverilog`/`yosys` each session. Prefer pure-Python golden generators.
- **ORFS ASAP7 is 4×-drawn.** Areas read 16× too large unless de-scaled — confirm the SITE size
  (`0.054×0.270`) before quoting µm².

**GF180 hardening**
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` DISABLES slew repair** (it passes `-slew_margin 0`) — an inverted
  setting. Restore ~20% or you get thousands of false max-slew/cap violations.
- **LibreLane escaped-identifier instance naming** takes 3 different forms across
  `PDN_MACRO_CONNECTIONS` regex / `instances` placement / YAML quoting — match each exactly.

Block-specific gotchas live in each block's `README.md` (`kve/README.md`, `tiu/README.md`).

## License / contributing
Develop in this monorepo. The `lambda-<block>` repos are read-only mirrors — do not push to them.
