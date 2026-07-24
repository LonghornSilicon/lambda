# Lambda — LonghornSilicon decode-attention accelerator (monorepo)

**Lambda** is the LonghornSilicon LLM decode-attention accelerator. This is the **monorepo**: one
clone, atomic cross-block commits, one CI. Each functional block is a self-contained folder under
`src/blocks/<block>/` (`sw/ rtl/ pdk/ docs/ research/`); cross-block integration lives in `chip/`;
the architecture spec / ISA is the standalone [`lambda-arch`](https://github.com/LonghornSilicon/lambda-arch)
repo; Cadence chamber tooling is in `tools/`. Each block auto-mirrors to a read-only `lambda-<block>`
repo so it keeps its own browsable/cloneable URL.

**Branch model — `main` is a clean scaffold; RTL lives on `rev0`.** `main` carries structure, docs,
`pdk/` configs, Python reference-models (the golden spec), tooling, and the proven `results/`
record — but **no `.sv`/`.v` RTL**. RTL is developed on the **`rev0`** revision branch (PR into
`rev0`; a lead blesses → merges to `main`). Longhorn Silicon is a talent-development tapeout —
students and leads write the RTL. **To see/work on RTL: `git checkout rev0`.** Full model:
[`docs/REVISION_SYNC_SOP.md`](docs/REVISION_SYNC_SOP.md) §6a.

**Targets — one RTL, multiple PDKs.** The **product target is TSMC 16nm (N16FFC)** — under NDA, so
we prove/estimate on open PDKs. **GF180MCU** is the near-term *chipathon shuttle* (**SSCS Chipathon
2026** — a real open-silicon tapeout of the decode attention datapath Q·Kᵀ → softmax → P·V + KVE +
TIU + ACU gate); **Sky130** is the flagship dev proof; **ASAP7 (7nm FinFET)** is the research
bracket closest to 16nm. GF180/Sky130 are proxies, not the destination — the 16nm hardening is a
separate private overlay pointing at this same RTL. See `AGENTS.md` for the full targets note.

New here? Read **`AGENTS.md`** first (front door + runbook + lab-notebook rules), then
**`DECISIONS.md`** and `docs/`.

## Layout

Restructured 2026-07 to the **`src/blocks/` taxonomy** (adopted from the `architecture` repo) — one
clean, legible map of every block including the not-yet-built ones. See `docs/REORG_NOTES.md`.

```
lambda/
├── src/
│   ├── blocks/
│   │   ├── acu/        Attention Compute Unit umbrella → mirror lambda-acu
│   │   │   ├── mate/               Q·Kᵀ + P·V matmul PEs         → mirror lambda-mate
│   │   │   ├── vecu/               decode online-softmax/RoPE/RMSNorm → mirror lambda-vecu
│   │   │   └── precision_controller/ INT8/FP16 per-tile gate     → mirror lambda-precision-controller
│   │   ├── kve/        KV Cache Engine — ChannelQuant codec       → mirror lambda-kve
│   │   ├── tiu/        Token Importance Unit — H2O keep/demote/evict → mirror lambda-tiu
│   │   ├── msc/        Memory Subsystem Controller   (spec-only stub)
│   │   ├── lsu/        Layer Sequencer               (spec-only stub)
│   │   └── hif/        Host Interface (PCIe Gen3 x1) (spec-only stub)
│   ├── isa/            Chip-level ISA (LSU opcodes, CSR map); per-block ISA lives in each block
│   └── golden/         Chip-level golden reference + index of per-block reference models
├── chip/     Cross-block integration: verif/ (tb_chip_cosim) + pdk/gf180/ (full-chip padring assembly)
├── tools/    Cadence-chamber launcher framework (reconciled from the architecture repo)
├── tests/integration/  Cross-block cosim + end-to-end decode traces
├── docs/     Chip-wide: STATUS, REVISION_SYNC_SOP, REVISIONS, ROADMAP, PROGRESS (generated),
│             documentation_standard, audits, dataflow_walkthrough, paper/.
├── research/ Chip-wide research (APA RL project + chip-wide exploration).
├── scripts/  Revision-Sync tooling (gen_progress, check_block_structure, cut_revision, rtl_doc_gate).
├── arch.yml  Machine-readable architecture (blocks, tiles, dataflow).
└── .github/workflows/mirror-blocks.yml  Auto-mirror each src/blocks/<block> to its standalone repo.
```

## Blocks

Each block folder is self-contained and auto-mirrors to a standalone read-only `lambda-<block>` repo.

| Block | Folder | Mirror repo |
|---|---|---|
| KV Cache Engine (ChannelQuant codec) | [`src/blocks/kve/`](src/blocks/kve/) | [`lambda-kve`](https://github.com/LonghornSilicon/lambda-kve) |
| Token Importance Unit (H2O keep/demote/evict) | [`src/blocks/tiu/`](src/blocks/tiu/) | [`lambda-tiu`](https://github.com/LonghornSilicon/lambda-tiu) |
| Attention Compute Unit (umbrella) | [`src/blocks/acu/`](src/blocks/acu/) | [`lambda-acu`](https://github.com/LonghornSilicon/lambda-acu) |
| &nbsp;&nbsp;├ MatE — Q·Kᵀ + P·V matmul PEs | [`src/blocks/acu/mate/`](src/blocks/acu/mate/) | [`lambda-mate`](https://github.com/LonghornSilicon/lambda-mate) |
| &nbsp;&nbsp;├ VecU — decode online-softmax | [`src/blocks/acu/vecu/`](src/blocks/acu/vecu/) | [`lambda-vecu`](https://github.com/LonghornSilicon/lambda-vecu) |
| &nbsp;&nbsp;└ Precision Controller — INT8/FP16 gate | [`src/blocks/acu/precision_controller/`](src/blocks/acu/precision_controller/) | [`lambda-precision-controller`](https://github.com/LonghornSilicon/lambda-precision-controller) |
| Memory Subsystem Controller | [`src/blocks/msc/`](src/blocks/msc/) *(spec-only stub)* | — |
| Layer Sequencer | [`src/blocks/lsu/`](src/blocks/lsu/) *(spec-only stub)* | — |
| Host Interface (PCIe Gen3 x1) | [`src/blocks/hif/`](src/blocks/hif/) *(spec-only stub)* | — |

Cross-block integration (cosim + full-chip PDK) lives in [`chip/`](chip/); chip-wide docs in
[`docs/`](docs/); chip-wide research in [`research/`](research/).

## Auto-mirror

On every push to `main` **and** `rev0`, `.github/workflows/mirror-blocks.yml` runs `git subtree
split --prefix=src/blocks/<block>` and force-pushes each block to the **same-named branch** of its
standalone **read-only** mirror — so each mirror has a `main` (scaffold) and a `rev0` (RTL):

| monorepo path | mirror repo | status |
|---|---|---|
| `src/blocks/kve` | [`LonghornSilicon/lambda-kve`](https://github.com/LonghornSilicon/lambda-kve) | active |
| `src/blocks/tiu` | [`LonghornSilicon/lambda-tiu`](https://github.com/LonghornSilicon/lambda-tiu) | active |
| `src/blocks/acu` | [`LonghornSilicon/lambda-acu`](https://github.com/LonghornSilicon/lambda-acu) | active (umbrella) |
| `src/blocks/acu/mate` | [`LonghornSilicon/lambda-mate`](https://github.com/LonghornSilicon/lambda-mate) | active |
| `src/blocks/acu/vecu` | [`LonghornSilicon/lambda-vecu`](https://github.com/LonghornSilicon/lambda-vecu) | active |
| `src/blocks/acu/precision_controller` | [`LonghornSilicon/lambda-precision-controller`](https://github.com/LonghornSilicon/lambda-precision-controller) | active |

Mirrors are **read-only** — open PRs against this monorepo; they propagate out on the next push.

Auto-mirror is **live** — the workflow uses the configured `MIRROR_PAT` repo secret to push each
block to its mirror on every commit to `main`. No setup needed.

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
