# Lambda — LonghornSilicon decode-attention accelerator (monorepo)

**Lambda** is the LonghornSilicon LLM decode-attention accelerator. This is the **block-major
monorepo**: one clone, atomic cross-block commits, one CI. Each functional block is a
self-contained top-level folder (`sw/ rtl/ pdk/ docs/ research/`); cross-block integration lives
in `chip/`. Each block is also auto-mirrored to a standalone read-only `lambda-<block>` repo so it
keeps its own browsable/cloneable URL.

**Targets — one RTL, multiple PDKs.** The **product target is TSMC 16nm (N16FFC)** — under NDA, so
we prove/estimate on open PDKs. **GF180MCU** is the near-term *chipathon shuttle* (**SSCS Chipathon
2026** — a real open-silicon tapeout of the decode attention datapath Q·Kᵀ → softmax → P·V + KVE +
TIU + ACU gate); **Sky130** is the flagship dev proof; **ASAP7 (7nm FinFET)** is the research
bracket closest to 16nm. GF180/Sky130 are proxies, not the destination — the 16nm hardening is a
separate private overlay pointing at this same RTL. See `AGENTS.md` for the full targets note.

New here? Read **`AGENTS.md`** first (front door + runbook + lab-notebook rules), then
**`DECISIONS.md`** and `docs/`.

## Layout

```
lambda/
├── kve/      KV Cache Engine — ChannelQuant codec (per-channel INT4 K + per-token INT4 V + FP16 outlier lane).
│             Complete block: sw/ rtl/ docs/ research/. Imported with history. → mirror lambda-kve.
├── tiu/      Token Importance Unit — H2O accumulated-mass keep/demote/evict for the KV cache.
│             Complete block. Imported with history. → mirror lambda-tiu.
├── acu/      Attention Compute Unit (MatE + VecU + precision_controller). Imported; all 5 tiles Sky130-signed (see acu/README.md). → mirrors lambda-acu/-mate/-vecu/-precision-controller.
├── chip/     Cross-block integration:
│             ├── verif/     tb_chip_cosim + vendored block RTL + Makefile (the cross-block cosim harness)
│             └── pdk/gf180/  full-chip padring assembly (imported from the former chipathon-lambda-acu; honest skeleton)
├── docs/     Chip-wide: STATUS.md, pdk_holes_audit.md, chipathon_rtl_closure_plan.md,
│             repo_reorg_plan.md, dataflow_walkthrough.md, documentation_standard.md, paper/.
├── research/ Chip-wide research (APA RL project + chip-wide exploration).
├── arch.yml  Machine-readable architecture (blocks, tiles, dataflow).
└── .github/workflows/mirror-blocks.yml  Auto-mirror each block to its standalone repo.
```

## Blocks

Each block folder is self-contained and auto-mirrors to a standalone read-only `lambda-<block>` repo.

| Block | Folder | Mirror repo |
|---|---|---|
| KV Cache Engine (ChannelQuant codec) | [`kve/`](kve/) | [`lambda-kve`](https://github.com/LonghornSilicon/lambda-kve) |
| Token Importance Unit (H2O keep/demote/evict) | [`tiu/`](tiu/) | [`lambda-tiu`](https://github.com/LonghornSilicon/lambda-tiu) |
| Attention Compute Unit (umbrella) | [`acu/`](acu/) | [`lambda-acu`](https://github.com/LonghornSilicon/lambda-acu) |
| &nbsp;&nbsp;├ MatE — Q·Kᵀ + P·V matmul PEs | [`acu/mate/`](acu/mate/) | [`lambda-mate`](https://github.com/LonghornSilicon/lambda-mate) |
| &nbsp;&nbsp;├ VecU — decode online-softmax | [`acu/vecu/`](acu/vecu/) | [`lambda-vecu`](https://github.com/LonghornSilicon/lambda-vecu) |
| &nbsp;&nbsp;└ Precision Controller — INT8/FP16 gate | [`acu/precision_controller/`](acu/precision_controller/) | [`lambda-precision-controller`](https://github.com/LonghornSilicon/lambda-precision-controller) |

Cross-block integration (cosim + full-chip PDK) lives in [`chip/`](chip/); chip-wide docs in
[`docs/`](docs/); chip-wide research in [`research/`](research/).

## Auto-mirror

On every push to `main`, `.github/workflows/mirror-blocks.yml` runs `git subtree split
--prefix=<block>` and force-pushes each block to its standalone **read-only** mirror repo:

| monorepo path | mirror repo | status |
|---|---|---|
| `kve` | [`LonghornSilicon/lambda-kve`](https://github.com/LonghornSilicon/lambda-kve) | active |
| `tiu` | [`LonghornSilicon/lambda-tiu`](https://github.com/LonghornSilicon/lambda-tiu) | active |
| `acu` | [`LonghornSilicon/lambda-acu`](https://github.com/LonghornSilicon/lambda-acu) | active (umbrella) |
| `acu/mate` | [`LonghornSilicon/lambda-mate`](https://github.com/LonghornSilicon/lambda-mate) | active |
| `acu/vecu` | [`LonghornSilicon/lambda-vecu`](https://github.com/LonghornSilicon/lambda-vecu) | active |
| `acu/precision_controller` | [`LonghornSilicon/lambda-precision-controller`](https://github.com/LonghornSilicon/lambda-precision-controller) | active |

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
