# AGENTS.md — KVE (KV Cache Engine)

> **Read this before touching KVE.** Also read `CLAUDE.md` (same content, for Claude Code) and the
> monorepo-root `AGENTS.md`. This file is the front door: context, runbook, lab-notebook rules.

## What this is
**KVE — the ChannelQuant KV-cache codec.** Per-channel INT4 keys (grouped, G=128) + per-token INT4
values + a static top-k FP16 outlier lane (CQ-4+), with an optional WHT value rotation (CQ-3-rot,
flat 3.0 b/val). ~3.8× KV-cache compression, near-lossless. Block 2 of the Lambda accelerator.

## Before you start — read these (don't skip; they exist so you don't repeat work)
- **`research/`** — design rationale, dead ends, experiments already run (also `analysis/`,
  `findings/`, `NOTES.md`). If about to run an experiment, check here first.
- **`DECISIONS.md`** — settled calls + rationale + date. Don't re-litigate unless the premise changed.
- **`## Known gotchas`** in `README.md` — pitfalls that cost time. Check before debugging.
- **`docs/`** — the codec spec, WHT rotation doc, ISA/HW contract.

## Runbook (exact commands — don't re-derive the flow)
```
# from kve/rtl
make sim            # top functional sim
make sim_kpath      # key-path parity
make sim_top        # full-top parity
make sim_wht_syn    # synthesizable WHT butterfly vs behavioral oracle
make sim_wht_pathb  # Path-B (store-rotated / unspin-once) bit-exactness
make synth          # Yosys synth smoke (use the *_syn.sv views — see gotchas)
# harden (Sky130 flagship sign-off)
#   librelane pdk/sky130/openlane/kv_cache_engine/config.json
```

## Lab-notebook standard — MANDATORY
Same as root `AGENTS.md`: in the SAME commit/PR — (1) docs travel with code, (2) log the decision in
`DECISIONS.md`, (3) log the gotcha in `## Known gotchas`, (4) record the experiment in `research/`,
(5) report honestly with numbers. Full standard: `../docs/documentation_standard.md`.

## Commit conventions
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c user.name=... -c user.email=...`.
This block mirrors to `LonghornSilicon/lambda-kve` (read-only) — develop in the monorepo.
