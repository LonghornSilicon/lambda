# AGENTS.md — TIU (Token Importance Unit)

> **Read this before touching TIU.** Also read `CLAUDE.md` (same content, for Claude Code) and the
> monorepo-root `AGENTS.md`. This file is the front door: context, runbook, lab-notebook rules.

## What this is
**TIU — the Token Importance Unit.** Per-token keep/demote/evict for the KV cache using H2O
accumulated attention mass (post-softmax). Block 3 of the Lambda accelerator; feeds the KVE tier
handshake.

## Before you start — read these (don't skip; they exist so you don't repeat work)
- **`research/`** — design rationale, dead ends, experiments already run (also `analysis/`). If
  about to run an experiment, check here first.
- **`DECISIONS.md`** — settled calls + rationale + date. Don't re-litigate unless the premise changed.
- **`## Known gotchas`** in `README.md` — pitfalls that cost time. Check before debugging.
- **`docs/`** — the tier-handshake doc, findings, and `paper/`.

## Runbook (exact commands — don't re-derive the flow)
```
# from tiu/rtl
make sim            # functional sim
make sim_realdata   # real-Qwen trace parity
make synth          # Yosys synth smoke
# harden
#   librelane pdk/sky130/openlane/token_importance_unit/config.json
```

## Lab-notebook standard — MANDATORY
Same as root `AGENTS.md`: in the SAME commit/PR — (1) docs travel with code, (2) log the decision in
`DECISIONS.md`, (3) log the gotcha in `## Known gotchas`, (4) record the experiment in `research/`,
(5) report honestly with numbers. Full standard: `../docs/documentation_standard.md`.

## Commit conventions
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c user.name=... -c user.email=...`.
This block mirrors to `LonghornSilicon/lambda-tiu` (read-only) — develop in the monorepo.
