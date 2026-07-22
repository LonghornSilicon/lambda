<!-- PROTOTYPE / TEMPLATE. At migration, one copy lands at the monorepo root and a per-block
     copy in each block dir (kve/, tiu/, acu/mate/, ...). Replace {{BLOCK}} placeholders.
     This is the FIRST file any human or agent reads. Keep it short and high-signal. -->

# AGENTS.md — {{BLOCK}}

> **Read this before touching {{BLOCK}}.** Also read `CLAUDE.md` if present (same content for
> Claude Code). This file is the front door: it routes you to context, gives the runbook, and
> states the lab-notebook rules you MUST follow. Following them is how we stop re-running
> experiments, re-building blocks, and re-hitting the same walls.

## What this is
{{one line: e.g. "KVE — the ChannelQuant KV-cache codec. Per-channel INT4 K + per-token INT4 V + FP16 outlier lane."}}

## Before you start — read these (don't skip; they exist so you don't repeat work)
- **`research/`** — the "why": design rationale, dead ends, experiments already run. If you're
  about to run an experiment, check here first — it may already be done.
- **`DECISIONS.md`** — settled calls + rationale + date. **Do not re-litigate a settled decision
  unless its stated premise changed.**
- **`## Known gotchas`** in `README.md` — pitfalls that cost someone time. Check before debugging.
- **`docs/`** — the block's spec / ISA / design notes.

## Runbook (exact commands — don't re-derive the flow)
```
# sim / unit test
{{make sim  |  make sim_<target>  |  pytest ...}}
# reference-model parity
{{make ...}}
# harden (per PDK)
{{scripts/harden.sh <block> <pdk>   |   librelane pdk/<pdk>/<block>.yaml}}
# cross-block cosim (chip/)
{{make -C chip/verif}}
```

## Lab-notebook standard — MANDATORY (this is the rule everyone follows)
Every change carries its own record. In the **same commit/PR** as your work:
1. **Docs travel with code.** Touch `rtl/` → update the block `README`/`docs/`. Never leave a repo
   describing something that's no longer true.
2. **Log the decision.** Made a real design/build call? One line in `DECISIONS.md`: *what · why · date*.
3. **Log the gotcha.** Lost time to something surprising (a tool bug, a PDK quirk, an env trap)?
   Add it to `## Known gotchas` so the next person/agent doesn't.
4. **Record the experiment.** Ran a measurement? Record *result · n · artifact · script* in
   `research/` (or a `CLAIMS.md`-style ledger) so it's never re-run to re-learn the answer.
5. **Report honestly.** If something didn't close / a corner failed / a check is waived, say so with
   numbers. A documented near-miss beats a faked pass.

Full standard: `docs/documentation_standard.md`. A PR that changes `rtl/` without touching
`docs/`/`DECISIONS.md` should be flagged by CI.

## Commit conventions
- Author as the project identity (`Chaithu Talasila <themoddedcube@gmail.com>`) via
  `git -c user.name=... -c user.email=...` — NOT your default git config, or commits won't link.
- Block RTL commits go on the block; cross-block/integration on `chip/`.
