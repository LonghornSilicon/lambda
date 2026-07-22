# AGENTS.md — ACU (Attention Compute Unit)

> **Read this before touching the ACU.** It is the front door: it routes you to context and states
> the lab-notebook rules you MUST follow (so nobody re-runs experiments, re-builds blocks, or
> re-hits the same walls).

## What this is
The decode-attention datapath: `Q·Kᵀ → softmax → P·V` with a per-tile INT8/FP16 gate. It splits into
three self-contained sub-blocks — **`mate/`** (matmul: qkt/pv/pv_fp16), **`vecu/`** (online softmax),
**`precision_controller/`** (the ratio gate). Each has its own `AGENTS.md`, `DECISIONS.md`, `README.md`.

## Before you start — read these
- **`<sub>/research/`** and **`research/apa-precision-policy/`** — the "why" (RL policy evolution that
  discovered the ratio gate, benchmarks, dead ends). Check before running any experiment.
- **`<sub>/DECISIONS.md`** and **`acu/DECISIONS.md`** — settled calls; don't re-litigate.
- **`## Known gotchas`** in `acu/README.md` and each sub-block README.
- **`acu/docs/acu_overview.md`** — the full legacy technical overview (paths are pre-reorg).

## Runbook (exact commands)
```
# RTL sim (shared harness lives in mate/rtl)
make -C acu/mate/rtl sim            # + sim_realdata / testvectors
# per-tile Sky130 sign-off
cd acu/mate/pdk/sky130/openlane/mate_qkt && librelane --dockerized config.json
# per-tile GF180 (tape-out PDK)
librelane acu/mate/pdk/gf180/librelane/mate_pv.yaml
# cross-block cosim (chip/) — vendored proxy RTL now points at acu/*/rtl
make -C chip/verif cosim
```

## Lab-notebook standard — MANDATORY (same commit/PR as your work)
1. **Docs travel with code** — touch `<sub>/rtl/` → update that sub-block's `README`/`docs/`.
2. **Log the decision** — one line in the sub-block `DECISIONS.md`: *what · why · date*.
3. **Log the gotcha** — add to `## Known gotchas`.
4. **Record the experiment** — *result · n · artifact · script* in `research/`.
5. **Report honestly** — a documented near-miss beats a faked pass (state waived corners with numbers).

## Commit conventions
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c` (not your default config, or
commits won't link). Block RTL commits go on the sub-block; integration on `chip/`.
