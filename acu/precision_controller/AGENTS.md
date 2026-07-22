# AGENTS.md — Precision Controller (acu/precision_controller)

> Front door for the ACU precision gate. Read before touching `precision_controller/`. Also read `acu/AGENTS.md`.

## What this is
The per-tile INT8/FP16 gate: `max(|S|)·N > 10·Σ|S|` → FP16 else INT8. Streaming, 1-cycle, ~30 FFs,
signed off on Sky130A.

## Before you start
- `research/` + `research/apa-precision-policy/` — how the ratio gate was discovered (entropy →
  ratio), fixed-point sims, threshold sweeps.
- `DECISIONS.md` — the ratio test, threshold=10, FF closed form, `.v` vs `.sv`.
- `## Known gotchas` in `README.md`; `docs/isa/precision_controller_isa.md`.

## Runbook
```
make -C acu/mate/rtl sim_precision_controller    # shared harness; 253 cases
cd acu/precision_controller/pdk/sky130/openlane/precision_controller && librelane --dockerized config.json
librelane acu/precision_controller/pdk/gf180/librelane/precision_controller.yaml
```

## Lab-notebook standard — MANDATORY (same commit)
Docs travel with code · log the decision · log the gotcha · record the experiment · report honestly.
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c`.
