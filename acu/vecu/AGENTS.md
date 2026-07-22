# AGENTS.md — VecU (acu/vecu)

> Front door for the ACU vector unit. Read before touching `vecu/`. Also read `acu/AGENTS.md`.

## What this is
The decode online-softmax slice `vecu_softmax` (exp-LUT + fp32 accumulator). Single-row; the full
programmable VecU (+ RoPE/RMSNorm) is future work.

## Before you start
- `research/` — softmax-slice exploration notes.
- `DECISIONS.md` — decode-only scope, the ~2% exp-LUT error, the pipelining call.
- `## Known gotchas` in `README.md`; `docs/vecu_softmax_rtl.md`.

## Runbook
```
make -C acu/mate/rtl sim_vecu_softmax   # shared harness (see mate/rtl/Makefile targets)
cd acu/vecu/pdk/sky130/openlane/vecu_softmax && librelane --dockerized config.json
librelane acu/vecu/pdk/gf180/librelane/vecu_softmax.yaml
```

## Lab-notebook standard — MANDATORY (same commit)
Docs travel with code · log the decision · log the gotcha · record the experiment · report honestly.
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c`.
