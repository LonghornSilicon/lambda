# AGENTS.md — MatE (acu/mate)

> Front door for the ACU matmul engine. Read before touching `mate/`. Also read `acu/AGENTS.md`.

## What this is
The matmul tiles: `mate_qkt` (Q·Kᵀ), `mate_pv` (INT8 P·V), `mate_pv_fp16` (FP16 P·V), plus the 8×8
MAC-array reference model. The shared multi-tile synth/EDA harness lives in `rtl/`.

## Before you start
- `research/` + `research/apa-precision-policy/analysis/` — sweeps (area-vs-bitwidth/tile), fixed-point sims.
- `DECISIONS.md` — INT32 accumulator, 8×8 grid, FP16 escape path.
- `## Known gotchas` in `README.md`.
- `docs/mac_array_design.tex`, `docs/mate_*_rtl.md`.

## Runbook
```
make -C acu/mate/rtl testvectors   # golden vectors from gen_*_vectors.py
make -C acu/mate/rtl sim           # directed cases
make -C acu/mate/rtl sim_realdata  # replay tiles
cd acu/mate/pdk/sky130/openlane/mate_pv && librelane --dockerized config.json
librelane acu/mate/pdk/gf180/librelane/mate_qkt.yaml
```

## Lab-notebook standard — MANDATORY (same commit)
Docs travel with code · log the decision · log the gotcha · record the experiment (result·n·artifact·script)
· report honestly. Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c`.
