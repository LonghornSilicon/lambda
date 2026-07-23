# AGENTS.md — VecU (acu/vecu)

> Front door for the ACU vector unit. Read before touching `vecu/`. Also read `acu/AGENTS.md`.

## What this is
Three synthesizable VecU slices: `vecu_softmax` (decode online softmax, exp-LUT + fp32 accumulator),
`rope` (rotary position embedding, sin/cos LUT + fp16 rotation), `rmsnorm` (RMSNorm, fp32
sum-of-squares + rsqrt LUT + per-element gain). All fp32-internal with one fp16 rounding on emit, all
micro-sequenced (one fp32-op/cycle). The full programmable VecU is future work.

## Before you start
- `research/` — softmax-slice exploration notes.
- `DECISIONS.md` — decode-only scope, the ~2% exp-LUT error, the pipelining call, the RoPE/RMSNorm build.
- `## Known gotchas` in `README.md`; `docs/{vecu_softmax,rope,rmsnorm}_rtl.md`.

## Runbook
```
make -C acu/mate/rtl sim_vecu_softmax     # softmax shares the MatE harness
make -C acu/vecu/rtl sim                   # rope + rmsnorm bit-exact TBs
make -C acu/vecu/rtl synth                 # Yosys FF/area + t:$dlatch = 0 assertion
python acu/vecu/sw/reference_model/rope_ref.py      # LUT-vs-exact error (bar b)
python acu/vecu/sw/reference_model/rmsnorm_ref.py   # LUT-vs-exact error (bar b)
cd acu/vecu/pdk/sky130/openlane/{vecu_softmax,rope,rmsnorm} && librelane --dockerized config.json
librelane acu/vecu/pdk/gf180/librelane/{vecu_softmax,rope,rmsnorm}.yaml
```

## Lab-notebook standard — MANDATORY (same commit)
Docs travel with code · log the decision · log the gotcha · record the experiment · report honestly.
Author as `Chaithu Talasila <themoddedcube@gmail.com>` via `git -c`.
