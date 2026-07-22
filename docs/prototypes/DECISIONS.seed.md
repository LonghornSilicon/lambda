<!-- PROTOTYPE SEED. At migration this splits into per-block DECISIONS.md + each block README's
     "Known gotchas" section. Captured from the 2026-07 work so it isn't re-litigated / re-hit.
     Format: what · why · date. Append-only; never delete, mark superseded. -->

# DECISIONS (seed) — settled calls, do not re-litigate unless the premise changed

## Chip-wide
- **Tapeout = SSCS Chipathon 2026 on GF180MCU** (LibreLane multi-macro padring), NOT Sky130 · the
  2026 shuttle PDK is GF180 · 2026-07-21. Sky130 stays as the flagship/dev-vehicle proof.
- **Tapeout boundary = the decode attention datapath** (Q·Kᵀ→softmax→P·V + KVE + TIU + ACU gate) ·
  it's the coherent completion of the RTL we actually built; projections/FFN GEMMs were never RTL
  and run off-chip for the shuttle · 2026-07-21.
- **Repo structure = monorepo `lambda`, block-major, auto-mirror per block** · one source of truth,
  self-contained blocks, standalone mirror repos with no drift · 2026-07-22. See `repo_reorg_plan.md`.
- **Online-softmax citation = Milakov & Gimelshein 2018**, not FlashAttention-3 · the recurrence
  predates FA; FA added the tiling · 2026-07-21.
- **Commit identity = `themoddedcube@gmail.com`** via `git -c` · other emails don't link to the
  GitHub account · 2026-07-20.

## MatE (acu/mate)
- **P·V accumulator = INT32, not INT24** · P·V reduces over the TOKEN axis, so width scales with
  context; INT24 overflows past ~520 flat tokens, INT32 covers ~133k · 2026-07-20.
- **8×8 grid = 64 PEs = 128 GOPS** · the reference-model default was a stale 16×16/256; corrected to
  match arch.yml/STATUS · 2026-07-21.
- **FP16 P·V escape path exists** · attention P·V routes per-tile INT8/FP16 via the precision
  controller (`max·N > 10·Σ`); weight/FFN GEMMs stay INT8×INT4 · 2026-07-18.

## VecU (acu/vecu)
- **Decode softmax slice only** (single-row online softmax + 64-entry exp LUT); full programmable
  VecU + RoPE/RMSNorm are later · the decode datapath doesn't need them · 2026-07-21.
- **exp LUT carries ~2% error vs exact softmax** · inherent to a 64-entry linear-interp LUT over
  [-16,0]; cosim tolerances are set FROM it, not tighter · 2026-07-21.

## KVE (kve)
- **WHT value rotation is RECONFIGURABLE** · the datapath carries a per-channel sign vector
  (`sign_flips_`, applied before the WHT on the value write path), which makes the rotation
  programmable: **fixed** (all-ones signs) is the accuracy-recommended default, **randomized**
  (loaded signs) is selectable — hardware supports both, no design-time pick · 2026-07.
  *(Supersedes the 2026-07-20 "FIXED locked" call, which was accuracy-only; since the sign vector
  already exists, keeping it reconfigurable costs ~nothing and preserves the option.)*
- **CQ-4 is the default at every head dim** (the "+" outlier lane is optional) · n=1000 reversed the
  n=250 screening: the lane only marginally helps at D=128, slightly hurts at D=64 · 2026-07-21.
- **KV storage behind a swappable `kv_sram` interface** (behavioral default; real gf180 SRAM macro
  in the pdk layer) · keeps block RTL PDK-agnostic · 2026-07-22.

# KNOWN GOTCHAS (seed) — pitfalls that cost time; check before debugging

## Environment / flow
- **LHS box venv is read-only, no numpy/pip.** Use `/home/shadeform/cuda_advisor/.venv/bin/python`
  for numpy; reinstall `iverilog`/`yosys` each session. Prefer pure-Python golden generators.
- **ORFS ASAP7 is 4×-drawn.** Areas read 16× too large unless de-scaled — confirm the SITE size
  (`0.054×0.270`) before quoting µm².
- **KVE synth: the behavioral `real`/`$fscanf` views abort yosys.** Use the `*_syn.sv` set
  (`cq_units_syn`, `wht_unit_syn`, `fp16_addsub_syn`, …).

## GF180 hardening
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` DISABLES slew repair** (it passes `-slew_margin 0`) — an inverted
  setting. Restore ~20% or you get thousands of false max-slew/cap violations.
- **gf180 SRAM macro power connects on Metal3.** Route the macro power to the M4 straps with a legal
  **Via3** — the Metal1/Metal2 route forces illegal Via1/Via2 stacks (7000+ DRC).
- **The gf180 SRAM abstract has ONE sub-min-width pin** (0.11 µm vs 0.28 µm) — an abstract artifact;
  the vendor GDS is clean. Use a DRC-view-only maglef that widens just that pin; run LVS on the real
  device view. (Re-DRC'ing the real GDS throws ~38k false bitcell errors.)
- **LibreLane escaped-identifier instance naming** takes 3 different forms across
  `PDN_MACRO_CONNECTIONS` regex / `instances` placement / YAML quoting — match each exactly.

## Numerics
- **FP16 can't be bit-exact to numpy `@`** (BLAS pairwise sum ≠ sequential MAC order). Verify FP16 RTL
  against a **sequential-fp32 golden**, and tolerance vs numpy (`rel_err < 5e-3`).
- **Long combinational fp path won't close at the slow corner** — e.g. two serial fp32 mults. Pipeline
  it (register the intermediate); decode is latency-tolerant so extra cycles are free.
