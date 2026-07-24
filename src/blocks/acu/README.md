# ACU — Attention Compute Unit

The **ACU** is the decode-attention datapath of the LonghornSilicon (Lambda) LLM inference
accelerator: `Q·Kᵀ → softmax → P·V`, with a per-tile INT8/FP16 precision gate. It lives at
`src/blocks/acu/` (reorg 2026-07; see `docs/REORG_NOTES.md`) and is a **multi-block unit**, so it
splits into three self-contained sub-blocks — each carrying the canonical block layout
(`sw/ rtl/ pdk/ docs/ research/`, per SOP §5.1) and mirroring on its own:

| Sub-block | What it is | Mirror |
|---|---|---|
| [`mate/`](mate/) | **MatE** — the matmul engine: `mate_qkt` (Q·Kᵀ scoring), `mate_pv` (INT8 P·V), `mate_pv_fp16` (FP16 P·V) + the 8×8 MAC-array reference model | `lambda-mate` |
| [`vecu/`](vecu/) | **VecU** — the decode online-softmax slice (`vecu_softmax`, exp-LUT + fp32 accum) plus RoPE / RMSNorm tiles | `lambda-vecu` |
| [`precision_controller/`](precision_controller/) | Per-tile INT8-vs-FP16 gate: the pre-softmax ratio test `max(|S|)·N > 10·Σ|S|` (30 FFs, 1-cycle) | `lambda-precision-controller` |

The assembled unit mirrors to **`lambda-acu`** (umbrella); each sub-block also mirrors on its own.

## Where things are
- **RTL + tb** → `src/blocks/acu/<sub>/rtl/` (SystemVerilog; the shared multi-tile synth/EDA harness
  lives in `src/blocks/acu/mate/rtl/`).
- **Reference models** → `src/blocks/acu/<sub>/sw/reference_model/`.
- **Hardening** → `src/blocks/acu/<sub>/pdk/<target>/`: `sky130/openlane/` (flagship dev/proof),
  `asap7/orfs/` (predictive 7nm bracket), `gf180/librelane/` (SSCS Chipathon 2026 shuttle configs).
- **Design notes / ISA** → `src/blocks/acu/<sub>/docs/`; ACU-wide docs → `src/blocks/acu/docs/`
  (incl. the legacy full overview `docs/acu_overview.md` — its paths are pre-reorg).
- **The RL research that birthed the ACU** (Adaptive Precision Attention policy evolution) →
  `research/apa-precision-policy/` at the repo root (chip-wide research).

## Status
Honest per-macro, per-PDK sign-off from `docs/PROGRESS.md` (generated from committed
`*metrics.json`; sign-off definitions in `docs/REVISION_SYNC_SOP.md` §5.2). Sign-off is **per flow**,
not per block.

- **sky130 (OpenLane) — signed-off:** `mate_pv` (75,660 µm² / 71 MHz), `mate_pv_fp16` (221,987 µm² /
  12 MHz), `mate_qkt` (198,901 µm² / 12 MHz), `vecu` `rope` (277,632 µm² / 10 MHz) and `vecu_softmax`
  (726,020 µm² / 10 MHz), `precision_controller` (8,642 µm² / 80 MHz). `rmsnorm` on sky130 is
  config-only.
- **gf180 (LibreLane) — signed-off:** `vecu` `rope` (432,140 µm² / 4 MHz) and `rmsnorm` (1,095,180 µm²
  / 4 MHz). Everything else on gf180 (`mate_*`, `vecu_softmax`, `precision_controller`) is
  **config-only** (declared, not run) — GF180 is the tape-out PDK, assembled via the chip-level
  padring in `chip/pdk/gf180/`.
- **asap7 (ORFS) — route-clean (NOT full sign-off):** `mate_pv` (2.0 GHz), `mate_pv_fp16` (286 MHz),
  `precision_controller` (1.18 GHz). ASAP7 runs no Magic-DRC and no LVS — a predictive 7nm bracket only.

## Known gotchas
- **"Signed off" is per flow.** Do not read "signed off on sky130" as "signed off everywhere" — most
  gf180 macros are config-only and every asap7 run is route-clean, not signed off.
- **ASAP7 ORFS is 4×-drawn** — areas read 16× too large unless de-scaled (confirm SITE `0.054×0.270`).
- **FP16 can't be bit-exact to numpy `@`** (BLAS pairwise sum ≠ sequential MAC). Verify FP16 RTL vs a
  sequential-fp32 golden; tolerance vs numpy `rel_err < 5e-3`.
- **The `vecu_softmax` exp-LUT carries ~2% error** vs exact softmax (64-entry linear interp over [-16,0]);
  cosim tolerances are set FROM it, not tighter.
- **Long combinational fp paths won't close at the slow corner** — pipeline them (decode is latency-tolerant).
- **`precision_controller.sv` is byte-identical to the out-of-band `attention-compute-unit` repo** —
  a one-time manual import with no auto-sync, so either side can drift silently. Future changes
  originate here (SOP §7).
- The 16nm area/speed projections **do not reconcile** between Sky130-scaled and ASAP7-derived (~5×
  apart); treat 16nm as a range pending a real Cadence 16FFC run. See `docs/acu_overview.md`.

## Branch model
`main` is a clean scaffold — structure, docs, `pdk/` configs, Python golden models, `results/` — but
**no `.sv`/`.v` RTL**. The RTL lives on the `rev0` revision branch: contributors PR into `rev0`, a
lead blesses (RTL reviewed, sign-off reproduced) and merges upstream to `main`. To view/work on RTL:
`git checkout rev0`. Full model: `docs/REVISION_SYNC_SOP.md` §6a.

## Before you touch this block
Read [`AGENTS.md`](AGENTS.md) (front door), the sub-block `DECISIONS.md`, and the sub-block README's
`## Known gotchas`. Follow the lab-notebook rule: docs travel with code in the **same** commit.
