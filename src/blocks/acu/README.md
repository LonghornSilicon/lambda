# ACU — Attention Compute Unit

The **ACU** is the decode-attention datapath of the LonghornSilicon (Lambda) LLM inference
accelerator: `Q·Kᵀ → softmax → P·V`, with a per-tile INT8/FP16 precision gate. It is a
**multi-block unit**, so — per the block-major layout (`docs/repo_reorg_plan.md`) — it splits
into self-contained sub-blocks, each carrying its own `sw/ rtl/ pdk/ docs/ research/`:

| Sub-block | What it is | Mirror |
|---|---|---|
| [`mate/`](mate/) | **MatE** — the matmul engine: `mate_qkt` (Q·Kᵀ scoring), `mate_pv` (INT8 P·V), `mate_pv_fp16` (FP16 P·V) + the 8×8 MAC-array reference model | `lambda-mate` |
| [`vecu/`](vecu/) | **VecU** — the decode online-softmax slice (`vecu_softmax`, exp-LUT + fp32 accum) | `lambda-vecu` |
| [`precision_controller/`](precision_controller/) | Per-tile INT8-vs-FP16 gate: the pre-softmax ratio test `max(|S|)·N > 10·Σ|S|` (~30 FFs, 1-cycle) | `lambda-precision-controller` |

The assembled unit mirrors to **`lambda-acu`** (umbrella); each sub-block also mirrors on its own.

## Where things are
- **RTL + tb** → `<sub>/rtl/` (SystemVerilog; the shared multi-tile synth/EDA harness lives in `mate/rtl/`).
- **Reference models** → `<sub>/sw/reference_model/`.
- **Hardening** → `<sub>/pdk/<target>/`: `sky130/openlane/` (flagship dev/proof, signed off),
  `asap7/orfs/` (predictive 7nm bracket), `gf180/librelane/` (SSCS Chipathon 2026 shuttle configs).
- **Design notes / ISA** → `<sub>/docs/`; ACU-wide docs → `acu/docs/` (incl. the legacy full
  overview `acu/docs/acu_overview.md` and the KVE↔ACU integration findings).
- **The RL research that birthed the ACU** (Adaptive Precision Attention policy evolution) →
  `research/apa-precision-policy/` at the repo root (chip-wide research).

## Signed-off status (Sky130A, real)
All five logic tiles sign off clean on Sky130A (DRC/LVS/antenna/IR 0/0/0/0): `mate_pv`, `mate_pv_fp16`,
`mate_qkt`, `vecu_softmax`, `precision_controller`. ASAP7 gives a predictive 7nm bracket for
`mate_pv`/`mate_pv_fp16`/`precision_controller`. GF180 is the tape-out PDK (LibreLane, via the
chip-level padring in `chip/pdk/gf180/`).

## Known gotchas
- **ASAP7 ORFS is 4×-drawn** — areas read 16× too large unless de-scaled (confirm SITE `0.054×0.270`).
- **FP16 can't be bit-exact to numpy `@`** (BLAS pairwise sum ≠ sequential MAC). Verify FP16 RTL vs a
  sequential-fp32 golden; tolerance vs numpy `rel_err < 5e-3`.
- **The `vecu_softmax` exp-LUT carries ~2% error** vs exact softmax (64-entry linear interp over [-16,0]);
  cosim tolerances are set FROM it, not tighter.
- **Long combinational fp paths won't close at the slow corner** — pipeline them (decode is latency-tolerant).
- The 16nm area/speed projections **do not reconcile** between Sky130-scaled and ASAP7-derived (~5× apart);
  treat 16nm as a range pending a real Cadence 16FFC run. See `acu/docs/acu_overview.md`.

## Before you touch this block
Read [`AGENTS.md`](AGENTS.md) (front door), the sub-block `DECISIONS.md`, and the sub-block README's
`## Known gotchas`. Follow the lab-notebook rule: docs travel with code in the **same** commit.
