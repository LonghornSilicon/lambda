# Chipathon RTL-Closure Plan — Decode Attention Datapath (Sky130)

**Decided 2026-07-21.** Target: the **SSCS Chipathon 2026** shuttle on **GF180MCU**
(GlobalFoundries 180nm, LibreLane multi-macro; template `sscs-chipathon-2026`
`examples/librelane_rtl2gds_gf180/04_counter_alu_multimacro`), ~2–3 months runway.
**PDK locked = GF180MCU.** The
Sky130 sign-offs we already have become **dev-vehicle proofs** (the RTL is physical); the shuttle
re-hardens on GF180. Boundary: the **decode attention datapath** — the coherent completion of the
RTL we have actually built. Projections (QKV/output) and the FFN GEMMs run **off-chip** (host-fed)
for this shuttle; the general 8×8 systolic GEMM/FFN engine is a larger, separate program that can
be added later.

## Repo-of-record split (standing convention, 2026-07-21)

- **Block RTL + block-level verification** live in each block's folder in this monorepo
  (`kve/`, `tiu/`, `acu/`), with the cross-block cosim in `chip/verif/`. This is where RTL is
  authored, unit-tested, and committed.
- **PDK work — GF180 LibreLane hardening, multi-macro integration, padring, GDSII, the tapeout
  package** — lives in **`chip/pdk/gf180/`**. It pulls each block's RTL in as a
  hardened macro; it does not author block RTL.

## Why this boundary

"Everything we've built" **is** the attention + KV datapath (KVE, TIU, precision-controller,
P·V). The projection/FFN GEMMs were *never* RTL — they are part of the un-built 8×8 systolic
array, spec-only from day one. So closing "our" gap = finishing the attention pass on-die.
The ACU (Attention Compute Unit) is only meaningfully "attention" once MatE does Q·Kᵀ scoring
and VecU does softmax; those are the two blocks we build here.

## The key feasibility lever: decode ⇒ matrix-**vector**

The validated workload is Qwen2 **decode** (one query token per step). That collapses the
hard 2D structures into 1D:
- **Q·Kᵀ** = one Q vector (D) · L cached K vectors → L scores. A **reduction engine**, not a
  full weight-stationary 2D systolic array. K arrives per-channel-dequantized FP16 from the
  KVE, so it reuses the existing FP16 datapath (INT8 Q → exact fp16, ×fp16 K, fp32 accumulate).
- **softmax** = one row → the running-max/running-sum recurrence, single-lane. Not the full
  8-lane programmable microcode VecU — just the decode softmax slice + the 64-entry exp LUT.
- **P·V** = already `mate_pv` (INT8) + `mate_pv_fp16` (FP16 escape).

## Closure harness: the cosim is the scoreboard

`architecture/rtl/tb/tb_chip_cosim.sv` runs green today with **reference stand-ins** for the
un-built stages (scores supplied as data; probabilities precomputed). We replace each stand-in
with **verified RTL one at a time, keeping the cosim green throughout** (bit-exact / toleranced
to the same golden). Closure metric = **# reference stand-ins → 0**. When it hits zero, the same
green cosim *is* a full-attention-datapath RTL sim on real Qwen tensors — that is the honest
"reliably tested end-to-end on 130nm."

Current cosim (post FP16 wiring, commit `2aaa471`):

| Cosim stage | Status |
|---|---|
| BLOCK 2 (KVE) reconstruct V̂ | **real RTL** |
| BLOCK 2b/2c (MatE P·V, INT8 + FP16 escape) | **real RTL** |
| BLOCK 3 (TIU) keep-tier + evict | **real RTL** |
| BLOCK 1 (ACU) precision gate | **real RTL** |
| **Q·Kᵀ score row** | **real RTL** (`mate_qkt`) — Phase 1 done 2026-07-21 |
| **softmax** | **real RTL** (`vecu_softmax`) — Phase 2 done 2026-07-21; **softmax-path stand-ins → 0** |
| **RoPE / RMSNorm** | reference stand-in (chip-top raw-Q/K path only; loaded Qwen tiles are pre-RoPE'd) ← integration |

## Phases

| Phase | Build | Cosim effect | Gate |
|---|---|---|---|
| **0** (now) | Plan doc; RTL-maturity honesty in STATUS; finish FP16 wiring (done) | stand-ins labeled | — |
| **1 — MatE Q·Kᵀ** ✅ done | Decode Q·Kᵀ reduction engine (INT8 Q × per-channel FP16 K → L scores); golden from `mac_array_ref`; bit-exact/toleranced TB; swap into cosim BLOCK 1 | scores → **real RTL** ✅ | full cosim `ALL BLOCKS PASS` ✅ (scores rel-err 4e-6) |
| **2 — VecU softmax slice** ✅ done | Wrote the golden first (`sw/reference_model/vecu_softmax_ref.py` in `attention-compute-unit`); single-row online-softmax + 64-entry exp LUT + linear interp + `exp(m_old-m_new)` rescale, fp32 accumulator; bit-exact TB + LUT-vs-exact bar (≈2%); swapped into cosim BLOCK 2d | probabilities → **real RTL** ✅ (softmax-path stand-ins → 0) | full cosim `ALL BLOCKS PASS` ✅ (softmax weights err 2e-4, attn-out rel-err 7e-4) |
| **3 — Integrate** | ACU top wrapper + mini decode-step control FSM; full-datapath cosim on real Qwen tiles | **stand-ins = 0** | end-to-end green |
| **4 — GF180 hardening** *(in `chipathon-lambda-acu`)* | Harden each block as a GF180 LibreLane macro (start with the already-signed logic blocks to de-risk the port early: precision-controller, mate_pv); then the integrated ACU (KVE SRAM macros, floorplan, hierarchy); 6 sign-off checks | — | clean GF180 sign-off per macro |
| **5 — Padring + submit** *(in `chipathon-lambda-acu`)* | `chip_core.sv` workshop-slot override + serial/SPI loader (≈20 pads ≪ D=128), stitch macros into the chipathon-2026 padring fork, cocotb GL sim, final GDS, MPW submit | — | shuttle-ready package |

## GF180 verification status

**Stage 1 done (2026-07-21, `chipathon-lambda-acu` `cde0bf2`, `docs/gf180_gls_report.md`).**
All five already-RTL blocks hardened on GF180MCU (LibreLane 3.0.5 via docker, gf180mcuD PDK):
setup/hold/DRC/LVS/antenna **clean** (residual max-transition only at the extreme `ss_125C_4v50`
corner; tt/ff clean; precision-controller clean at all corners). **Gate-level end-to-end GLS**
(`tb/tb_gls_e2e.sv`) reproduces the cosim on a real Qwen tile against the hardened netlists +
gf180 cells: INT8 P·V **int32 bit-exact**, FP16 P·V **1.74e-4**, ACU gate matches & discriminates,
TIU match. **Honest boundary:** GL-in-the-loop = `mate_pv`/`mate_pv_fp16`/`precision_controller`/
`token_importance_unit`; KVE reconstruct is the small combinational RTL path feeding the GL P·V;
**KVE KV storage synthesizes to FF register arrays (no `gf180mcu_fd_ip_sram` macro yet) at a
depth-2 proxy — NOT real KV capacity** (real SRAM macro = TODO).

**Stage 2 done (2026-07-21, `chipathon-lambda-acu` `8679212`).** `mate_qkt` + `vecu_softmax`
hardened on GF180 and added to the GL e2e: the full **Q·Kᵀ→softmax→P·V compute datapath** is now
gate-level verified on gf180 cells (mate_qkt scores rel-err 0.0, softmax weights 2.13e-4,
closed-loop attention out 5.42e-4 vs reference; Stage-1 no regression). `mate_qkt` closes 6/6.
**Timing closed (2026-07-21, `chipathon-lambda-acu` `c63e705`):** `vecu_softmax` was pipelined
(3-stage exp/rescale/accumulate split, +358 FFs, bit-exact, cosim re-confirmed `ALL BLOCKS PASS`)
and re-hardened — ss-corner setup **−26.5 ns → +19.2 ns** at a tightened 300 ns clock. **All seven
compute macros now meet setup AND hold at every corner**, setup TNS = 0. GL e2e still ALL PASS.

**Remaining honest items (none block correctness; setup/hold/DRC/LVS closed):**
1. **KVE `gf180mcu_fd_ip_sram` macro** — the one gate-level hole; KV storage hardens as FF
   register arrays at the depth-2 proxy today (not real KV capacity).
2. **`vecu_softmax` area reclaim** — the aggressive resize to hit ss ~2×'d its cells (55k→101k,
   1.49 mm²); a more even pipeline split / higher utilisation reclaims it.
3. **ss-corner max-transition (slew)** on the large fp16 / register-array blocks — physical-opt
   follow-up (does not affect setup/hold/DRC/LVS).

## Risk register (honest)

- ~~**VecU is the long pole** — no golden model exists yet (write `vecu.py` before RTL); the
  transcendental LUTs + rescale are fiddly.~~ **Resolved (Phase 2, 2026-07-21):** the decode
  online-softmax golden + `vecu_softmax` RTL are written and bit-exact; the 64-entry exp LUT
  carries ≈2% error vs exact softmax (the P·V-vs-reference cosim tolerance is set from it).
  RoPE / RMSNorm remain for integration but are not on the decode-softmax critical path.
- **Top-level Sky130 close** — the integrated datapath is far larger than the individual tiles;
  the KVE SRAM macros make floorplanning real.
- **2–3 months is tight but plausible** *with the decode simplification + parallel agents*. It is
  **not** plausible for the full systolic GEMM + FFN engine — which is exactly why those are
  off-chip for this shuttle.

## Verification discipline

Every new block: golden-first, bit-exact (INT) or toleranced (`rel_err < 5e-3`, FP) TB, **then**
swap into the cosim and re-run the *full* regression — no phase advances on a red cosim.
