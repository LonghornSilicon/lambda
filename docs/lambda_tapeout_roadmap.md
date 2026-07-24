# Lambda — Roadmap to Tape-out (SUPERSEDED — see docs/ROADMAP.md)

> **⚠️ SUPERSEDED by [`docs/ROADMAP.md`](ROADMAP.md)** (the canonical timeline of record). This
> doc's "December 2026 full-chip" framing predates the two-track decision: the GF180/Chipathon proxy
> track now runs **in parallel** with the hallmark N16 product track (**N16 tapeout ~Fall 2027**,
> +1 semester buffer), not as a single sequential path. Retained for its per-block build detail;
> for dates and track structure, read `ROADMAP.md`.

**(Historical framing below.)** The near-term GF180 work is a milestone toward the full chip. This
doc captures where every block
stands, the two tracks (chipathon submission + full chip), and the path to December.

## Where every block stands (2026-07-23)

| Block | RTL | Sky130 | GF180 | ASAP7 | In Dec chip? |
|---|---|---|---|---|---|
| **KVE** (ChannelQuant KV codec) | ✅ | ✅ signed | ✅ real-SRAM macro, DRC/LVS 0 | — | ✅ |
| **TIU** (H2O importance) | ✅ | ✅ | ✅ | — | ✅ |
| **precision_controller** (ACU gate) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MatE — mate_pv / _fp16 / _qkt** | ✅ | ✅ | ✅ (macro) | ✅ (pv/fp16) | ✅ |
| **VecU — vecu_softmax** | ✅ | ✅ | ✅ (macro) | — | ✅ |
| **RoPE** | ❌ no RTL | — | — | — | ✅ needed |
| **RMSNorm** | ❌ no RTL | — | — | — | ✅ needed |
| **MatE — general GEMM/FFN 8×8 systolic** | ❌ no RTL | — | — | — | ✅ needed (weights/FFN) |
| **VecU — full SIMD (SiLU, residual, sampling)** | ❌ partial (softmax only) | — | — | — | ✅ needed |
| **MSC / MHC** (LPDDR5X + PagedAttention table) | ❌ no RTL | — | — | — | ✅ needed |
| **LSU** (32-inst schedule RISC) | ❌ no RTL (asm only) | — | — | — | ✅ needed |
| **HIF** (PCIe Gen3 x1) | ❌ vendor IP | — | — | — | integrate IP |

**So: the whole decode-attention datapath is RTL-real + per-block GDS-clean on GF180 and Sky130.**
The gap to a full chip is the *rest of the SoC* — RoPE/RMSNorm, the general GEMM engine, full VecU,
the memory controller, the sequencer, and the host IP.

---

## Track 1 — Chipathon 2026 submission (near-term): KV-compression coprocessor

**Scope:** KVE + TIU + precision_controller only (~**0.7 mm²**, fits the fixed 4.21 mm² workshop core
*comfortably* — proven: the full fp16 attention datapath does **not** fit, congestion >1.0 even at the
minimal tile). This is Lambda's **headline research contribution** — ChannelQuant KV compression +
H2O importance — as a real, taped-out coprocessor; the host feeds attention scores.

**To do:**
1. A coprocessor `chip_core` variant instantiating **only** KVE + TIU + precision_controller (drop
   MatE/VecU), wired to the 20 bidir pads via the existing `spi_loader`.
2. `SLOT=workshop make librelane` → full-chip GDS (this one **closes** — 0.7 mm² in a 4.21 mm² core).
3. Team issue + proposal + the fork (registration mechanics), slides, demo video.

**Status:** all three blocks done; the flow (`submit.sh`, `config_fullchip.yaml`, the 3 PnR fixes) is
proven to CTS. Remaining = the coprocessor top + the closing PnR (fits, so it should close).

---

## Track 2 — Full Lambda chip (December tape-out)

The December chip is the whole SoC on a **die sized for it** (NOT the 4.21 mm² workshop slot — the
attention datapath alone needs more routing area; December uses an appropriate die). Two work-streams:

### (A) Finish the missing RTL blocks
Ordered by size/tractability (build golden-first, bit-exact TB, then GF180+Sky130 harden — the
established block flow):
1. **RoPE** — pair-rotation + sin/cos LUT. Small, well-defined. *(days)*
2. **RMSNorm** — sum-of-squares + rsqrt LUT + scale. Small. *(days)*
3. **LSU** — 32-inst in-order sequencer + 4 KB microcode RAM. Assembler exists. *(1–2 wks)*
4. **Full VecU** — fold RoPE/RMSNorm + SiLU/GELU + residual + sampling into the softmax datapath
   as a programmable SIMD lane. *(2–3 wks)*
5. **MatE general 8×8 GEMM/FFN systolic** — the weight-stationary array for QKV/FFN/logits (Q·Kᵀ +
   P·V tiles already exist). Canonical design, golden = `mac_array_ref`. *(2–4 wks)*
6. **MSC / MHC** — PagedAttention block table + SRAM crossbar (LPDDR5X PHY = vendor IP). *(3–4 wks)*
7. **HIF** — integrate a PCIe hard-IP block. *(integration)*

### (B) 16nm numbers (port from what we have)
We already have GF180 (180 nm), Sky130 (130 nm) sign-offs and an ASAP7 (7 nm FinFET) bracket per
block. **Port to TSMC N16FFC by ratio** (ASAP7 is the best device-family proxy — FinFET), giving
16nm area/power/fmax *estimates* with disclaimers, per the existing `pdk_holes_audit.md` methodology.
This is what the December design review needs — and it's mostly a compute pass, not new RTL.

### (C) Full-chip integration + a real die
`lambda_acu` (the 13-state decode FSM) already assembles the datapath and is cocotb-verified over
SPI. For December: add the missing blocks to the top, size the die appropriately, close full-chip
PnR (hierarchical macro assembly — each block a hardened macro, top-level places them).

---

## Immediate execution queue (gated by the session-limit reset, 8:30am UTC)
1. **Coprocessor `chip_core` + closing GDS** (Track 1) — the achievable submission win.
2. **RoPE + RMSNorm RTL** (Track 2A, smallest first) — new blocks, golden-first, GF180+Sky130 harden.
3. **16nm porting pass** (Track 2B) — ratio-port every block's numbers; light compute.
4. Then LSU → full VecU → general MatE → MSC, in that order.

*The chipathon proves the KV coprocessor in silicon this cycle; December proves the whole chip.
Every block we harden now is a December block — nothing here is throwaway.*
