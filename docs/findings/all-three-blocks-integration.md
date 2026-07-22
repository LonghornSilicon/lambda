# All-Three-Blocks Integration Test — TIU + KVCE + APA on Qwen2

**Status:** Complete — the three live blocks compose and hold accuracy.
**Date:** 2026-07-17.
**One-line:** Stacking Token Importance Unit eviction (25% KV budget) + ChannelQuant
4-bit KV + APA ~all-INT8 attention on Qwen2 costs only ~3% HellaSwag acc_norm vs FP16;
graded value demotion recovers ~1pt on the 0.5B model — at a memory cost (~5.9 vs 4.0
b/val), not free; sub-4-bit demotion to offset it craters (see `graded-value-2bit.md`).

---

## Setup

`analysis/full_stack_integration.py` composes all three blocks in one custom
Qwen2 attention, in chip order: KVCE decompresses K/V → attention scores → TIU rules
keep/evict (and optional per-token value tier) → APA routes the S·V MAC INT8/FP16.
HellaSwag acc_norm, n=1000, gold TIU config (25% KV budget, recent-window ratio 0.5).

## Result (Δ vs FP16 full-cache baseline)

| config | Qwen2-0.5B (D=64) | Qwen2-1.5B (D=128) | Llama-3.2-1B (D=64) |
|---|---|---|---|
| fp16 | 0.489 | 0.590 | 0.564 |
| TIU evict only | −0.016 | −0.034 | −0.019 |
| KVCE cq4+ only | −0.015 | −0.003 | −0.005 |
| APA only | +0.001 | −0.002 | +0.001 |
| TIU + KVCE | −0.018 | −0.025 | −0.025 |
| **ALL 3 (evict + cq4+ + APA)** | **−0.033** | **−0.030** | **−0.017** |
| ALL 3 with **CQ-3-rot** values (evict + WHT-INT3 + APA) | **−0.036** | **−0.036** | — |
| ALL 3 + graded value demotion | −0.023 | −0.029 | −0.024 |

Each block alone is within (or near) the ±0.02 per-block gate. The full stack lands
at ~−0.02 to −0.03 — the expected cumulative cost of three aggressive, independent
optimizations (75% cache eviction × 4-bit KV × INT8 compute). APA remains free
(≈FP16, 99.999% INT8) on every model. To stay under ±0.02 combined, back the TIU budget
off to ~35% (where TIU alone is ≈−0.006); 25% is the aggressive operating point.

**Cross-family:** the stack was re-run on **Llama-3.2-1B** (different family) and lands
at **−0.017** — actually *better* than Qwen2, and every block behaves the same
qualitatively (APA free, KVCE ~−0.005, TIU the dominant cost). The three-block design is
not tuned to Qwen; it holds across families.

## Two integration findings

1. **Per-token graded demotion is incompatible with ChannelQuant's KEY path.**
   ChannelQuant compresses keys *per-channel over a token group*; assigning
   individual key tokens different bit-widths degenerates to the per-token-key codec
   that collapses GQA accuracy (−0.10, the failure ChannelQuant was built to avoid).
   Measured: graded-keys drove the stack to −0.17. **Keys must stay uniform
   per-channel; only VALUES (which are already per-token) can be graded.**

2. **Graded VALUE demotion helps — but it's a memory *tax*, not free.** Mapping token
   importance → value precision (top 10% FP16 / top 25% CQ-8 / next CQ-4 / rest CQ-4+)
   beats uniform CQ-4+ by ~+0.010 on Qwen2-0.5B (−0.023 vs −0.033), neutral on 1.5B.
   **Correction to an earlier "starve the rest" framing:** the ladder only *promotes*
   above 4-bit and never demotes below, so its average is **~5.9 vs 4.0 b/val** — the
   +1pt is bought with ~47% more value memory. Making it memory-neutral by 2-bitting
   low-importance tokens does NOT work with the current codec: uniform-INT2 craters
   (−0.09/−0.28) and a memory-neutral 2-bit-floor ladder is a wash on 0.5B and −0.065
   on 1.5B — the down-weighting doesn't rescue naive sub-4-bit
   (`graded-value-2bit.md`). A grouped-INT2 + residual-window codec (KIVI-style) might,
   but that needs a real 2-bit RTL tier (absent today).

## Implication for the TIU↔KVCE interface

The tier signal the TIU emits should drive the **value** path's precision per token,
and the **eviction** decision for both K and V — but NOT a per-token key bit-width.
Keys are demoted collectively (the whole cache uses one ChannelQuant key tier); the
TIU's per-token lever is evict-or-keep for keys, and evict/demote for values.

## Reproduce

```sh
python analysis/full_stack_integration.py --model Qwen/Qwen2-0.5B --n 1000 --frac 0.25 --recent_ratio 0.5
```
