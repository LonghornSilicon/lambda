# Does a SMARTER 2-bit VALUE tier make graded demotion memory-neutral?

**Status:** Tested — **no.** A KIVI/KVQuant-style grouped-INT2 codec *massively*
improves low-bit value quant (it rescues the naive-2bit disaster, especially on
1.5B), but a **memory-neutral** graded ladder built on it (~4.1 b/val: top FP16 +
recent CQ4 + boring grouped-INT2) still **does not match uniform CQ-4** on either
model. The 4-bit floor stands even for a smart 2-bit codec at these scales.
**Date:** 2026-07-19.

## The open door (from `graded-value-2bit.md`)

Naive per-token symmetric INT2 craters (−0.089 / −0.281 on 0.5B / 1.5B), and a
memory-neutral naive-2bit ladder is a wash on 0.5B and −0.065 on 1.5B. The stated
open door: KIVI/KVQuant make 2-bit viable with **per-group** quant + a small
**recent-token residual window**. Does that smarter codec make memory-neutral
grading work — promote heavy hitters, starve boring tokens at grouped-2bit, break
even on bits — *especially on 1.5B, where naive-2bit failed hardest*?

## What changed (`analysis/graded_grouped2.py`)

- **Grouped INT2:** one INT2 scale per group of `G2` channels along head-dim D
  (per-group amax → scale → clamp(round, −2, 1)). Extra bits accounted: **2 + 16/G2**
  per value for the fp16 group scales (G2=32 → 2.5 b, G2=16 → 3.0 b).
- **Residual window:** most-recent R fraction of tokens (largest position index)
  held at CQ-4 regardless of importance.
- **Neutral ladder:** top `t` by accumulated-mass importance → FP16; recent R → CQ4;
  everything else → grouped-INT2. Tuned to ~4.0 b/val (measured reported, not assumed).
- Keys held uniform CQ-4+ per-channel throughout; no eviction, no APA. Same
  AttentionInterface plumbing/gotchas as `graded_2bit.py`.

## Results (HellaSwag acc_norm, n=1000, keys uniform CQ-4+)

### Qwen2-0.5B (FP16 = 0.489)

| value ladder | avg b/val | acc_norm | Δ vs FP16 |
|---|---|---|---|
| fp16 | 16.0 | 0.489 | — |
| **uniform4** (the 4-bit floor) | 4.0 | 0.474 | **−0.015** |
| grouped2_uniform G2=32 | 2.50 | 0.444 | −0.045 |
| grouped2_uniform G2=16 | 3.00 | 0.453 | −0.036 |
| grouped2_resid G2=32 R=0.1 | 2.65 | 0.427 | −0.062 |
| grouped2_resid G2=32 R=0.2 | 2.80 | 0.441 | −0.048 |
| grouped2_resid G2=16 R=0.2 | 3.20 | 0.444 | −0.045 |
| graded_neutral G2=32 t=.10 R=.10 | 4.12 | 0.444 | −0.045 |
| graded_neutral G2=32 t=.10 R=.15 | 4.19 | 0.452 | −0.037 |
| **graded_neutral G2=16 t=.07 R=.10** (best) | 4.14 | 0.463 | **−0.026** |

### Qwen2-1.5B (FP16 = 0.590) — the decisive test

| value ladder | avg b/val | acc_norm | Δ vs FP16 |
|---|---|---|---|
| fp16 | 16.0 | 0.590 | — |
| **uniform4** (the 4-bit floor) | 4.0 | 0.587 | **−0.003** |
| grouped2_uniform G2=32 | 2.50 | 0.496 | −0.094 |
| grouped2_uniform G2=16 | 3.00 | 0.548 | −0.042 |
| grouped2_resid G2=32 R=0.1 | 2.65 | 0.481 | −0.109 |
| grouped2_resid G2=32 R=0.2 | 2.80 | 0.505 | −0.085 |
| grouped2_resid G2=16 R=0.2 | 3.20 | 0.540 | −0.050 |
| graded_neutral G2=32 t=.10 R=.10 | 4.12 | 0.549 | −0.041 |
| graded_neutral G2=32 t=.10 R=.15 | 4.19 | 0.546 | −0.044 |
| **graded_neutral G2=16 t=.07 R=.10** (best) | 4.14 | 0.568 | **−0.022** |

## Verdict

**No — memory-neutral grouped grading does not match uniform-CQ4, at ~4.1 b/val.**

- **0.5B:** best neutral (G2=16, 4.14 b/val) = −0.026 vs uniform4 (4.0 b/val) = −0.015.
  Loses by 1.1 pt at *slightly more* memory.
- **1.5B:** best neutral = −0.022 vs uniform4 = −0.003. Loses by 1.9 pt.

The neutral ladders land at 4.1–4.2 b/val — a hair *above* the 4.0 uniform-CQ4
operating point — and still lose. Tuning them down toward exactly 4.0 only starves
more tokens to grouped-2bit, which would widen the gap, not close it. Promoting the
top ~10% to FP16 does not recover what dropping the boring 75–80% to grouped-2bit
costs — the same failure mode as the naive experiment, just far less severe.

## What grouping *did* buy (the real win)

Grouped-INT2 is a large, genuine improvement over naive per-token INT2 — it just
doesn't cross the 4-bit floor:

1. **Grouping alone crushes the naive-2bit crater**, especially on 1.5B: naive
   uniform2 was −0.281; grouped2 is **−0.094 at 2.5 b (G2=32)** and **−0.042 at
   3.0 b (G2=16)**. On 0.5B, −0.089 → −0.045 / −0.036. The KIVI grouping is exactly
   the fix for the D=128 blow-up.
2. **The neutral ladder improved from −0.065 to −0.022 on 1.5B** (naive → grouped) —
   a 4.3-pt rescue. Grouping turns the 1.5B catastrophe into "slightly below uniform4."
3. **Finer grouping beats coarser:** G2=16 (3.0 b) > G2=32 (2.5 b) everywhere. The
   extra fp16 scale is worth more than the bit it costs at these scales.

## The residual window backfired

Adding a CQ-4 recent-token window made things **worse at fixed codec**:
grouped2_resid is below plain grouped2_uniform on *both* models despite spending
more bits (e.g. 0.5B: resid_g32_R.1 −0.062 at 2.65 b vs uniform_g32 −0.045 at 2.5 b).
Reason: the window replaces **grouped-INT2** tokens with **per-token CQ-4** tokens,
and at D=128 *grouping matters more than bit-depth* — per-token CQ-4 (one scale over
128 dims) is a **weaker** codec than grouped-INT2 (one scale per 16–32 dims). This
flips a KIVI assumption: KIVI's residual window is **FP16**; a **4-bit** per-token
residual window is a downgrade, not an upgrade. A useful residual window here would
have to be FP16 (expensive) or itself grouped.

## Takeaways

- **Best memory-neutral point:** G2=16, top 7% FP16, recent 10% CQ4 (4.14 b/val) —
  −0.026 / −0.022. **Still loses to uniform-CQ4** on both models.
- **Best G2 = 16** (finer grouping wins); **residual window at CQ4 hurts** (use FP16
  or grouped if you keep one at all).
- **The 4-bit floor holds.** A smart 2-bit codec makes low-bit values *survivable*
  (no more cratering) but does not make graded value demotion memory-neutral. For the
  shipped codec, graded value grading remains a memory *tax*, not free — and uniform
  CQ-4 at 4.0 b/val is the best ~4-bit operating point.
- Hardware note unchanged: the KVCE RTL still packs only INT4/INT8; grouped-INT2
  would need a new 2-bit pack tier *and* per-group scales, and this study says that
  investment does not buy a memory-neutral accuracy win at 0.5B–1.5B.

Reproduce: `python analysis/graded_grouped2.py --model Qwen/Qwen2-0.5B --n 1000`
