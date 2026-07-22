# Can a sub-4-bit VALUE tier make graded demotion memory-neutral?

**Status:** Tested — **no, not with a naive INT2 tier.** Graded value demotion is a
memory *tax*, not free; trying to offset it by 2-bitting low-importance tokens
craters accuracy (badly at D=128). The 4-bit floor is a hardware-packing limit AND,
for naive per-token quant, close to a practical accuracy floor.
**Date:** 2026-07-18.

## The question

The graded value ladder (`analysis/full_stack_integration.py`) only *promotes* —
top 10% → FP16, next 15% → CQ-8, rest → CQ-4/CQ-4+. Floor is 4 bits; nothing goes
below. So its average is **5.88 b/val vs uniform CQ-4's 4.0** — graded buys ~+1pt by
spending ~47% more value memory, not for free. If "boring" tokens could drop to
2 bits, grading could be **memory-neutral**: promote heavy hitters, starve the rest,
break even on bits. Is 4-bit an accuracy floor (2-bit craters) or just a codec/HW
limit (no sub-4-bit tier)?

## Answer

- **Codec math:** no barrier — the reference `_q_per_token(x, bits)` is bit-parameterized,
  so INT2 is a one-line change.
- **Silicon:** the KVCE RTL packs only INT4/INT8 (`pack_int4`/`pack_int8`) — **no 2-bit
  tier exists in hardware.** That is the real "4-bit floor."
- **Accuracy:** measured below — naive 2-bit is a genuine accuracy floor, and demoting
  low-mass tokens to 2-bit does **not** escape it.

## Result (HellaSwag acc_norm, n=1000, keys held uniform CQ-4+ to isolate the value ladder)

| value ladder | avg b/val | Qwen2-0.5B Δ vs FP16 | Qwen2-1.5B Δ vs FP16 |
|---|---|---|---|
| fp16 | 16.0 | — | — |
| **uniform4** (today's floor) | 4.0 | −0.015 | −0.003 |
| uniform2 | 2.0 | −0.089 | −0.281 |
| graded_promote (current design) | 5.88 | −0.004 | −0.009 |
| **graded_neutral** (top10 FP16 / 30 CQ4 / bottom60 CQ2) | 4.11 | −0.019 | −0.065 |

1. **Uniform 2-bit craters** (−0.089 / −0.281 — near random at 1.5B); worse at larger
   D. Confirms the KIVI/KVQuant 4-bit floor.
2. **Memory-neutral graded doesn't pay off.** At ~equal memory, `graded_neutral` is a
   wash on 0.5B (−0.019 vs uniform4 −0.015) and a 6.5-pt loss on 1.5B. Demoting the
   boring 60% costs more than promoting the top 10% recovers.
3. **Graded's +1pt is a memory tax.** It only helps at the promote-only 5.88 b/val
   operating point; you cannot break even by starving.

## Why the down-weighting intuition breaks

"Boring by *total* accumulated mass" ≠ boring for *every* query — attention is peaky
and query-specific, so a low-total-mass token can still be some query's top attendee.
2-bit error there corrupts that query's output, and across 60% of tokens those errors
aggregate. Bigger models (D=128) have richer value spaces that 2-bit destroys harder.

## Caveat / open door

This is *naive* per-token symmetric INT2 (no grouping, no residual). KIVI/KVQuant make
2-bit viable with **per-group** quant + a small FP16 **recent-token residual window**.
A smarter 2-bit tier *might* make memory-neutral grading work — but it needs a real
2-bit path in the RTL (absent today) and is unproven. For the shipped codec, graded
value demotion trades memory for accuracy; it is not free and not memory-neutral.

Reproduce: `python analysis/graded_2bit.py --model Qwen/Qwen2-0.5B --n 1000`
