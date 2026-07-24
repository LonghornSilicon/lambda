# H2O Deep Analysis — Long-Context Knee, Per-Head vs Shared, Accumulator Bit-Width

**Status:** Complete — three follow-up studies to `h2o-analysis.md`, all sizing the RTL.
**Date:** 2026-07-18.
**One-line:** On genuinely long context the KV-budget knee is sharp and 25% stays
near-lossless; the cheaper **shared** budget is **not** viable at that budget (−0.04
HellaSwag), so keep per-head selection; and an **8-bit** fixed-point accumulator
(**SCORE_WIDTH = 8**) is loss-free, 6 bits is not.

All on Qwen2-0.5B, gold config carried from `h2o-analysis.md` (recent_ratio = 0.5).
Scripts: `analysis/h2o_longctx.py`, `analysis/h2o_perhead_vs_shared.py`,
`analysis/h2o_accum_bits.py`. Results: the matching `*.json` in `analysis/`.

---

## 1. Long-context knee — WikiText-2 perplexity

HellaSwag sequences are short, so eviction barely triggers there (kept ≈ 0.97 even at
a 25% budget) and the knee is soft. This test concatenates WikiText-2-raw test into
non-overlapping 1024-token windows (131,072 tokens total) and measures perplexity
under H2O eviction vs KV budget — every window is now long enough that the budget
actually bites.

| KV budget | perplexity | ×full | kept frac |
|---|---|---|---|
| 1.00 | 14.789 | 1.000 | 1.000 |
| 0.50 | 14.794 | 1.000 | 0.750 |
| 0.35 | 14.838 | 1.003 | 0.577 |
| 0.25 | 14.945 | 1.010 | 0.437 |
| 0.15 | 15.387 | 1.040 | 0.278 |
| 0.10 | 16.108 | 1.089 | 0.189 |
| 0.05 | 18.583 | 1.257 | 0.097 |

The `kept frac` now genuinely reflects eviction (0.44 at a 25% budget on 1024-token
windows, vs ~0.97 on short HellaSwag). Perplexity is flat down to 50% (+0.0%),
near-flat at 25% (**+1.0%**), then bends: +4.0% at 15%, +8.9% at 10%, +25.7% at 5%.

**Takeaway:** on real long context the 25% gold budget is near-lossless (+1% ppl) and
the knee sits around 10–15% — a much sharper, cleaner knee than HellaSwag showed, and
it confirms the 25% budget target with margin.

## 2. Per-head vs shared budget

`h2o_analysis.py` selects heavy hitters **per head**: each (layer, head) keeps its own
top-(C−L) tokens, so eviction differs head-to-head and the HW tracks a keep-set per
head. A **shared** budget ranks tokens by attention mass **summed over heads** and
keeps one per-layer keep-set every head shares — 1/H the accumulator/top-k width and a
single eviction index per layer (whole KV rows drop). Cheaper silicon; does it cost
accuracy? HellaSwag acc_norm, n=500 (fp16 full cache = 0.498):

| KV budget | per-head | shared | gap (shared − per-head) |
|---|---|---|---|
| 0.50 | 0.496 | 0.500 | +0.004 |
| 0.35 | 0.496 | 0.478 | −0.018 |
| **0.25 (gold)** | **0.492** | **0.452** | **−0.040** |
| 0.15 | 0.454 | 0.412 | −0.042 |

Shared matches per-head at a loose 50% budget, but the gap widens as the budget
tightens — at the gold 25% budget shared costs **−0.040** (well past the ±0.02
per-block gate), and −0.042 at 15%. Heads disagree on which tokens matter, and that
disagreement is exactly what a shared keep-set throws away when slots are scarce.

**Takeaway:** shared budget is **not viable** at the gold 25% budget — keep per-head
selection in the RTL. Shared is only acceptable at loose budgets (≥50%), where there
is no eviction pressure and thus little reason to prefer it.

## 3. Accumulator bit-width → SCORE_WIDTH

The RTL stores each token's accumulated post-softmax mass in a fixed-point accumulator.
Modeled as an unsigned uniform accumulator spanning [0, Tk] (mass can reach the context
length: Σ_j acc = i+1); B bits give 2^B−1 levels, round-to-nearest + saturate, applied
**before** the top-k heavy-hitter selection. Gold config (25% budget), HellaSwag n=500.
fp32 accumulator reference = 0.492 (fp16 full cache = 0.498).

| SCORE_WIDTH | acc_norm | Δ vs fp32 accumulator |
|---|---|---|
| fp32 (ref) | 0.492 | — |
| 16 | 0.498 | +0.006 |
| 12 | 0.496 | +0.004 |
| 10 | 0.498 | +0.006 |
| **8** | **0.490** | **−0.002** |
| 6 | 0.470 | −0.022 |
| 4 | 0.456 | −0.036 |

Accuracy is flat (within noise, occasionally above the fp32 gold — quantization only
perturbs top-k ties) down to **8 bits** (Δ = −0.002). At 6 bits it falls off the cliff
(−0.022, past the gate) and 4 bits is clearly broken (−0.036). The knee is between 6
and 8.

**Takeaway / RTL sizing:** **SCORE_WIDTH = 8 bits** is the recommended accumulator
width — loss-free with margin, half the storage of a naive 16-bit register. If a
comfortable guard band is wanted (e.g. longer context than HellaSwag's short
sequences, where more tokens compete near the top-k boundary), **10 bits** buys margin
at trivial cost; do **not** go to 6.

---

## Summary for the RTL

| Question | Answer |
|---|---|
| KV budget target (long context) | 25% near-lossless (+1% ppl); knee ~10–15% |
| Shared budget viable? | **No** at 25% (−0.04 acc_norm) — keep **per-head** selection |
| Accumulator width | **SCORE_WIDTH = 8 bits** (loss-free; 6b breaks; 10b for margin) |

## Notes / fixes

- `h2o_longctx.py` corrects the kept-fraction accounting: the H2O `recent=(i−j)<L`
  mask is True for all future positions (j>i), so `keep = recent | heavy` counts the
  whole future. The attention math is unaffected (future `A` = 0 after softmax), but
  the reported kept-fraction was inflated (~0.998). The stat now ANDs `keep` with
  `causal` before counting, giving the true retained fraction (0.44 at 25% budget).
  The eviction/attention behavior is identical to `h2o_analysis.py`.
- All three scripts reuse the established conventions: fp32 QK^T scores (fp16
  overflows to NaN), self-built causal mask (transformers 5.x passes
  `attention_mask=None`), and `recent` AND causal for eligibility.
