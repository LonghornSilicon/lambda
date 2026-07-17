# H2O Accumulated-Mass Token Retention on Qwen2 — Analysis Phase

**Status:** Complete — algorithm validated, proceed to RTL.
**Date:** 2026-07-17.
**One-line:** H2O (keep recent window + top heavy-hitters by accumulated post-softmax
attention mass) holds HellaSwag acc_norm within −0.006 of a full KV cache down to a
25% cache budget on Qwen2-0.5B; it degrades sharply below ~15%.

---

## Motivation

Block 3 (Token Importance Unit) must keep the KV cache within a fixed on-die budget
as context grows. That requires a per-token importance signal and an eviction policy.
The ACU sparsity study already settled the *signal*: **post-softmax attention mass
predicts token importance (r≈0.99); pre-softmax proxies do not (r≈0)**. H2O is the
canonical policy built on that signal, and is what we're carrying into silicon.

## Method

`analysis/h2o_analysis.py` registers a custom Qwen2 attention (transformers 5.x
`AttentionInterface`) that, per (layer, head):

1. Computes the causal post-softmax attention matrix `A` (scores in fp32 — fp16
   QK^T over D=128 overflows to NaN).
2. Accumulates each key's received mass `acc[i,j] = Σ_{q≤i} A[q,j]` (`A.cumsum`).
3. For a fixed cache budget of C tokens (fraction of sequence length), once the
   sequence exceeds C, retains a recent window of L = C/2 tokens plus the top
   (C − L) heavy hitters by `acc`, and masks out the rest, renormalizing `A`.

Evaluated on HellaSwag (n=500) across budget fractions 1.0 → 0.10.

## Result

| KV budget | acc_norm | Δ vs full |
|---|---|---|
| 1.00 | 0.498 | — |
| 0.75 | 0.490 | −0.008 |
| 0.50 | 0.496 | −0.002 |
| 0.35 | 0.496 | −0.002 |
| 0.25 | 0.492 | −0.006 |
| 0.15 | 0.454 | −0.044 |
| 0.10 | 0.376 | −0.122 |

Near-lossless (|Δ| ≤ 0.006) down to a **25% budget**; a sharp knee below ~15%.
Consistent with the H2O paper's ~20%-cache claim, now on Qwen2.

## Implications for the RTL

- **Datapath = accumulator + streaming top-k.** Per token: one add to its running
  mass; per step: maintain the top-(C−L) set. This is the same streaming shape as
  the precision controller (block 1) — a small, closed-form-FF datapath.
- **Budget target ≈ 25–30% of context** for near-lossless operation; expose C and
  the recent-window ratio L/C as parameters.
- **Integration:** map importance rank to a KV Cache Engine tier
  (keep→CQ-8, demote→CQ-4, evict→drop) rather than a hard keep/drop — "mixed-precision
  retention." Quantifying the accuracy of graded demotion (vs binary evict) is the
  recommended next analysis.

## Next

- Per-head vs shared-budget ablation (H2O is per-head here; shared budget is cheaper HW).
- Graded demotion (tier mapping) accuracy vs binary eviction.
- Longer-context trace (the HellaSwag knee is short-sequence-limited; confirm on 2–4K).
- Fixed-point / integer accumulator precision study before RTL.
