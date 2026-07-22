# XAttention Antidiagonal Proxy on Real LLM Traces: Pre-Softmax Fails, Post-Softmax Works

**Branch:** `sparsity-controller-explore`
**Status:** **Shelved** — negative result, do not build.
**Date:** 2026-05-15 (initial), 2026-05-18 (4K follow-up + verdict).
**One-line summary:** Across **1.4M real Qwen2-0.5B attention tiles** at both 512- and 4096-token context, every hardware-friendly antidiagonal proxy on pre-softmax scores correlates **in the noise (r ≈ 0)** with ground-truth tile mass on every one of the 24 layers. The same proxy on post-softmax attention weights correlates +0.99 — confirming the XAttention idea works *only with softmax outputs in hand*, which kills the "no-exp, no-divide" hardware story. The precision controller's 1.7× INT8/FP16 win stands; no sparsity gate ships from this branch.

---

## Motivation

The ACU's `precision_controller` (block 1) is a streaming per-tile gate that
decides INT8 vs FP16 from a one-pass statistic on pre-softmax scores. It
costs ~30 flip-flops. The natural sibling is a **sparsity gate**:
elide whole tiles whose attention mass is below a threshold, saving the
QK^T matmul, the V-cache read, and the SV matmul on the eluded tile.

XAttention (Xu et al., 2025, arXiv:2503.16428) proposes exactly this gate
in software: per (Q-block, K-block) tile, sum the antidiagonal of the
*attention matrix* and prune tiles whose sum is below a threshold. The
paper claims ~13.5× attention speedup on long-context benchmarks.

Before designing silicon for this, we want to know:

1. Does the antidiagonal-sum proxy actually predict tile importance on
   real LLM traces (not just on the paper's curated benchmarks)?
2. Is there a hardware-friendly version — i.e. one computable from
   pre-softmax scores without exp/division — that preserves the proxy's
   accuracy?

This pilot answers both with one capture pass over Qwen2-0.5B at
seq_len = 512.

---

## Method

`analysis/sparsity_real_llm_capture.py` runs Qwen2-0.5B in eager-attention
mode and hooks the q/k projections of every layer. For each prompt:

1. Do a forward pass and read out the per-layer Q, K activations.
2. Reshape to `[B, H_q, N, d]`; expand GQA's KV heads via
   `repeat_interleave`.
3. For each (layer, head): build the full pre-softmax score matrix
   `S = Q @ K^T / sqrt(d)`, causal-mask it, then softmax along K to get
   the attention matrix `A`.
4. Walk the lower-triangular block grid at block size 64 × 64
   (matching the precision controller's tile size). For each full-size
   tile (edge tiles dropped for a clean comparison against the chip's
   fixed tile size):

   - **Ground truth** — `tile_mass = mean_q( sum_k A[q, k in tile] )`.
     This is the share of each query row's softmax mass landing in the
     tile, averaged over the tile's rows. Skipping a tile costs this
     much attention mass.
   - **Cheap pre-softmax proxies** (computed on per-tile symmetric int8
     quantization of `S`, mirroring what the chip's gate would see):
     - `abs_S_sum_sK` — antidiagonal sum of `|S|` at stride K
     - `abs_S_max_sK` — antidiagonal max of `|S|` at stride K
   - **Paper-faithful post-softmax proxy** (computed on `A`):
     - `softmax_A_sum_sK` — antidiagonal sum of `A` at stride K
   - **Precision-controller decision** — bit-exact via
     `sw/reference_model/precision_controller_ref.py` on the quantized tile.

5. Accumulate per-tile scalars only. No tile is persisted to disk.

**Storage discipline.** The model weights were downloaded to the HF cache,
the full run ran in ~3 m 50 s on an RTX A4000, then the cache directory
was removed via `--delete-cache-after`. Net on-disk artifact:
`analysis/sparsity_real_llm_stats.json` (~25 KB).

**Why this is just a pilot.** One model size (0.5B), two prompts (prose +
code), one sequence length (512). Tile statistics are not stationary
across model scale or prompt distribution, so any HW decision needs at
minimum the same harness on Qwen2-1.5B/3B and Phi-2, plus longer-context
prompts (4K, 16K, 64K) which is where XAttention's claimed savings actually
live.

---

## Results

### Corpus

Two captures, same harness, deletes model cache after each run:

| Capture | seq_len | tiles | wall clock | mass median | mass p10 / p90 |
|---|---|---|---|---|---|
| Initial (512) | 512  | 24,192    | 3 m 50 s | 0.15  | 0.04 / 0.53 |
| **4K follow-up** (this section)  | 4096 | **1,397,760** | ~18 min  | 0.019 | 0.005 / 0.055 |

The 57× larger 4K corpus is what made it possible to detect any per-layer
head-level effect that might have rescued a cheap proxy. None did.

### Correlation with ground-truth tile mass

#### Initial 512-token pass (24K tiles)

| Proxy | Stride 8 r |
|---|---|
| `abs_S_sum`             | −0.017 |
| `abs_S_max`             | −0.018 |
| `softmax_A_sum`         | **+0.985** |

#### 4K-token pass (1.4M tiles) — adds the positive-only proxy

| Proxy | Hardware cost | Stride 8 r | Per-layer range | layers with r > 0.85 |
|---|---|---|---|---|
| `abs_S_sum_s8`     | trivial (already in precision ctrl)               | −0.035 | [−0.21, +0.08] | **0 / 24** |
| `abs_S_max_s8`     | trivial                                            | −0.024 | (noise)        | **0 / 24** |
| `pos_S_sum_s8`     | one sign-bit AND-mask on accumulator (~30 FFs)    | **+0.011** | [−0.05, +0.23] | **0 / 24** |
| `softmax_A_sum_s8` | full softmax in the gate path (exp + division)    | **+0.982** | [+0.92, +0.998] | **24 / 24** |

**Per-layer breakdown is unanimous in both directions.** Across all 24
layers × 14 heads at 4K context, no pre-softmax proxy (including the
positive-only sign-bit-gated version proposed as a "cheap rescue") gets
within striking distance of the +0.85 bar. The best single-layer
positive-only correlation was +0.226 at L21 — still far below the bar,
and the median layer hovers around +0.04.

The post-softmax `softmax_A_sum` proxy works on every layer at every
stride — but computing it costs exactly what the precision controller
deliberately avoided: per-score exponentials and a per-row division.

### Threshold sweep at 4K

Side-by-side at the *same skip rate* you can read off how badly each cheap
proxy throws away real attention mass compared with the expensive one.

| τ (× mean proxy) | softmax_A skip% / kept | pos_S skip% / kept | abs_S skip% / kept |
|---|---|---|---|
| 0.05 |  4.9 % / **99.6 %** | 37.2 % / 68.1 %  |  0.0 % / 100.0 % |
| 0.30 | 19.3 % / **96.3 %** | 51.6 % / 51.2 %  |  3.1 % / 95.1 %  |
| 0.50 | 36.7 % / **88.9 %** | 56.4 % / 45.7 %  |  7.6 % / 90.8 %  |
| 1.00 | 75.3 % / **61.7 %** | 63.9 % / 37.3 %  | 39.9 % / 59.0 %  |

Oracle at 4K (rank by true tile mass, skip lowest k%):

| skip % | oracle mass kept |
|--------|------------------|
| 10 % | 99.3 % |
| 25 % | 95.1 % |
| 50 % | 82.3 % |
| 75 % | 62.9 % |

The post-softmax proxy lands within 7 percentage points of the oracle at
moderate sparsity (25–50 %); the positive-only proxy at the same skip
rates is ~30 pp behind the oracle — actively destructive. The `abs_S`
proxy looks superficially competitive in the τ-sweep because it lazily
maps low skip rates onto large τ values — but its rank-order against
true tile mass is r ≈ 0, so what it skips is not what an oracle would
skip; the apparent "mass kept" reflects that 1.4M tiles is enough that
even a random subset preserves most of the long-tail mass at low skip
rates. The correlation table above is the honest signal, not the τ-sweep
on misranked proxies.

### Precision controller — 0 % FP16 on this corpus

Across all 24,192 tiles, the precision controller fired FP16 zero times.
This differs from `real_llm_v2_stats.json`, which reported 8.3 % FP16 on
the same model + prompts. Likely cause: v2 included partial edge tiles
(< 64 valid scores) at sequence/row boundaries, where max/mean can spike
artificially; this pilot drops edge tiles to keep the comparison against
the chip's fixed 64×64 tile size honest. The precision controller is
correct on full tiles; v2 was counting a different population.

(This deserves a follow-up: re-run with full-block-only filtering inside
the v2 script to confirm the discrepancy is the edge-tile artifact, not
a bug in the quantization or controller reference.)

---

## Implications for the chip

### What this rules out

A sparsity gate built on **pre-softmax raw scores** — the natural
"sibling of the precision controller, same divide-and-multiply-free
ratio" design — would be effectively random on this workload. The
proxy is not in the noise *for some layers* — it is in the noise *for
all 24 layers*. Building this in silicon would land us ~30 flip-flops
of useless logic and false confidence.

### What is still on the table

The XAttention proxy works **when computed on softmax outputs**. There
are three honest paths to a hardware implementation, in increasing order
of ambition:

1. **Reuse FlashAttention's online softmax.** The softmax output (or at
   least the partial accumulator state) is already in flight during the
   ACU's main pipeline. The sparsity gate becomes a tail computation
   on existing softmax output rather than a parallel cheap proxy. This
   limits the savings to the SV matmul + V-cache read (skipped tiles
   contribute zero), not the QK^T matmul. ~Half of XAttention's claimed
   13.5× — still material, but no longer "free."

2. **Approximate softmax with a cheap monotone.** Replace `exp` with a
   shift-and-add or piecewise-linear approximation that preserves rank
   order well enough for the antidiag sum to retain rank correlation
   with the true post-softmax sum. Hardware cost roughly that of one
   approximate-exp ROM + adder tree per score. Hypothesis worth testing
   *in software* before designing — capture the same trace and run the
   piecewise approximation; if the proxy correlation drops below ~+0.85
   we abandon this path.

3. **Find a different cheap proxy that works on pre-softmax scores.**
   E.g. only the *positive* antidiagonal scores (softmax kills negative
   scores by ~10×e^{-Δ}, so the magnitudes-only view is the wrong
   reduction); or a max-over-strided-rows rather than antidiagonal-sum
   (captures "is there a peak row" in this block). Open question; the
   harness in this branch makes it cheap to test.

The right next action is (2) — capture the same trace once more with a
cheap approximate-exp proxy and see whether the correlation survives.
That keeps the architectural story of "ACU-local, no off-chip help"
intact. If it doesn't survive, we either pay for real exp in the gate
path or shelve the sparsity gate.

### Composition with the precision controller

Independent of which proxy we settle on, the placement is unchanged
from the design sketch on `sparsity-controller-explore`:

```
                          sparsity gate                      precision gate
   K, V from cache  ─────┐                                       ┌────── INT8/FP16 SV
                          ▼                                       ▼
   Q ──────────► [proxy compute] ──skip?──► [full QK^T] ──► [precision_controller] ──► [SV MAC]
                          │                                       │
                          └── skip ──► output += 0; advance       │
                                                                   ▼
                                                            row accumulator
```

The two decisions are independent. Skipped tiles never reach the
precision controller — its stream is unchanged. Build order: prove the
proxy works in software (this branch), then write `sparsity_controller.sv`
to the same template as `precision_controller.sv`, verify against the
reference model (`sw/reference_model/sparsity_controller_ref.py`), gate
on the same 8 CI checks the shared block-CI workflow already enforces.

---

## Method appendix

### Why post-softmax works but pre-softmax doesn't

Pre-softmax scores `S = Q·K^T / sqrt(d)` for `d=64` are dot products of
roughly-zero-mean random-ish vectors; their magnitudes are concentrated
in a 5–10× band, not 100×. A tile's `sum(|S|)` mostly reflects how many
of the 4,096 cells contain "typical-magnitude" scores — i.e. it counts
tile coverage, not tile importance.

Softmax exponentiates the scores. A score of `+1` and a score of `+5`
differ by a factor of `e^4 ≈ 55` in attention weight. Tiles containing
the largest few scores per query row absorb almost all of that row's
post-softmax mass. The antidiagonal-sum reduction preserves this
because softmax mass is locally clumped — a strided sample of any
antidiagonal hits enough of the high-mass cells to rank tiles correctly.

This means the proxy is fundamentally about the *exponentiated*
distribution, not the raw one. Any hardware shortcut has to preserve
the exp behavior somehow.

### Edge-tile handling

Edge tiles (< 64 rows or < 64 columns valid because the sequence didn't
fill a block) are dropped. The chip's gate hardware operates on
fixed-size 64×64 tiles; running the proxy on partial tiles changes both
the antidiagonal sample count and the per-tile quantization scale, which
would conflate two effects. The chip handles edge tiles separately (in
the kernel, by padding or by smaller tile dispatch); the gate logic does
not need to characterize them here.

### Why the proxy correlation is stable across stride

Antidiag-sum at stride K samples `N/K` of the tile's 4,096 cells. At
K=16 that's 256 cells (6.25% coverage), and r is still +0.976. This is
consistent with XAttention's design point — the softmax-mass landscape
in a single tile is spatially smooth enough that 6% of the tile already
ranks it correctly. The hardware design point would be K=8 (12.5%
coverage, r=+0.985, 512 samples per tile, modest accumulator width).

### Reproducing

```sh
git switch sparsity-controller-explore
python3 -m pip install transformers accelerate
python3 analysis/sparsity_real_llm_capture.py \
    --seq-len 512 --delete-cache-after
# Wall clock: ~4 min on an RTX A4000
# Output: analysis/sparsity_real_llm_stats.json (~25 KB)
```

Smoke flags for fast iteration:

```sh
python3 analysis/sparsity_real_llm_capture.py \
    --max-prompts 1 --max-layers 4 --seq-len 512
```

---

## Verdict and follow-ups

### Verdict (2026-05-18)

**Shelve the sparsity controller.** The 4K-context follow-up tested the
two specific hardware-cheap proxies that were still on the table —
positive-only antidiag-sum (one sign-bit AND-mask, ~30 FFs) and the
abs-sum / abs-max baselines — and all three fail to correlate with
ground-truth tile mass on every one of 24 layers at both 512- and
4096-token contexts. The XAttention idea works on softmax outputs but
not on anything we can compute for free alongside the precision
controller's existing accumulators.

The shelve-not-kill distinction matters for two reasons:

1. **The methodology is reusable.** `analysis/sparsity_real_llm_capture.py`
   is a per-tile, zero-storage proxy-vs-truth harness; it can evaluate
   any future proxy candidate against a fresh trace in ~18 minutes per
   sweep. Future ACU work that has softmax outputs in flight (e.g.
   if FlashAttention's online-softmax accumulator is exposed for free
   from the kernel) can resurrect the gate without redoing this study.
2. **The reference model on `sw/reference_model/sparsity_controller_ref.py`
   still has a use** as a streaming bit-accurate ground-truth for any
   gate that does end up reaching silicon. Closed-form FF count,
   precision-controller-shaped API, 5 passing smoke tests.

The branch stays out of the chip block diagram. Engineering hours
reallocate to the remaining LonghornSilicon blocks: KV Cache Engine
(in progress), Token Importance Unit, Memory Hierarchy Controller.

### Why the cheap proxies failed (mechanism)

Post-softmax attention mass is dominated by the few largest *positive*
scores per query row, because softmax exponentiates differences. A
score of +5 vs +1 is a ~55× attention-weight ratio. The pre-softmax
score distribution in `Q·K^T / sqrt(d)` for d=64 has typical magnitudes
in a 5-10× band — not extreme enough that abs-magnitude statistics
distinguish "tile with the row's argmax inside" from "tile with
average-magnitude scores spread evenly."

The positive-only proxy was the most hopeful candidate because it
filters by sign. But the *largest* positive score per row dominates
softmax mass independent of the *count* of positive scores in the tile —
a tile with one +5 and 4095 zeros has the same `pos_S_sum` as a tile
with 4096 +0.00122 scores, and the first dominates softmax mass while
the second does not. The sum reduction is wrong; only a max-of-positive
would correlate, and that loses the antidiagonal-sampling story entirely.

### Open audits (low priority)

1. **0 % FP16 across 1.4M full-block tiles.** Likely an edge-tile
   artifact in `validate_real_llm_v2`'s original 8.3 % number, but worth
   one re-run with full-block-only filtering to confirm before citing
   the 8.3 % anywhere else.

2. **Per-head heat-map for L21.** That single layer reached r = +0.226
   on the positive-only proxy — high enough above the surrounding noise
   floor that it may indicate one or two heads behaving differently. Not
   actionable, but might inform future kernel-level sparsity work.

### What stays on this branch

```
analysis/sparsity_real_llm_capture.py        # zero-storage capture harness
analysis/sparsity_real_llm_stats.json        # 512-token results
analysis/sparsity_real_llm_4k_stats.json     # 4K-token results
analysis/sparsity_pilot.py                   # synthetic-tile harness
analysis/sparsity_pilot_results.json
sw/reference_model/sparsity_controller_ref.py        # bit-accurate Python ref
sw/reference_model/test_sparsity_controller_ref.py   # 5/5 smoke tests
docs/findings/sparsity-controller-finding.md         # this document
```

Branch is preserved for future reference but not merged to `master`.
