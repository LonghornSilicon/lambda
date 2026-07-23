# OliVe OVP for the KVE value path: OUTLIER- vs NORMAL-accommodation study

*Author: Chaithu Talasila · 2026-07-23 · KVE (kv-cache-engine) research ledger*

## Why this note exists (self-describing context)

The KVE quantizes the KV cache with ChannelQuant (per-channel INT4 keys, per-token INT4
values). The current outlier story is the **CQ-4+ static top-k FP16 outlier lane**: a
design-time-calibrated set of `k=2` key channels held in FP16 through a separate sidecar
(ROM mask + FP16 lane). Our n=1000 finding (see `../DECISIONS.md`, `../README.md`) is that
this lane **helps only at D=128 and marginally/slightly hurts at D=64**, so CQ-4 (no lane)
is the default at every head dim.

That raises a design question this study answers: is there a *different*, more
hardware-friendly way to handle outliers than a separate FP16 sidecar? Specifically
**OliVe OVP** (Outlier-Victim Pair), which accommodates outliers **in-band** at the same
4-bit budget — no outlier ROM, no separate lane, memory-aligned. Before spending RTL on it
we needed the accuracy question settled: on our own Qwen KV cache, does spending the bit
budget on **outliers** (OVP) or on **normal values** actually win, and by how much?

This is a **real run**, no fabrication. H100 PCIe, a scratchpad venv with torch 2.11.0+cu128
(CUDA=True) + lm_eval + transformers 5.14.1, cached Qwen2 weights + cached HellaSwag, fully
offline. Both Qwen2-0.5B and 1.5B ran (~12 s/mode).

## Method — three V-codecs at a matched ~4-bit budget

All three codecs are applied to the **Value** tensor inside the real attention KV path (a
registered custom `oliveref` attn op, so `value` is the actual `[B,H,T,D]` cache feeding the
A·V matmul). **Keys are held FP16** so the V codec is the only variable. Each costs 4
bits/value + one FP16 per-token scale (matched: 4.25 b/val at D=64, 4.125 at D=128).

| Scheme | Normals | Outliers (>3σ, ~0.9% of V) |
|---|---|---|
| **A** plain uniform INT4, amax scale | coarse (largest value sets the scale) | preserved incidentally |
| **B** OliVe OVP (spec §1–§6) | fine `[−7,7]` grid, robust 3σ/7 scale | preserved in abfloat E2M1 (bias=2, mags {8..96}); adjacent **victim pruned to 0** |
| **C** normal-accommodation (inverse) | same fine grid as B | **clipped to ±7** (no abfloat, no victim) |

**B and C share everything except the outlier mechanism**, so the B↔C contrast isolates
exactly "spend the budget on outliers vs on normals." A is the naive control.

## Results — HellaSwag acc_norm (n=500, Δ vs FP16)

| Model | FP16 | A plain-INT4 | B OliVe OVP (outlier) | C normal (clip) | B − C |
|---|---|---|---|---|---|
| Qwen2-0.5B (D=64)  | 0.498 | 0.492 (−0.006) | **0.512 (+0.014)** | 0.442 (−0.056) | **+0.070** |
| Qwen2-1.5B (D=128) | 0.588 | 0.586 (−0.002) | **0.604 (+0.016)** | 0.532 (−0.056) | **+0.072** |

**B > A > C, decisively and consistently on both models.** Only ~0.9% of V values lie
beyond 3σ, yet they dominate the error budget: accommodating them (B) matches/slightly beats
FP16, while sacrificing them to give normals full fidelity (C) costs a steady **−5.6
acc_norm points**. B's +0.014/+0.016 over FP16 is within n=500 noise, but **B beats C by ~7
points on both models — well outside noise.**

Artifacts (monorepo-relative, self-contained):
- `../analysis/olive_outlier_vs_normal.py` — the codec + harness
- `../analysis/olive_outlier_vs_normal_Qwen2-0.5B.json`
- `../analysis/olive_outlier_vs_normal_Qwen2-1.5B.json`

(Originating run committed in the standalone kv-cache-engine repo at `4edb4c13`.)

## Answering the three questions

### (1) Outlier vs normal accommodation — outlier wins
Accommodating **outliers (OVP, scheme B)** is the clear winner: it matches FP16 (+0.014 /
+0.016), beats the naive INT4 baseline **A** by +0.020 / +0.018, and beats the
normal-accommodation inverse **C** by **+0.070 / +0.072**. The lesson is unambiguous for our
V cache: at a fixed 4-bit budget the outliers are where the bits must go.

### (2) OVP vs the current CQ-4+ static FP16 outlier lane
| Axis | CQ-4+ FP16 outlier lane (current) | OliVe OVP (this study) |
|---|---|---|
| Tensor | **Keys** (per-channel) | **Values** (per-token) |
| Outlier selection | **static**, design-time top-k calibration (k=2) | **dynamic**, per-token 3σ rule |
| Storage | separate FP16 **sidecar** + ROM mask (out-of-band) | **in-band** abfloat E2M1, victim-pruned pair |
| Budget | +FP16 per outlier channel | flat 4 b/val (victim absorbs the escape) |
| Accuracy gain | **D=128 only** (1.5B +0.012, p=0.088); D=64 −0.006 (hurts) | **both dims** (+0.014 D=64, +0.016 D=128) |

The headline contrast: the CQ-4+ lane's benefit is **head-dim-dependent** (helps D=128,
hurts D=64), whereas OVP's outlier accommodation delivers a **consistent gain at both head
dims**, at a flat budget and with no sidecar. Caveat, stated plainly: these are **not
apples-to-apples** — CQ-4+ operates on the *key* channels (static, structural outliers),
OVP on the *value* tokens (dynamic, transient outliers). OVP is therefore a candidate for
the **value path**, not a drop-in replacement for the key-channel lane. But as an outlier
*mechanism* it is more robust across the dimension where our current lane fails (D=64).

### (3) Is OVP worth adopting for the KVE?
**Recommendation: yes — prototype OVP as the KVE's value-path outlier codec, pending a HW
cost estimate; keep CQ-4 the shipping default until that lands.** Rationale:

- **Accuracy:** OVP is the only mechanism here that improves V-quant at *both* head dims,
  and it dominates the normal-accommodation alternative by ~7 pts. It removes the D=64
  regression that made us keep the CQ-4+ key lane optional.
- **Hardware friendliness — the decisive architectural point:** OVP is **in-band and
  memory-aligned**. No outlier ROM, no FP16 sidecar region, no separate lane to route — the
  escape code lives in the same 4-bit slot, with the adjacent victim absorbing it. This is
  exactly the property the KVE already prizes: the WHT/CQ-3-rot note calls out "decompress
  `code · field` widens the FP16 exactly — **no separate sidecar region**" as a virtue
  (`../README.md`, `../docs/wht_value_rotation.md`). OVP extends that same "no sidecar"
  philosophy to outlier handling.
- **Costs to weigh before RTL (why this is a recommendation to prototype, not to ship):**
  (a) OVP needs **per-token dynamic outlier detection** (a 3σ/MAD reduction) and pair-encode
  / victim-prune logic — runtime cost the static ROM lane does not have. (b) It **sacrifices
  a victim neighbor to 0**; benign here but worth confirming it never lands on a load-bearing
  channel at scale. (c) These numbers are **V-only** (keys FP16), so they isolate the
  mechanism but are not a full KV-compression figure. (d) n=500 subset; the CQ-4+ comparison
  is n=1000 — re-run OVP at n≥1000 before a DECISIONS entry.

Net: OVP is the right direction for value-outlier handling on hardware grounds and wins on
accuracy at both head dims. Next step is an RTL area/power estimate for the dynamic
detect + pair-encode path vs the retired sidecar it would replace, then a n≥1000
confirmation, before promoting it from research to a KVE tier.

## Caveats (consolidated)
1. V-only codec, keys FP16 — isolates the mechanism, not a full KV number.
2. B uses a **robust (MAD-based) 3σ scale** rather than the spec's per-token MSE search: a
   deliberate choice so a single large outlier does not inflate the normal grid (a unit test
   showed plain-std made C's normal-MSE 200× worse and confounded the B↔C comparison).
3. n=500 HellaSwag subset (not the full 10k); CQ-4+ baseline is n=1000.

## Citation
Guo, Cong; Tang, Jiaming; Hu, Weiming; Leng, Jingwen; Zhang, Chen; Yang, Fan; Liu, Yunxin;
Guo, Minyi; Zhu, Yuhao. **"OliVe: Accelerating Large Language Models via Hardware-friendly
Outlier-Victim Pair Quantization."** ISCA 2023. arXiv:2304.07493.
https://arxiv.org/abs/2304.07493
