# Token Importance Unit

This is the **Token Importance Unit (TIU)** block of the LonghornSilicon LLM
inference accelerator — **block 3 of four** targeting TSMC 16nm FinFET (N16FFC)
tape-out. It decides, per cached token, whether to **keep, demote, or evict** its
KV entry — so the KV cache stays within a fixed on-die budget as context grows.

> **Status: analysis phase.** The retention algorithm (H2O accumulated-mass) is
> validated on real Qwen2 traces (below). RTL, synthesis, and Sky130 sign-off are
> the next milestones, following the pattern of
> [`adaptive-precision-attention`](https://github.com/LonghornSilicon/adaptive-precision-attention)
> (block 1) and [`kv-cache-engine`](https://github.com/LonghornSilicon/kv-cache-engine) (block 2).

---

## TL;DR

| | |
|---|---|
| **What** | Per-token importance scorer + eviction/demotion controller for the KV cache |
| **Why** | KV cache grows linearly with context; a fixed on-die budget needs a policy for *which* tokens to drop first |
| **How** | **H2O** — accumulate each token's post-softmax attention mass; keep a recent local window + the top "heavy-hitter" tokens by accumulated mass; evict the rest |
| **Signal** | Post-softmax attention mass (the ACU sparsity study proved pre-softmax proxies fail at r≈0, post-softmax works at r≈0.99) |
| **Integration** | Emits the **tier signal** that the KV Cache Engine already consumes (keep → CQ-8, demote → CQ-4, evict → drop) — mixed-precision retention |
| **Verified (algorithm)** | HellaSwag acc_norm within **−0.006** of full cache down to **25% KV budget** on Qwen2-0.5B (n=500) |
| **Status** | Analysis phase complete; RTL next |

---

## How H2O works

The **Heavy-Hitter Oracle** (Zhang et al., 2023) observation: attention mass is
highly concentrated — a small, stable set of tokens receives most of the attention
across the whole sequence. Track them and you can throw the rest away.

Per (layer, head), for each cached key token *j*:

1. **Accumulate** its received attention mass: `acc[i,j] = Σ_{q≤i} A[q,j]`
   (a running sum, one add per token per step — cheap and streaming).
2. Maintain a fixed cache budget of **C** tokens. Once the sequence exceeds C, keep
   - a **recent local window** of L tokens (recency matters for coherence), plus
   - the **top (C − L) heavy hitters** by accumulated mass,

   and **evict** everything else. Evicted tokens' K/V are never attended to again.

The scorer is an accumulator + a running top-k — a natural streaming datapath, the
same shape as the precision controller (block 1). This is prior art as an
*algorithm*; **the contribution of this block is the streaming silicon
implementation** and its integration with the ChannelQuant tier signal.

---

## Algorithm result — verified on Qwen2

HellaSwag `acc_norm`, n=500, H2O eviction applied to every layer/head of
Qwen2-0.5B (recent-window share = 50% of budget). "KV budget" is the cache size C
as a fraction of the sequence length:

| KV budget | acc_norm | Δ vs full cache |
|---|---|---|
| 100% (full) | 0.498 | — |
| 75% | 0.490 | −0.008 |
| 50% | 0.496 | −0.002 |
| 35% | 0.496 | −0.002 |
| **25%** | **0.492** | **−0.006** |
| 15% | 0.454 | −0.044 |
| 10% | 0.376 | −0.122 |

**H2O holds accuracy to within −0.006 of the full cache down to a 25% KV budget,
then falls off sharply below ~15%.** This sizes the block: a cache of ~25–30% of
context is near-lossless on this workload. (The classic H2O ~20% result,
reproduced on Qwen2.)

Reproduce:

```sh
python analysis/h2o_analysis.py --model Qwen/Qwen2-0.5B --n 500
```

Gold config: **recent-window ratio 0.5, KV budget 25%** (an even recency/heavy-hitter
split wins at every budget; see `docs/findings/h2o-analysis.md`).

---

## All three blocks together

The TIU is the last of the three live blocks. Composed in chip order — KVCE
decompress → scores → **TIU** keep/evict → APA route — on Qwen2 (HellaSwag n=1000,
gold config), Δ vs the FP16 full-cache baseline:

| config | Qwen2-0.5B | Qwen2-1.5B |
|---|---|---|
| TIU evict (25% budget) | −0.016 | −0.034 |
| KVCE (cq4+) | −0.015 | −0.003 |
| APA | +0.001 | −0.002 |
| **ALL 3 stacked** | **−0.033** | **−0.030** |
| ALL 3 + graded value demotion | −0.023 | −0.029 |

Stacking **75% cache eviction × 4-bit KV × ~all-INT8 attention** costs only ~3%
acc_norm. Two findings (`docs/findings/all-three-blocks-integration.md`): per-token
*key* demotion is incompatible with ChannelQuant's per-channel key path (keys stay
uniform per-channel), but per-token *value* demotion — the "mixed-precision
retention" payoff — recovers ~1pt by spending bits on heavy hitters.

```sh
python analysis/full_stack_integration.py --model Qwen/Qwen2-0.5B --n 1000 --frac 0.25 --recent_ratio 0.5
```

---

## How this fits in LonghornSilicon

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (16FFC)       │
│                                                                      │
│   ┌──────────────────┐          ┌────────────────────────┐          │
│   │  ACU (block 1)   │  scores  │  Token Importance Unit  │          │
│   │  precision ctrl  │─────────▶│  (this repo, block 3)   │          │
│   │  INT8 vs FP16    │          │  H2O accumulated mass   │          │
│   └────────┬─────────┘          │  → keep / demote / evict│          │
│            │  K, V              └───────────┬────────────┘          │
│            ▼                                │ tier signal            │
│   ┌─────────────────────────┐               ▼                       │
│   │  KV Cache Engine        │◀───── keep→CQ-8 / demote→CQ-4 / evict │
│   │  (block 2) ChannelQuant │                                       │
│   └─────────────┬───────────┘                                       │
│                 ▼                                                     │
│   ┌─────────────────────────┐   ┌──────────────────────┐             │
│   │ Memory Hierarchy Ctrl.  │◀─▶│ Off-chip LPDDR5X      │             │
│   │ (block 4)               │   │ (cold KV + weights)   │             │
│   └─────────────────────────┘   └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

The TIU closes the loop on the KV cache: the ACU produces attention scores → the
TIU accumulates per-token importance and rules keep/demote/evict → the KV Cache
Engine applies the resulting precision tier (or frees the slot). Together the three
live blocks turn a linearly-growing FP16 KV cache into a **bounded, mixed-precision**
one.

| Block | Repo | Role |
|---|---|---|
| ACU (Attention Compute Unit) | [adaptive-precision-attention](https://github.com/LonghornSilicon/adaptive-precision-attention) | INT8 vs FP16 per tile, MAC array |
| KV Cache Engine | [kv-cache-engine](https://github.com/LonghornSilicon/kv-cache-engine) | ChannelQuant compress/decompress |
| **Token Importance Unit** | **this repo** | Per-token keep/demote/evict (H2O) |
| Memory Hierarchy Controller | not yet | On-die SRAM ↔ off-chip LPDDR5X |

---

## Repo layout

```
token-importance-unit/
├── analysis/          # Python: H2O algorithm study, trace capture, test-vector gen
│   ├── h2o_analysis.py                 # accuracy vs KV-budget sweep (this is the result above)
│   └── h2o_qwen05b_n500.json           # measured curve
├── rtl/               # SystemVerilog DUT + testbenches (next milestone)
├── openlane/          # LibreLane Sky130 sign-off (next milestone)
├── paper/             # block write-up
└── docs/              # design notes, CI docs
```

## Roadmap

- [x] Algorithm validated (H2O accumulated-mass on Qwen2; near-lossless to 25% budget)
- [x] Gold config chosen (recent-ratio 0.5, 25% budget)
- [x] All-3-blocks integration verified (TIU+KVCE+APA compose within ~3% of FP16)
- [x] Deep analysis: long-ctx knee, per-head vs shared (keep per-head), accumulator width (10b)
- [x] RTL: accumulator + serialized-argmin eviction datapath, closed-form FF count (113 FFs)
- [x] Directed + randomized self-checking testbench (iverilog), 20/20 bit-exact
- [x] Yosys synth FF-count + Sky130 first pass: DRC/LVS/antenna/timing clean (`docs/sky130_signoff.md`)
- [ ] Close residual max-slew (pipeline the argmin compare to cut mux fanout) → 0-violation sign-off
- [ ] Replay testbench from real Qwen2 eviction traces
- [ ] Integration: tier-signal handshake with the KV Cache Engine
- [ ] Paper section with hardware results

## References

- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, NeurIPS 2023.
- LonghornSilicon ACU sparsity study (`adaptive-precision-attention/docs/findings/sparsity-controller-finding.md`) — post-softmax attention mass predicts token importance (r≈0.99); pre-softmax proxies do not.
