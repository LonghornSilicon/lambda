# TIU ↔ KV Cache Engine — Tier-Signal Handshake

How the Token Importance Unit (block 3) tells the KV Cache Engine (block 2) how to
treat each cached token: **keep**, **demote**, or **evict**. Verified end-to-end with
the ACU precision controller (block 1 / APA) in the loop — see below.

## Signals (on `token_importance_unit`)

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `tier_threshold` | in | SCORE_WIDTH | Accumulated-mass boundary between keep and demote (programmable). |
| `tier_keep` | out | N_SLOTS | Per-slot, combinational: `tier_keep[k] = valid[k] && (score[k] >= tier_threshold)`. |
| `evict_slot` / `evict_valid` | out | SLOT_WIDTH / 1 | The heavy-hitter-oracle victim to drop (from the serialized argmin). |

`tier_keep` is emitted as **N parallel comparators** (one per slot), not a muxed read,
so it adds no fanout to the argmin datapath — this is what kept the Sky130 sign-off at
0 violations after the port was added.

## Protocol

Per cached token `t` occupying slot `s`:

1. **Evict** — when the cache is full and a new token arrives, the TIU raises
   `evict_valid` with `evict_slot` = the minimum-mass valid slot. The KVCE **drops**
   that slot's K and V and frees it. (This is the per-token lever that applies to
   **both** K and V.)
2. **Keep / demote** — for a surviving token, KVCE reads `tier_keep[s]` when it
   (re)compresses that token's **value**:
   - `tier_keep[s] = 1` (heavy hitter) → store the **value at CQ-8** (per-token INT8).
   - `tier_keep[s] = 0` (demote)       → store the **value at CQ-4** (per-token INT4).

> **RETIRED under CQ-3-rot (branch, 2026-07-20).** The WHT-rotated INT3 value tier
> (`kv-cache-engine` `docs/wht_value_rotation.md`; the Walsh–Hadamard value rotation itself
> originates in TurboQuant+ / [`themoddedcube/turboquant-plus`](https://github.com/themoddedcube/turboquant-plus),
> by Chaithu Talasila — the flat-INT3 CQ-3-rot application is the joint idea of Abhiram Bandi + Chaithu Talasila)
> makes **all** values a flat, uniform 3 bits — better than the CQ-8/CQ-4 ladder at less
> memory and no calibration. There is no per-token value bit-width left to select, so the
> **value-precision role of `tier_keep` is retired**: the TIU keeps only its **evict-or-keep**
> lever (item 1), which applies to K and V alike. The `tier_keep` comparators may be left
> dormant or removed; the eviction datapath is unchanged. This section describes the
> pre-CQ-3-rot design and is kept for provenance.

## Why the tier drives VALUES, not KEYS

ChannelQuant compresses **keys per-channel** (one scale per channel dim, shared across a
token group) — that is what protects GQA's few high-magnitude key channels. There is no
clean way to give individual key *tokens* different bit-widths without falling back to
per-token key scaling, which collapses GQA accuracy (~−0.10; measured −0.17 in the full
stack). **Values** are already per-token quantized, so a per-token precision tier is
natural there. Therefore:

- The TIU's per-token lever for **keys** is **evict-or-keep only** (binary).
- The TIU's **demote** tier is a **value-path** precision selector.
- The whole key cache shares one ChannelQuant key tier (set globally, not per token).

(See `docs/findings/all-three-blocks-integration.md` and the
`channelquant-tiu-compatibility` note.)

## Verified with APA (all three blocks)

`analysis/full_stack_integration.py` composes TIU (evict + threshold keep/demote) →
KVCE (per-token value tier + uniform per-channel keys) → APA (INT8/FP16-routed S·V) in
one Qwen2 attention. Measured (Qwen2-0.5B, HellaSwag n=1000, 25% KV budget):

| config | acc_norm | Δ vs FP16 | APA INT8 |
|---|---|---|---|
| fp16 | 0.489 | — | — |
| APA only | 0.490 | +0.001 | 99.99% |
| **ALL 3 + tier(handshake)** | **0.458** | **−0.031** | **99.99%** |

The `tier(handshake)` row uses **exactly the RTL `tier_keep` semantics** (2-tier
keep→CQ-8 / demote→CQ-4 by an accumulated-mass threshold). The full stack — TIU
eviction + this tier + KVCE + APA's ~all-INT8 S·V — holds within the ~3% combined
envelope of FP16, with **APA fully active (99.99% INT8)**. The handshake composes with
APA with no interaction: APA operates on the attention scores (codec/tier-agnostic),
so the TIU tier and KVCE codec are transparent to it.

## RTL ↔ model correspondence

The RTL `tier_keep` (a single accumulated-mass threshold → keep/demote) is the hardware
realization of the model's importance→precision map. The model's continuous
rank-fraction mapping is a design-space generalization; the shipped silicon uses the
single programmable `tier_threshold`, which the software layer sets from a calibration
pass (e.g., the mean or a percentile of accumulated mass).
