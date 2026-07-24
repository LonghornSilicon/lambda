# ChannelQuant → KVE Hardware Contract

**Status:** v0.2 (P2 pins locked) · **Date:** 2026-07-01 · (v0.1: 2026-06-22)
> **v0.2 (2026-07-01) — four P2 parameters pinned** for the KVE streaming
> datapath, all doc-level (no change to packing, dequant, or golden vectors —
> the vectors are byte-identical since `08d5287`, so 3-way parity stays valid):
> **(1)** group size **G=128 for all D** (§3.1); **(2)** outlier ROM format +
> location + **k=2**, lane **optional at D=128** per the C16 headline finding
> (§4.1); **(3)** decompress read bus **fp32** (§1); **(4)** **EPS=2⁻¹⁴ final**,
> no regeneration (§1). See the resolved Open-items list at the foot.
**Audience:** the KVE silicon block (a block of the Longhorn chip) that
implements ChannelQuant. **This document is a contract, not an implementation.**
It specifies *exactly what the hardware must reproduce* so that KVE RTL passes
3-way Python↔C++↔SV bit-exact parity against the golden vectors emitted by
`reference/` (see `reference/testvectors/`).

It supersedes spec §5 ("Hardware delta"), which described a now-out-of-scope RTL
plan. Where this contract and §5 disagree, this contract wins. The algorithm
itself (§3), evidence (§2), compression math (§4), and gates (§7) in
`REVAMP_SPEC.md` remain binding.

> **Authority of the reference model.** Where prose here and the committed
> reference model disagree, the **reference model and its golden vectors are the
> ground truth**. This document explains intent; the vectors define correctness.

---

## 0. Scope of what KVE must reproduce

KVE compresses and decompresses a transformer KV cache for **KV heads only**
(GQA), per layer. For each KV head: keys `K[t, c]` and values `V[t, c]`, token
index `t`, channel index `c ∈ [0, D)`, `D ∈ {64, 128}` (parameterize; do not
hardcode 64). All quantization is **uniform signed integer** (no Lloyd-Max, no
FP4, no rotation, no JL, no routing).

Three tiers, selected by a mode register:

| Tier | Keys | Values | Outlier lane |
|---|---|---|---|
| **CQ-8** | INT8 per-token | INT8 per-token | none |
| **CQ-4** | INT4 per-channel (grouped, G) | INT4 per-token | none |
| **CQ-4+** | INT4 per-channel (grouped, G) | INT4 per-token | top-k key channels in FP16 |
| **CQ-3-rot** | INT4 per-channel (grouped, G) | **WHT-rotated INT3 per-token** (`pack_int3`, 3.0 b/val) | top-k key channels in FP16 |

CQ-3-rot (branch `wht-turboquant-values`) adds a fixed Walsh-Hadamard rotation to the
per-token **value** row before the standard amax/quant rule (§2), taking values to 3 bits;
keys and the outlier lane are unchanged. The rotation is add/sub only and is undone once on
the P·V output (Path B). Full spec + bit-exact evidence: `docs/wht_value_rotation.md`.

---

## 1. Quantization rule (exact, bit-deterministic)

Signed integer, symmetric, with a per-axis floating scale.

```
qmax(b) = 2^(b-1) - 1            # INT4 → 7 ;  INT8 → 127
qmin(b) = -2^(b-1)              # INT4 → -8 ;  INT8 → -128
s       = max(amax / qmax(b), EPS)        # EPS = 2^-14 (smallest fp16 normal-ish floor)
q       = clamp( round_half_to_even(x / s), qmin(b), qmax(b) )
x_hat   = q * s                            # dequant
```

- **`amax`** is the max-absolute over the scale axis (defined per path in §2/§3).
- **Rounding is round-half-to-even** (banker's rounding) — matches `numpy.rint` /
  `torch.round`. The HW rounder MUST use round-half-to-even, not round-half-away.
  Ties are rare in practice but must match for bit-exact parity.
- **Clamp** is applied *after* rounding, to the inclusive integer range
  `[qmin, qmax]`. Note the asymmetry: INT4 range is **[-8, 7]** (−8 is legal).
- **`EPS = 2^-14`** floors the scale so an all-zero group does not divide by zero.
  **FINAL** — the reference model pins `2^-14` and the committed golden vectors
  were generated with it (`reference/testvectors/manifest.json: eps =
  6.103515625e-05`). The earlier `1e-8` (c17) is retired; no further EPS change or
  vector regeneration is coming.

### Scale numeric format
- Scales are stored as **IEEE-754 fp16** (binary16). Dequant `x_hat = q * s` is an
  fp16 (or wider) multiply; HW may carry the product in fp32 and round once.
- The *quantize* step computes `x / s`. HW may implement this as a multiply by a
  precomputed reciprocal `r = 1/s`; if so, `r` must be computed such that
  `round_half_to_even(x * r)` equals `round_half_to_even(x / s)` for all golden
  vectors. **The golden vectors are the arbiter** — if a reciprocal approximation
  breaks parity, it is non-conformant.

### Decompress output (read-side bus) — PINNED
- The decompressed cache `x_hat = q * s` is emitted as **IEEE-754 fp32**. The
  golden `expected_{K,V}_hat` are `float32` and 3-way parity is defined in fp32,
  which the current C++/SV already match bit-exactly. **The codec read bus is
  fp32.** If a downstream attention core wants fp16, apply a single narrowing
  round **outside** the codec's parity boundary — that cast is not part of the
  golden vectors and must not alter the packed stream or the fp32 `_hat` gold.

---

## 2. Values — per-token INT4/INT8 (streaming, no buffer)

```
for each value token t:
    s_v[t] = max( amax_d |V[t, :]| / qmax(b), EPS )     # one scale per token, over D dims
    v_q[t, d] = clamp( round_half_to_even(V[t, d] / s_v[t]), qmin(b), qmax(b) )
```

- Scale axis = **the D feature dims of one token**. Known the instant the token
  arrives → quantize immediately, **no residual buffer**.
- `b = 8` for CQ-8, `b = 4` for CQ-4 / CQ-4+.
- Stored per token: the payload `v_q[t, :]` (D × b bits) + one fp16 scale `s_v[t]`.

---

## 3. Keys — per-channel INT4 (grouped) + residual buffer

Keys are scaled **per channel over a group of G tokens** — this is what localizes
the outlier channels. Because a channel's group-max is not known until the group
is full, the in-flight group is buffered in FP16.

### 3.1 Group size G and residual-buffer semantics
- **G** (tokens per group) is a config register. Validated set {32, 64, 128, 256};
  **default G = 128** — pinned by the Phase-2 group-size Pareto (`analysis/
  c22_group_size_sweep.py`, `analysis/fig_group_size_pareto.png`): acc_norm is
  statistically flat in G (n=250, gaps ≪ Wilson CI) on Qwen2-{0.5B,1.5B,7B}, so
  G=128 is the effective-bits floor that still streams cheaply (~4.13 combined
  bits/value for CQ-4; G=256/full save <0.06 b but need a larger residual buffer).
  Still parameterize — but ship G=128.
- **G is independent of head dim D — ship G=128 for both D=64 and D=128.** The
  D=64 golden vectors use G=64 only to exercise a second grouping and a
  partial-group flush (test coverage), **not** as a shipped D=64 default.
  Confirmed at headline n=1000: acceptance passes at G=128 on Qwen2-1.5B & 7B
  (`analysis/c23_{q15,q7}_headline.json`).
- The **residual buffer** holds the current, not-yet-full group of the most
  recent keys in **FP16**: `≤ G tokens × D × 16 bits` per KV head. Fixed size.
- **Flush (quantize a block) when EITHER:**
  1. the buffer reaches **G tokens** (full group), or
  2. the sequence ends / cache is finalized with a **partial** group of
     `g < G` tokens still in the buffer.
- On flush, for that block of `g ∈ [1, G]` tokens:
  ```
  for each channel c:
      s_k[c] = max( amax over the g tokens of |K[t, c]| / qmax(4), EPS )   # per-channel scale
      for each token t in block:
          k_q[t, c] = clamp( round_half_to_even(K[t, c] / s_k[c]), -8, 7 )
  ```
  i.e. **amax is taken over the tokens actually present in the block** (g, not G).
  Partial final groups are legal and must be handled identically with g<G.
- Each flushed key block stores: payload `k_q` (g × D × 4 bits) + **D fp16
  per-channel scales** `s_k[0..D-1]` (one scale bank per block).
- Keys are read back by dequant `K_hat[t,c] = k_q[t,c] * s_k[c]`. Tokens already
  flushed are immutable; only the in-flight buffer is FP16.

### 3.2 Why per-channel keys / per-token values are asymmetric
Keys have fixed large-magnitude **channels** (a weight property → per-channel
scale contains them); values do not (per-token suffices, and it streams with no
buffer). KVE's datapath is therefore K/V-asymmetric: buffered per-channel keys,
streaming per-token values.

---

## 4. Outlier-channel lane (CQ-4+ only)

The top-k key channels (by magnitude) are held in **FP16** instead of INT4.

### 4.1 Static mask — calibrated offline, shipped as ROM
- The outlier channel indices are **a property of the trained weights**, stable
  across inputs (validated, c19: top-2 stability 0.958/0.986/0.984 on Qwen2
  {0.5B/1.5B/7B}; layer-0 perfectly pinned). **No runtime top-k, no argsort in
  silicon.**
- KVE ships a **static per-(layer, KV-head) outlier mask**:
  - **k** = number of outlier channels (config). **Default k=2 at both D=64 and
    D=128** (as calibrated for every shipped model). Parameterize k.
  - Format: **k channel indices** per (layer, head), each `ceil(log2(D))` bits,
    stored in a **ROM**. (Equivalently a D-wide bitmask; the reference emits both
    the index list and the bitmask — KVE may use either, but the *selected
    channel set* must match exactly.)
- **Committed ROM artifact (P2 loads this, not the mask embedded in the vectors):**
  `reference/masks/<tag>_k<k>.npz` (+ a `.json` summary), one per model. The `.npz`
  carries:
  - `outlier_idx` — `int64 [L, n_kv, k]`, the k channel indices per (layer, KV head).
  - `outlier_bitmask` — `uint8 [L, n_kv, D]`, the same selection as a D-wide mask.
  - scalars `n_layers, n_kv_heads, head_dim, topk`.
  Committed: Qwen2 `q05_k2` (D=64), `q15_k2`/`q7_k2` (D=128); plus `mistral_k2`
  (D=128, the non-Qwen generalization ROM). The mask is model-weight-specific:
  each deployed model needs its own calibration run.
- The calibrator (`analysis/outlier_calibration.py`, Phase 2) produces this ROM
  content per model/layer/head deterministically (n_calib=128, dataset order).
- **When to enable the lane (Phase-3 finding, C16).** At **D=64** the "+" lane
  helps (0.5B: 0.428 vs CQ-4 0.408, n=250 screening). At **D=128** the paired
  headline run (n=1000, `analysis/c23_*_headline.json`) finds **no significant
  CQ-4+ vs CQ-4 difference** (1.5B Δ=+0.012 CI[−0.001,0.025] p=0.09; 7B Δ=−0.002
  CI[−0.017,0.013] p=0.90 — both span 0). So **the D=128 default is CQ-4** (lane
  disabled → drop the ROM+sidecar, an area win); build the outlier lane as an
  **optional/bypassable** datapath needed for D=64 and any explicit CQ-4+ config.

### 4.2 Datapath
- **Outlier channels** (in the mask): stored in **FP16** in a sidecar lane, NOT
  quantized. Decompress = identity (the FP16 value).
- **Non-outlier channels**: per-channel INT4 exactly as §3, but their per-channel
  scales are computed over the **non-outlier channels only** (the outlier columns
  are excluded from the INT4 path entirely — they do not get an INT4 scale).
- Decompress reassembles the full D-wide key: outlier columns from the FP16
  sidecar at their masked indices, the rest from `k_q * s_k`.

---

## 5. Bit / packing layout per tier

Per KV head, per layer. `D` = head dim, `G` = group size, `g` = tokens in a key
block, `T` = tokens in a value run.

### CQ-8
- Values: `v_q` int8 `[T, D]` (8b/elem) + fp16 `s_v[T]`.
- Keys: int8 per-token, same shape/format as values (8b/elem + fp16 per-token
  scale). (CQ-8 keys are per-token, NOT per-channel — it is the simple lossless
  floor; no residual buffer, no scale bank.)

### CQ-4
- Values: `v_q` int4 `[T, D]` (4b/elem, two per byte) + fp16 `s_v[T]`.
- Keys: per key block — `k_q` int4 `[g, D]` (4b/elem) + fp16 `s_k[D]` (D
  per-channel scales).

### CQ-4+
- Values: identical to CQ-4.
- Keys: as CQ-4 but with **k channels removed from the INT4 payload** and stored
  in an FP16 sidecar:
  - INT4 payload: `k_q` int4 `[g, D−k]` over non-outlier channels + fp16
    `s_k[D−k]`.
  - FP16 sidecar: `[g, k]` fp16 values for the outlier channels.
  - Outlier mask (from ROM): k indices (or D-bit bitmask) — not stored per block,
    it is static per (layer, head).

### Nibble packing
- INT4 nibble order within a byte: **little-endian element order** — element `2i`
  in the low nibble (bits [3:0]), element `2i+1` in the high nibble (bits [7:4]).
  Signed values stored two's-complement in the nibble.
- If a row's element count is odd, the final nibble's high half is zero-padded.
- INT8 elements are one byte each, two's-complement. **The golden vectors define
  the canonical byte stream; KVE must match it byte-for-byte.**

---

## 6. Effective-bits accounting (must match RTL packer, Phase-2 verification)

D=64 reference (spec §4):

| Path | payload b/val | scale overhead | eff. bits/val | ratio vs fp16 |
|---|---|---|---|---|
| Value (per-token) | 4 | 16/64 = 0.25 | 4.25 | 3.76× |
| Key (per-ch, G=128) | 4 | 16/128 = 0.125 | 4.13 | 3.88× |
| Key CQ-4+ (top-2 FP16) | 4 | +0.125 + (2/64)(16−4) = 0.375 | 4.50 | 3.55× |
| Combined CQ-4 | — | — | ~4.2 | ~3.8× |
| Combined CQ-4+ | — | — | ~4.4 | ~3.6× |

KVE's actual packed sizes must reconcile with this table (the per-channel scale
overhead depends on G; the outlier overhead depends on k and D).

---

## 7. Config / INFO registers KVE should expose

(Carried from the predecessor CSR scaffolding, minus QJL/rotation registers.)

| Register | Meaning |
|---|---|
| `MODE` | tier select: CQ-8 / CQ-4 / CQ-4+ |
| `D` | head dim (64 / 128) |
| `G` | key group size (32 / 64 / 128 / 256) |
| `OUTLIER_K` | k outlier channels (0 disables the lane → CQ-4) |
| `SCALE_BANK_DEPTH` | = D (per-channel key scale bank) |
| `RESID_DEPTH` | = G (residual buffer token capacity) |
| `OUTLIER_ROM_BASE` | base address of the per-(layer,head) outlier-index ROM |

---

## 8. Parity acceptance (the handoff gate)

KVE conformance = **bit-exact** Python↔C++↔SV on the golden vectors in
`reference/testvectors/`, for **both compress and decompress**, for **all three
tiers**, including:
- a full group (g=G) and at least one **partial group** (g<G) for keys;
- D ∈ {64, 128};
- CQ-4+ with k=2 and the static mask applied;
- the exact packed byte stream (§5), not just the dequantized tensors.

Anything that is bit-exact on dequantized values but differs in the packed byte
layout is **non-conformant** — the byte stream is part of the contract.

---

## Open items (pinned as the reference model lands)
- ~~Final **EPS** value~~ — **RESOLVED: `2^-14`, final** (§1); vectors already
  generated with it (`manifest.eps = 6.103515625e-05`). No regeneration coming.
- ~~Decompress bus dtype~~ — **RESOLVED: fp32** (§1, read-side bus); C++/SV parity
  already matches the fp32 `_hat` gold.
- ~~Final group size G~~ — **RESOLVED: G=128, all D** (§3.1); Phase-2 knee,
  confirmed at headline n=1000.
- ~~Outlier k / ROM~~ — **RESOLVED: k=2 all D; lane optional at D=128** (§4.1, C16);
  ROM at `reference/masks/<tag>_k2.npz`.
- Reciprocal-vs-divide equivalence (§1) — confirmed only by passing golden vectors.
- CQ-8 keys per-token vs per-channel (§5) — chosen per-token for the simple floor;
  revisit only if a CQ-8-per-channel tier is ever requested.
- Silicon results (area/Fmax vs TurboQuant+) are produced by the **KVE block**,
  not here; the method paper notes them as forthcoming.
