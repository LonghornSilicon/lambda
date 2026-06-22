# ChannelQuant → KVCE Hardware Contract

**Status:** v0.1 (algorithm-side, pre-reference-model) · **Date:** 2026-06-22
**Audience:** the KVCE silicon block (a block of the Longhorn chip) that
implements ChannelQuant. **This document is a contract, not an implementation.**
It specifies *exactly what the hardware must reproduce* so that KVCE RTL passes
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

## 0. Scope of what KVCE must reproduce

KVCE compresses and decompresses a transformer KV cache for **KV heads only**
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
- **`EPS = 2^-14`** floors the scale so an all-zero group does not divide by zero;
  it matches the reference `clamp_min`. (The reference c17 used `1e-8`; the
  contract pins `2^-14` so the float math is representable in fp16 — the reference
  model will be updated to this value and the golden vectors regenerated. Until
  then, treat the exact EPS as **TBD-pinned-by-reference**.)

### Scale numeric format
- Scales are stored as **IEEE-754 fp16** (binary16). Dequant `x_hat = q * s` is an
  fp16 (or wider) multiply; HW may carry the product in fp32 and round once.
- The *quantize* step computes `x / s`. HW may implement this as a multiply by a
  precomputed reciprocal `r = 1/s`; if so, `r` must be computed such that
  `round_half_to_even(x * r)` equals `round_half_to_even(x / s)` for all golden
  vectors. **The golden vectors are the arbiter** — if a reciprocal approximation
  breaks parity, it is non-conformant.

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
  default chosen in Phase 2 (group-size Pareto). Parameterize.
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
buffer). KVCE's datapath is therefore K/V-asymmetric: buffered per-channel keys,
streaming per-token values.

---

## 4. Outlier-channel lane (CQ-4+ only)

The top-k key channels (by magnitude) are held in **FP16** instead of INT4.

### 4.1 Static mask — calibrated offline, shipped as ROM
- The outlier channel indices are **a property of the trained weights**, stable
  across inputs (validated, c19: top-2 stability 0.958/0.986/0.984 on Qwen2
  {0.5B/1.5B/7B}; layer-0 perfectly pinned). **No runtime top-k, no argsort in
  silicon.**
- KVCE ships a **static per-(layer, KV-head) outlier mask**:
  - **k** = number of outlier channels (config; default **k=2** at D=64). k scales
    with D — parameterize.
  - Format: **k channel indices** per (layer, head), each `ceil(log2(D))` bits,
    stored in a **ROM**. (Equivalently a D-wide bitmask; the reference emits both
    the index list and the bitmask in the golden vectors — KVCE may use either,
    but the *selected channel set* must match exactly.)
- The calibrator (`analysis/outlier_calibration.py`, Phase 2) produces this ROM
  content per model/layer/head and commits it as a data artifact.

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
  the canonical byte stream; KVCE must match it byte-for-byte.**

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

KVCE's actual packed sizes must reconcile with this table (the per-channel scale
overhead depends on G; the outlier overhead depends on k and D).

---

## 7. Config / INFO registers KVCE should expose

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

KVCE conformance = **bit-exact** Python↔C++↔SV on the golden vectors in
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
- Final **EPS** value (§1) — pin to the reference, regenerate vectors.
- Reciprocal-vs-divide equivalence (§1) — confirmed only by passing golden vectors.
- CQ-8 keys per-token vs per-channel (§5) — chosen per-token for the simple floor;
  revisit only if a CQ-8-per-channel tier is ever requested.
- Silicon results (area/Fmax vs TurboQuant+) are produced by the **KVCE block**,
  not here; the method paper notes them as forthcoming.
