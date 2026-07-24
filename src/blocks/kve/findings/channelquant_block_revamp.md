# KVE block revamp — implement ChannelQuant (retire the TurboQuant+ datapath)

**Status:** design / pre-implementation · **Date:** 2026-06-22
**Scope:** the KVE *silicon block* (this repo) — datapath, RTL, CSR/ISA, ACU
integration, synth. NOT the algorithm (that's the `channelquant` repo).
**Algorithm source of truth:** `channelquant/docs/HW_CONTRACT.md` +
`channelquant/REVAMP_SPEC.md`. **Verification golden:** vendored hermetically at
[`../rtl/tb/testvectors/channelquant/`](../rtl/tb/testvectors/channelquant/README.md)
(channelquant commit `08d5287`; upstream `channelquant/reference/testvectors/`).

---

## 0. What this is

The KVE block stays — it is block 2 of 4 in the Longhorn chip (streaming K/V
compress-on-write / decompress-on-read, between the ACU and the memory
hierarchy). What changes is the **codec it implements**: the TurboQuant+ vector
codec (PolarQuant + QJL + Walsh–Hadamard rotation) is retired and replaced by
**ChannelQuant** (per-channel-key INT4 / per-token-value INT4 / static
outlier-channel isolation). The block's role, ports, SRAM, and ACU handshake are
unchanged; the internal datapath is rebuilt.

**Why:** TurboQuant+ hit ~3.5× but with a −0.10 HellaSwag acc_norm collapse on
GQA (the codec delocalized per-channel key-outlier error so no protection caught
it). ChannelQuant reaches ~3.6–3.8× *near-lossless* by scaling keys per-channel.
Full evidence + algorithm: `channelquant/`. This doc is the silicon plan to
realize it.

---

## 1. Hard dependency — **LANDED 2026-06-22** (verification unblocked)

The RTL **design** below is startable now. RTL **verification** (3-way parity)
was blocked on the algorithm lane's golden vectors + contract — **both have now
landed, verified**, vendored at
[`rtl/tb/testvectors/channelquant/`](../rtl/tb/testvectors/channelquant/README.md)
(channelquant commit `08d5287`):

- **ChannelQuant Phase 1 golden vectors** — ✅ 9 vectors `(input K/V, expected
  packed payload, expected decompressed K/V)` for CQ-8 / CQ-4 / CQ-4+, covering a
  full key group (g=G), partial groups (g<G), and D ∈ {64,128}; CQ-4+ at k=2 with
  the static mask. `.npz` (reference truth) + `$readmemh`-loadable `hex/`. The
  reference reproduces c17 bit-exactly and `torch`==`numpy` per tier (details in
  the vendored README). **P3 (3-way parity) is now startable.**
- **`HW_CONTRACT.md`** — ✅ vendored. Pins the exact quant rule (round-half-to-
  even, clamp INT4 [−8,7]/INT8 [−128,127], `EPS=2^-14`), fp16 scale format,
  packing layout (§5), group-flush semantics (§3), outlier-mask format (§4).
  **Do not guess these** — implement against the contract; if it is silent (e.g.
  the decompress-bus product format, flagged open), raise it to the channelquant
  lane, don't invent.

---

## 2. RTL teardown — current blocks and their fate

Current `rtl/` (TurboQuant+ datapath):

| Block | Fate | Reason |
|---|---|---|
| `rotation_unit.sv` (WHT butterfly) | **DELETE** | no rotation in ChannelQuant |
| `qjl_unit.sv` (1-bit JL sketch) | **DELETE** | no residual sketch |
| `norm_unit.sv` (L2 norm) | **REPURPOSE → `amax_unit.sv`** | need per-axis amax (channel-max for K, token-max for V), not L2 norm |
| `quantizer.sv` (3-bit Lloyd-Max) | **REPLACE** | uniform signed INT4/INT8 round+clamp — smaller, no centroid ROM |
| `packer.sv` | **KEEP, re-lane** | repack to 4/8-bit lanes + scale sidecar + outlier FP16 lane |
| `decompressor.sv` | **SIMPLIFY** | just `q * scale` (+ re-insert FP16 outlier channels); no inverse-WHT, no JL reconstruct |
| `sram_controller.sv` | **KEEP + extend** | add scale-bank storage + residual-group buffer management |
| `kv_cache_engine.sv` (top) | **REWIRE** | new datapath, new CSR fields |
| **NEW** `residual_buffer.sv` | add | FP16 hold for the in-flight (not-yet-full) key group |
| **NEW** `scale_bank.sv` | add | per-channel K scales (D entries) + per-token V scale FIFO |
| **NEW** outlier-mask ROM | add | static per-layer top-k key-channel indices |

Net: the codec gets **smaller and likely faster** — the O(n log n) WHT butterfly
and the JL projection matrix are gone; complexity moves from *arithmetic* to
*buffering/scheduling*. Expect area down and Fmax up vs the TurboQuant+ baseline;
confirm in §6.

---

## 3. The datapath (what to build)

Parameterize on `D` = head_dim (64 for Qwen2-0.5B, 128 for 1.5B/7B — do NOT
hardcode 64; the old block fixed `VECTOR_DIM=64`).

### 3.1 Value path — per-token INT4 (streaming, easy)
- On each value vector write: `amax_unit` computes amax over the D dims of *that
  token* → one scale; `quantizer` does `round(V/scale)` clamp [−8,7]; `packer`
  emits 4-bit lanes + the per-token scale. No buffering — scale is known the
  instant the token arrives. This is the simpler, higher-compression path.

### 3.2 Key path — per-channel INT4 (grouped, needs the residual buffer)
This is the hard part and the block's defining new mechanism. Per-channel scaling
needs the **column (channel) max over a group of tokens** — which you don't have
until the group fills. Implement the KIVI streaming answer:
- Incoming key vectors accumulate into `residual_buffer` (FP16), and per-channel
  running amax accumulates into `scale_bank` (D entries).
- When the group reaches **G tokens** (start G=128; the channelquant lane picks
  the final G from its Pareto — read it from `HW_CONTRACT.md`): freeze the D
  per-channel scales, quantize the whole G×D block `round(K/scale[c])` clamp
  [−8,7], `packer` emits it, residual buffer clears for the next group.
- The current in-flight (partial) group is served to the ACU from the **FP16
  residual buffer directly** on decompress — it is not yet quantized. Decompress
  logic must select: quantized-group path vs in-flight-FP16 path by token index.

### 3.3 Outlier isolation (CQ-4+ tier)
- A **static per-layer ROM** holds the top-k key-channel indices (k=2 default;
  calibrated offline by the channelquant lane — c19 confirmed they're
  input-stable, layer-0 perfectly so). At compress time, those k channels bypass
  the INT4 quantizer into an FP16 sidecar lane; the rest go per-channel INT4. At
  decompress, the sidecar overwrites those k channels. **No runtime argsort in
  silicon** — this is the whole reason the "+" tier is cheap.

### 3.4 Tiers (CSR-selected)
| Tier | K path | V path | ~ratio |
|---|---|---|---|
| CQ-8 | per-token INT8 | per-token INT8 | 2× (lossless floor) |
| CQ-4 | per-channel INT4 (grouped) | per-token INT4 | ~3.8× |
| CQ-4+ | CQ-4 + top-k FP16 outlier lane | per-token INT4 | ~3.6× (best acc) |

No adaptive per-token bit routing (research ruled it out — granularity, not
bit-allocation, is the lever).

---

## 4. CSR / ISA map changes

Drop TurboQuant+ registers, add ChannelQuant ones:

| Action | Register |
|---|---|
| REMOVE | `INFO_PQ_BITS`, `INFO_QJL_BITS` (no PolarQuant / QJL) |
| REPURPOSE | `INFO_CR_K` / `INFO_CR_V` keep meaning (compression ratio, 8.8 fixed-pt) |
| ADD | `INFO_GROUP_SIZE` (G, key group length) |
| ADD | `INFO_OUTLIER_K` (k channels in FP16; 0 ⇒ CQ-4, no "+") |
| ADD | `INFO_SCALE_FMT` (scale numeric format per HW_CONTRACT) |
| ADD | `CFG_TIER` (CTRL field: 0=CQ-8, 1=CQ-4, 2=CQ-4+) |
| ADD | `INFO_HEAD_DIM` (D, now parameterized — old block hardcoded 64) |
| BUMP | `INFO_VERSION` (ISA major bump — incompatible codec) |
| ADD | outlier-mask load interface (per-layer ROM image programming) |

---

## 5. ACU integration

The ACU handshake is unchanged in shape (KVE decompresses K/V → ACU computes
Q·Kᵀ → precision controller routes INT8/FP16 → MAC). Deltas to verify:
- **Decompress latency** now has two paths (quantized-group vs in-flight-FP16
  residual) — confirm the ACU's read timing tolerates the residual-buffer select.
- **Tier signaling**: `CFG_TIER` is set per-layer/per-deployment, not per-tile —
  simpler than TurboQuant+'s per-tile assumptions. Confirm with the `acu` block
  (`acu/precision_controller`) that no per-tile codec signaling is expected.
- Outlier FP16 sidecar adds a few channels of bandwidth on read — account for it
  in the memory-hierarchy bandwidth model (block 4).

---

## 6. Verification & synthesis plan (gate each)

1. **Unit tests** (now, vs Python reference behaviorally): amax_unit, quantizer,
   residual_buffer group-flush, scale_bank, outlier bypass, decompress select.
2. **3-way parity** (vectors landed; gated only on an SV simulator on PATH): Python ↔ C++ ↔ SV bit-exact,
   compress AND decompress, for CQ-8 / CQ-4 / CQ-4+. Reuse the existing
   testvector/parity harness in `rtl/tb/`.
3. **Real-data trace** (`tb_realdata.sv`): run a captured Qwen2 K/V trace, confirm
   reconstructed-tensor rMSE matches the reference within tolerance.
4. **Synthesis** (`openlane/`, Sky130 first @ 80 MHz target, then 16FFC @ 800 MHz):
   report area, Fmax, power; **compare head-to-head with the TurboQuant+ baseline**
   (expect smaller/faster — no WHT, no JL). This delta is a paper result.

**Acceptance:** bit-exact parity on all tiers + synth closes at target Fmax +
area ≤ TurboQuant+ baseline.

---

## 7. Risks

1. **Residual-group buffer cost.** G=128 × D × FP16 per KV head — size it, and the
   scale-bank, against the area budget. If too costly, smaller G (worse ratio) or
   CQ-8-only fallback. This is the main HW unknown (algorithm risk is already
   retired by c17/c19).
2. **Group-flush timing.** The quantize-the-whole-group event is bursty vs the
   streaming value path — make sure it doesn't stall writes or starve the ACU read
   port. May need double-buffering the residual group.
3. **Contract drift.** If the channelquant lane changes G, k, rounding, or packing
   after RTL starts, re-sync. Pin a `HW_CONTRACT.md` version in the RTL.

---

## 8. Phasing

- **P0 (now):** this spec + RTL teardown branch; delete rotation/qjl, stub the new
  blocks, rewire top + CSR. No dependency.
- **P1 (now):** implement value path (easy) + amax/quantizer/packer/decompress;
  unit-test vs Python reference.
- **P2 (now):** implement key path — residual_buffer + scale_bank + group flush +
  outlier ROM. Unit-test.
- **P3 (vectors landed; gated only on an SV simulator on PATH):** 3-way bit-exact parity, real-data
  trace.
- **P4:** synth + area/Fmax vs TurboQuant+ → silicon results for the joint paper.

Keep `NOTES.md` dated entries with provenance for every parity run and synth
result — these feed the paper's hardware-evaluation section.
