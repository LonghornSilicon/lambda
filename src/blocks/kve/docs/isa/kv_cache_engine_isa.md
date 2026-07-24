# KV Cache Engine — Interface Specification

**Status**: Codec integrated; RTL signed off (all CI gates green, Sky130). Stable
for the KV cache engine block only; will be unified with the rest of the
LonghornSilicon ISA when the Token Importance Unit and Memory Hierarchy Controller
blocks land.

**Version**: `kv-isa-0.2` (ChannelQuant codec) — 2026-07-03.

**Scope**: The externally-visible interface to the `kv_cache_engine` block as it
appears on the chip, an FPGA prototype (ZCU102/104), or the bit-accurate software
model. **v0.2** replaces the retired TurboQuant+ codec (PolarQuant + QJL + WHT
rotation) with **ChannelQuant** — per-channel INT4 keys, per-token INT4 values,
and static outlier-channel isolation — per the algorithm contract frozen in the
sibling `channelquant` repository. (The full typeset spec is
[`kv_cache_engine_isa.pdf`](kv_cache_engine_isa.pdf) / `.tex`.)

---

## 1. Block overview

ChannelQuant quantizes on the axis that dominates the KV error budget for GQA
models — a few fixed high-magnitude **key channels**:

- **Key path (per-channel INT4)** — keys are scaled *per channel* over a group of
  `G` tokens, then quantized to INT4; the top-`k` worst channels (tier CQ-4+) are
  held FP16 from a static calibrated ROM mask.
- **Value path (per-token INT4)** — values are scaled per token → INT4 (INT8 for
  the CQ-8 tier).

**Tiers** (`INFO_TIER`): `0` CQ-8 (per-token INT8 K+V) · `1` CQ-4 · `2` CQ-4+.

Decompression is per channel (`code · scale`, in fp32); outlier channels replay
their stored FP16. Effective ~4 bits/value, ~3.8× vs FP16, near-lossless
(HellaSwag acc_norm within ~0.5 pt of FP16 at the best value tier per model — CQ-4 on
Qwen2-0.5B, CQ-4+ on 1.5B; CQ-4 is the default at every head dim, the "+" outlier lane
only helps at D=128 — see the block README accuracy table and `../../DECISIONS.md`).

Bit-exact reference: `sw/reference_model/channelquant_ref.{hpp,cpp}` and the frozen
Python reference `../channelquant/reference/channelquant_ref.py` (3-way parity).

---

## 2. Register map (AXI-Lite, v0.2)

256-byte window, 32-bit word-aligned registers.

| Offset | Name | Access | Purpose |
|---|---|---|---|
| `0x00` | `CTRL` | RW | bit[0] soft_reset, bit[1] enable |
| `0x04` | `STATUS` | R | bit[0] idle, bit[3] sram_full |
| `0x08` | `INFO_DIM` | R | head dim `D` |
| `0x0C` | `INFO_TIER` | R | 0=CQ-8, 1=CQ-4, 2=CQ-4+ |
| `0x10` | `INFO_GROUP` | R | key group size `G` |
| `0x14` | `INFO_SRAM_DEPTH` | R | SRAM entries |
| `0x18` | `INFO_CR_K` | R | key compression ratio (8.8 fixed-point) |
| `0x1C` | `INFO_CR_V` | R | value compression ratio (8.8 fixed-point) |
| `0x20` | `INFO_VERSION` | R | `0x00020000` (v0.2) |
| `0x24` | `OCCUPANCY` | R | valid SRAM entries |
| `0x28` | `WRITE_ADDR` | RW | write / key-group-base address |
| `0x2C` | `READ_ADDR` | RW | read address (write launches a decompress) |
| `0x30` | `KV_SELECT` | RW | 0=key, 1=value |
| `0x34` | `IRQ_MASK` | RW | interrupt enable mask |
| `0x38` | `IRQ_STATUS` | R/W1C | interrupt pending |
| `0x3C` | `INFO_OUTLIER_K` | R | top-k FP16 outlier key channels (CQ-4+) |
| `0x40` | `INFO_SCALE_DEPTH` | R | per-channel scale-bank depth (= D) |
| `0x44` | `INFO_RESID_DEPTH` | R | residual-buffer depth (= G) |

Reserved offsets read `0xDEADBEEF`; (syn) values are fixed at synthesis time.

---

## 3. Streaming interfaces

- **`s_axis_kv` (write)** — `tdata` = one FP16 coordinate (`COORD_WIDTH`=16);
  `VECTOR_DIM` beats per token; `tlast` on the last beat; `tuser` 0=key/1=value.
  Value tokens (and CQ-8 keys) compress per token. **CQ-4/CQ-4+ keys are grouped**:
  program `WRITE_ADDR` to the group base, stream `G` key tokens back-to-back; the
  engine buffers the group, freezes the per-channel scales, and emits `G` per-token
  records to `base, base+1, …`. (Partial-group flush, `g<G`, is planned; the
  datapath already supports it.)
- **`m_axis_kv` (read)** — `tdata` = one **fp32** decompressed coordinate
  (`OUT_WIDTH`=32); writing `READ_ADDR` launches a decompress that streams
  `VECTOR_DIM` fp32 beats.
- **Eviction** — `evict_needed` / `evict_addr` to the Memory Hierarchy Controller
  when SRAM is full.

---

## 4. ChannelQuant algorithm

Round-half-to-even; INT4 clamp `[-8,7]`, INT8 `[-128,127]`; FP16 scales,
`EPS=2^-14`. (Contract frozen in `../channelquant`, v0.2.)

- **Key (per-channel INT4)**: buffer `G` tokens → per-channel amax `a_c` → FP16
  scale `s_c = max(a_c/qmax, EPS)` (qmax=7) → `round(k/s_c)` to INT4. CQ-4+: the
  top-k ROM-mask channels are stored FP16 instead.
- **Value (per-token INT4)**: per-token amax → FP16 scale → INT4 (INT8 for CQ-8).
- **Unified per-channel SRAM record** `{tag, D×FP16 field, D×INT4 code}`: keep
  channel → `{group scale, INT4}`; **outlier channel → `{raw FP16, code +1}`** so
  decompress `code·field` widens the FP16 exactly — no separate sidecar, no
  read-side mask.
- **Serialized datapath**: one shared scale/quant/dequant unit walked across the D
  channels (a single parallel divide cone stalls P&R) — bit-exact with the oracle,
  synthesizes / passes formal / place-and-routes on Sky130.

Effective bits/value: key CQ-4 `4 + 16/G`; CQ-4+ `+ (k/D)(16−4)`; value `4 + 16/D`.

---

## 5. Synthesis-time configuration

| Parameter | Default | Notes |
|---|---|---|
| `VECTOR_DIM` | 64 | head dim D |
| `TIER` | 1 | 0=CQ-8, 1=CQ-4, 2=CQ-4+ |
| `KEY_GROUP` | 128 | tokens per key group (G) |
| `OUTLIER_K` | 0 | FP16 outlier channels (CQ-4+ → 2) |
| `SCALE_WIDTH` | 16 | FP16 scale width |
| `COORD_WIDTH` | 16 | FP16 input width |
| `OUT_WIDTH` | 32 | fp32 output width |
| `SRAM_DEPTH` | 16 | behavioral reg array (Sky130); real macro at 16FFC |

The synth/formal/OpenLane CI gates use a small flop-based gate proxy of the default
params; shipped D/G/depth are set per instantiation.

---

## 6. Change log

- `kv-isa-0.2` (2026-07-03): TurboQuant+ → ChannelQuant. New `INFO_TIER`/`GROUP`/
  `OUTLIER_K`/`SCALE_DEPTH`/`RESID_DEPTH`; `INFO_PQ_BITS`/`INFO_QJL_BITS` removed;
  fp32 read bus. RTL integrated + signed off (all CI gates green, Sky130).
- `kv-isa-0.1` (2026-05-14): first public draft (TurboQuant+, retired).
