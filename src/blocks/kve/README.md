# KV Cache Engine (KVE)

**A streaming compress-on-write / decompress-on-read engine for transformer KV-cache
tensors** — block 2 of the LonghornSilicon (Lambda) decode-attention accelerator. It sits
between the ACU (attention compute unit) and the memory hierarchy, cutting off-chip
LPDDR5X KV-cache bandwidth ~3.8× (near-lossless) so longer context fits the same memory
budget. Product target: **TSMC 16nm (N16FFC)**; proven on open PDKs (Sky130 flagship, GF180
chipathon).

**Layout** (canonical block template): `sw/ rtl/ pdk/ docs/ research/` (+ `analysis/`,
`findings/`, `NOTES.md`, `DECISIONS.md`). Block root: `src/blocks/kve/`.

> **Branch model — RTL lives on `rev0`, not `main`.** `main` is a clean scaffold (docs,
> `pdk/` configs, Python golden reference, `results/`) with **no `.sv`/`.v`**. Contributors
> branch from `rev0` and PR into it; a lead blesses and merges to `main`. **To see/build the
> RTL: `git checkout rev0`.** Full model: [`docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §6a.

> **Building a compiler / integrating this block?** This block's interface spec is
> [`docs/isa/kv_cache_engine_isa.md`](docs/isa/kv_cache_engine_isa.md) (KV data format).
> The chip-level compiler guide and documentation standard live under the monorepo
> [`docs/`](../../../docs/).

---

## Codec: ChannelQuant (per-channel INT4 keys + per-token INT4 values)

The block stays; the codec it implements is **ChannelQuant** — the KIVI/KVQuant recipe.
The predecessor TurboQuant+ (PolarQuant + QJL + Walsh–Hadamard rotation) was **retired
2026-06-22**: it reached ~3.5× compression but collapsed HellaSwag `acc_norm` on GQA models
(−0.10, 0.316 vs 0.420 FP16 on Qwen2-0.5B). Root cause: KV quant error on GQA is dominated
by a few fixed high-magnitude **key channels**, and the rotation step delocalized that error
so no per-token protection caught it. The retired datapath is archived on branch
[`legacy/turboquant-plus`](../../../../tree/legacy/turboquant-plus).

The algorithm is prior art (KIVI ICML'24, KVQuant 2024); **the contribution of this block is
the streaming silicon implementation.**

### TL;DR

| | |
|---|---|
| **What** | Streaming compress/decompress engine for transformer KV-cache tensors |
| **Why** | Cuts off-chip LPDDR5X KV bandwidth ~3.8× (near-lossless), enabling longer context per memory budget |
| **How** | Per-channel INT4 keys (grouped, G=128) + per-token INT4 values + static top-k FP16 outlier-channel isolation (CQ-4+) |
| **K/V asymmetry** | K: per-channel scale over a token group (the GQA-critical axis); V: per-token scale |
| **Tiers** | CQ-8 (per-token INT8 K+V); CQ-4 (per-channel INT4 K / per-token INT4 V — the **default**); CQ-4+ (CQ-4 + k=2 FP16 outlier channels); **CQ-3-rot** (CQ-4+ keys + WHT-rotated per-token **INT3** values, flat 3.0 b/val, ~4.8×; [`docs/wht_value_rotation.md`](docs/wht_value_rotation.md)) |
| **Verified** | RTL bit-exact vs golden (`sim_kpath`/`sim_top`), 3-way Python↔C++↔SV parity |
| **Accuracy** | HellaSwag `acc_norm` within ~0.5 pt of FP16 at the best value tier per model |
| **Status** | **Sky130: signed off @10 MHz** (GDS committed, 1 documented ss-corner max-cap caveat). GF180: config-only. 16nm sign-off is future work. See [Status](#status). |

---

## How ChannelQuant works

A few **fixed key channels** carry most of the GQA quant error. ChannelQuant scales
**per channel** on the key path (so those channels get their own scale) and isolates the
worst top-k as FP16 outliers.

**Key path — per-channel INT4 (`cq_key_path`)**
1. Buffer a group of **G=128** key tokens (`residual_buffer`).
2. Take the per-channel max over the group (`amax_unit`, key mode) and freeze **D per-channel
   FP16 scales** (`scale_bank`).
3. Quantize each keep-channel to **INT4**; the top-k outlier channels (CQ-4+, k=2 from a
   static calibrated ROM mask) are held **FP16** instead.

**Value path — per-token INT4 (`cq_value_path`)**
- Per-token amax → FP16 scale → INT4 (INT8 for CQ-8). No grouping.
- **CQ-3-rot (branch):** a fixed Walsh–Hadamard rotation of each value row before the
  per-token amax/INT3 quant (`wht_unit` + `cq_wht_value`) drops values to a flat **3.0
  bits/value**, near-lossless (keys untouched). Hardware runs "Path B": store rotated, sum
  A·V in rotated space, undo the rotation once on the MatE output (`wht_inverse_out`). The
  rotation is reconfigurable (fixed/randomized sign vector; fixed is the default — see
  [`DECISIONS.md`](DECISIONS.md)). See [`docs/wht_value_rotation.md`](docs/wht_value_rotation.md).

**Unified per-channel SRAM record** `{tag, D×FP16 field, D×INT4 code}` — keep channel →
`{group scale, INT4 code}`; **outlier channel → `{raw FP16, code +1}`** so decompress
`code · field` widens the FP16 exactly (no separate sidecar, no read-side mask).

**Area/timing:** each compute core (scale / quant / dequant) carries an fp16 divider, so
instead of D parallel units the datapath **serializes one shared unit** across the D channels
(a single divide cone was what stalled place-and-route). Bit-exact with the behavioral oracle.

---

## Accuracy — verified end-to-end on Qwen2

HellaSwag `acc_norm`, n=1000, G=128, ChannelQuant K̂/V̂ inserted into the model's KV path.
Artifacts: [`analysis/channelquant_hw.py`](analysis/channelquant_hw.py) →
`analysis/cq_hw_qwen*_cq4plus.json`.

| Model | FP16 | CQ-4 (Δ) | CQ-4+ (Δ) | bits/value |
|---|---|---|---|---|
| Qwen2-0.5B (D=64) | 0.4260 | **0.4210 (−0.005)** | 0.4150 (−0.011) | ~4.19 / 4.38 |
| Qwen2-1.5B (D=128) | 0.5220 | 0.5050 (−0.017) | **0.5170 (−0.005)** | ~4.13 / 4.22 |

Both tiers clear the ≤0.02 acceptance gate at **~4 bits/value (≈3.8× KV compression)**.
The **CQ-4+ FP16-outlier lane earns its keep only at D=128** (1.5B: 0.517 vs CQ-4's 0.505);
at **D=64 it slightly hurts**, so plain **CQ-4 is the default at every head dim** (n=1000
reversed the n=250 screening — [`DECISIONS.md`](DECISIONS.md), 2026-07-21). The best tier
per model lands within **~0.5 pt of FP16**.

> **Two eval harnesses, same conclusion.** Some analysis docs quote higher absolute FP16
> baselines (~0.489 / 0.590) from a different `lm_eval` version/subset. The reported metric is
> the **Δ vs FP16**, which is consistent (within the ≤0.02 gate) across both pipelines — do
> not "reconcile" the absolute numbers by overwriting one with the other.

---

## How this fits in LonghornSilicon

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (N16FFC)       │
│                                                                      │
│   ┌──────────────────┐          ┌────────────────────────┐          │
│   │  ACU (block 1)   │  scores  │ Token Importance Unit   │          │
│   │  precision ctrl  │─────────▶│ (block 3)  H2O          │          │
│   │  INT8 vs FP16    │          │ → keep / evict          │          │
│   └────────┬─────────┘          └───────────┬────────────┘          │
│            │  K, V                           │ tier signal            │
│            ▼                                 ▼                        │
│   ┌─────────────────────────┐                                        │
│   │  KV Cache Engine (this)  │ ChannelQuant: K → per-channel INT4    │
│   │  compress on write /     │ (+outlier FP16), V → per-token INT4    │
│   │  decompress on read      │                                        │
│   └─────────────┬───────────┘                                        │
│                 ▼                                                     │
│   ┌─────────────────────────┐   ┌──────────────────────┐             │
│   │ Memory Hierarchy Ctrl.  │◀─▶│ Off-chip LPDDR5X      │             │
│   │ (block 4)               │   │ (cold KV + weights)   │             │
│   └─────────────────────────┘   └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

| Block | Repo | Role |
|---|---|---|
| ACU (Attention Compute Unit) | [`src/blocks/acu`](../acu) | INT8 vs FP16 per tile, MAC array |
| **KV Cache Engine** | **this block** | ChannelQuant compress on write, decompress on read |
| Token Importance Unit | [`src/blocks/tiu`](../tiu) | Per-token keep/evict (H2O) |
| Memory Hierarchy Controller | spec-only stub | On-die SRAM ↔ off-chip LPDDR5X |

---

## Layout

```
src/blocks/kve/
├── rtl/                      # (rev0 only) SystemVerilog DUT + tb/
│   ├── kv_cache_engine.sv    #   Top: AXI-Lite CSR + AXI-Stream, ChannelQuant FSM + SRAM
│   ├── cq_key_path.sv        #   Grouped per-channel INT4 key codec (serialized)
│   ├── cq_value_path.sv      #   Per-token INT4/INT8 value codec (serialized)
│   ├── cq_units_syn.sv       #   Synthesizable fp16 cores (scale/quant/dequant); *_syn = synth-safe
│   ├── cq_units.sv           #   Behavioral `real` oracle (parity TBs only — NOT for synth)
│   ├── amax_unit.sv, residual_buffer.sv, scale_bank.sv, wht_unit*.sv
│   ├── sram_controller.sv, kv_sram.sv   # swappable KV-store (behavioral default; gf180 real SRAM)
│   └── tb/                    # sim, sim_kpath, sim_top, sim_wht_*, … (+ vendored golden vectors)
├── pdk/
│   ├── sky130/openlane/kv_cache_engine/  # LibreLane Sky130 flow + committed results/
│   └── gf180/                # chipathon: librelane/{kve,kve_store_gf180}.yaml + kve_gf180_sram/
├── sw/reference_model/       # channelquant_ref.{hpp,cpp} + Python ref + parity tests
├── analysis/                 # ChannelQuant accuracy studies + golden test-vector gen
├── findings/, research/      # the "why": methodology, dead ends, revamp record
├── docs/                     # ISA spec, WHT rotation doc, HW contract, CI notes
├── NOTES.md                  # dated lab notebook (every parity/synth result)
└── DECISIONS.md              # settled calls
```

---

## Status

Per the authoritative sign-off matrix — [`docs/PROGRESS.md`](../../../docs/PROGRESS.md)
(generated) and [`docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §5.2 for
the sign-off definitions.

| PDK | Macro | Status | Die | Freq |
|---|---|---|---|---|
| **sky130** | `kv_cache_engine` | **signed-off** | 236,211 µm² (0.236 mm²) | 10 MHz (100 ns constraint) |
| gf180 | `kve` | config-only (declared, not run) | — | — |
| gf180 | `kve_store_gf180` | config-only (declared, not run) | — | — |

**Sky130A sign-off** (LibreLane 3.0.5, PDK `8afc8346`, 9 IPVT corners) — GDS committed at
[`pdk/sky130/openlane/kv_cache_engine/results/`](pdk/sky130/openlane/kv_cache_engine/results/)
([`SIGNOFF.md`](pdk/sky130/openlane/kv_cache_engine/results/SIGNOFF.md)). Setup, hold, DRC
(Magic + KLayout), LVS, antenna and power-grid are **all 0 across all 9 corners**. Core util
59%, ~1.79 mW; implied f_max ~24 MHz (ss) / ~49 MHz (tt) / ~78 MHz (ff) at the 100 ns
constraint (WNS 0). RTL/formal verification: `sim_top` + `sim_kpath` bit-exact, 3-way
Python↔C++↔SV reference parity, Yosys synth + FF-count + RTL≡netlist equivalence.

**Two documented caveats (honest report — not sign-off failures):**
- **ss-corner max-cap = 5 (+ max-slew = 1503), entirely at the slow ss corner**, from the
  high-fanout async-reset (`rst_n`) tree across the flop array. Functionally clean
  (recovery/removal slack +90 ns); tracked reset-tree-buffering physical-opt item. Not in the
  headline sign-off counts — see [`SIGNOFF.md`](pdk/sky130/openlane/kv_cache_engine/results/SIGNOFF.md).
- **Storage is a `SRAM_DEPTH=2` flop-array proxy**, not a real Sky130 SRAM macro. The
  ~12.8k std cells / 0.236 mm² reflect logic + depth-2 proxy, not production KV capacity — a
  separately-tracked hole (no real SRAM was faked).

**Not yet:** GF180 hardening run, TSMC 16FFC sign-off (waiting on PDK access), FPGA
prototype, integration with TIU + Memory Hierarchy Controller. Chip roadmap:
[`docs/ROADMAP.md`](../../../docs/ROADMAP.md).

---

## Reproduce

Toolchain: **iverilog 12.0** + **yosys** (CPU-only). RTL is on `rev0`.

```sh
git checkout rev0
cd src/blocks/kve/rtl
make sim_top      # top-level ChannelQuant end-to-end (per-token V + grouped keys), bit-exact
make sim_kpath    # grouped per-channel INT4 key path, 6/6 bit-exact
make sim_wht_pathb_syn  # Path-B store-rotated / unspin-once, bit-exact (CQ-3-rot)
make sim          # + sim_realdata sim_vpath sim_amax sim_syn sim_cq

cd ../sw/reference_model && make test-all   # C++ + Python reference parity

# synthesis / Sky130 sign-off (use the *_syn.sv views — see gotchas):
cd ../rtl && yosys -s synth.ys
librelane pdk/sky130/openlane/kv_cache_engine/config.json
```

---

## Register map (AXI-Lite, ISA v0.2)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | `CTRL` | RW | bit[0]: soft_reset, bit[1]: enable |
| `0x04` | `STATUS` | R | bit[0]: idle, sram_full |
| `0x08` | `INFO_DIM` | R | head dim D |
| `0x0C` | `INFO_TIER` | R | 0=CQ-8, 1=CQ-4, 2=CQ-4+ |
| `0x10` | `INFO_GROUP` | R | key group size G |
| `0x14` | `INFO_SRAM_DEPTH` | R | SRAM entries |
| `0x18` | `INFO_CR_K` | R | key compression ratio (8.8 fixed-point) |
| `0x1C` | `INFO_CR_V` | R | value compression ratio (8.8 fixed-point) |
| `0x20` | `INFO_VERSION` | R | ISA version (`0x00020000` = v0.2) |
| `0x24` | `OCCUPANCY` | R | valid SRAM entries |
| `0x28` | `WRITE_ADDR` | RW | target write / group-base address |
| `0x2C` | `READ_ADDR` | RW | target read address (write launches a decompress) |
| `0x30` | `KV_SELECT` | RW | 0=key, 1=value |
| `0x34` | `IRQ_MASK` | RW | interrupt enable mask |
| `0x38` | `IRQ_STATUS` | R/W1C | interrupt pending status |
| `0x3C` | `INFO_OUTLIER_K` | R | top-k FP16 outlier channels (CQ-4+) |
| `0x40` | `INFO_SCALE_DEPTH` | R | per-channel scale-bank depth (= D) |
| `0x44` | `INFO_RESID_DEPTH` | R | residual-buffer depth (= G) |

Full ISA: [`docs/isa/kv_cache_engine_isa.pdf`](docs/isa/kv_cache_engine_isa.pdf).

---

## Known gotchas
Pitfalls that cost time — check before debugging. (Chip-wide gotchas: monorepo-root `README.md`.)

- **Synth: the behavioral `real`/`$fscanf` views abort yosys.** Use the `*_syn.sv` set
  (`cq_units_syn`, `wht_unit_syn`, `fp16_addsub_syn`, …) for synthesis / LibreLane.
- **FP16 can't be bit-exact to numpy `@`** (BLAS pairwise sum ≠ sequential MAC order). Verify
  FP16 RTL against a **sequential-fp32 golden**, tolerance vs numpy (`rel_err < 5e-3`).
- **Long combinational fp path won't close at the slow corner** (e.g. two serial fp32 mults).
  Pipeline it (register the intermediate); decode is latency-tolerant.
- **gf180 SRAM macro power connects on Metal3.** Route macro power to the M4 straps with a
  legal **Via3** — an M1/M2 route forces illegal Via1/Via2 stacks (7000+ DRC).
- **The gf180 SRAM abstract has ONE sub-min-width pin** (0.11 µm vs 0.28 µm) — an abstract
  artifact; the vendor GDS is clean. Use a DRC-view-only maglef that widens just that pin; run
  LVS on the real device view. (Re-DRC'ing the real GDS throws ~38k false bitcell errors.)
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` DISABLES slew repair** (passes `-slew_margin 0`) — restore
  ~20% or you get thousands of false max-slew/cap violations.

---

## Acknowledgments & citation

The ChannelQuant codec follows the per-channel-key / per-token-value + outlier recipe of
**KIVI** (Liu et al., ICML 2024) and **KVQuant** (Hooper et al., 2024); this block contributes
the streaming silicon implementation. The open flow uses Yosys, OpenROAD, LibreLane, and the
SkyWater Sky130 PDK. WHT value-rotation idea: Abhiram Bandi + Chaithu Talasila.

```bibtex
@misc{kv_cache_engine_2026,
  title  = {KV Cache Engine: A Streaming Silicon Implementation of ChannelQuant
            (Per-Channel INT4) KV-Cache Compression},
  author = {LonghornSilicon},
  year   = {2026},
  url    = {https://github.com/LonghornSilicon/lambda}
}
```
