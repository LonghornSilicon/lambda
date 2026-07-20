# KV Cache Engine

> ⚙️ **You are on the `rtl` branch — the RTL home.** All SystemVerilog, testbenches,
> OpenLane, golden vectors, and the reference-model parity gate live here (not on `main`,
> which carries docs/analysis/reference-source/paper). **The RTL and its sign-off are 130nm
> (SkyWater Sky130)** — the best open-PDK proxy we have; we use it for *16nm estimates and
> functional/physical realizability*, not as the 16nm design itself. The chip target is
> TSMC 16nm (N16FFC); the commercial 16nm implementation (Cadence/Synopsys) is future work.
> Merges to `main` land only docs/analysis/reference; RTL changes stay on this branch.

This is the **KV Cache Engine (KVE)** block of the LonghornSilicon LLM inference
accelerator — block 2 of four targeting TSMC 16nm FinFET (N16FFC) tape-out. It is a streaming
compress-on-write / decompress-on-read engine for transformer KV-cache tensors,
sitting between the ACU (attention compute unit) and the memory hierarchy.

> **Building a compiler / integrating this block?** Start with the chip-level
> [Compiler Programming Guide](https://github.com/LonghornSilicon/architecture/blob/main/docs/compiler_programming_guide.md)
> (KV data format = §5) and the [documentation standard](https://github.com/LonghornSilicon/architecture/blob/main/docs/documentation_standard.md).
> This block's interface spec is [`docs/isa/kv_cache_engine_isa.md`](docs/isa/kv_cache_engine_isa.md).

> ## ✅ ChannelQuant revamp COMPLETE — codec: TurboQuant+ → ChannelQuant
>
> **The block stays; the codec it implements was replaced and is now fully
> integrated, synthesizable, and signed off.** TurboQuant+ (PolarQuant + QJL +
> Walsh–Hadamard rotation) was **retired 2026-06-22**: it reaches ~3.5×
> compression but with a **−0.10 HellaSwag acc_norm collapse on GQA** models
> (0.316 vs 0.420 FP16 on Qwen2-0.5B). Root cause: KV quant error on GQA is
> dominated by a few fixed high-magnitude **key channels**, and the rotation step
> delocalizes that error so no per-token protection catches it.
>
> The successor codec is **ChannelQuant** — **per-channel-key INT4 / per-token-value
> INT4 / static outlier-channel isolation** (the KIVI/KVQuant recipe). The
> algorithm is prior art (KIVI ICML'24, KVQuant 2024); **the contribution of this
> block is the streaming silicon implementation.**
>
> **Status (master, 2026-07-03): DONE.**
> - RTL fully wired into the top (`kv_cache_engine.sv`): keys → grouped per-channel
>   INT4 (`cq_key_path`), values → per-token INT4 (`cq_value_path`), outlier lane +
>   unified per-channel SRAM record. All cores serialized (one shared scale / quant /
>   dequant), no `real`, no latches, checker-clean.
> - **All CI gates green** — functional, synthesis (FF-count), formal
>   RTL≡netlist equivalence, reference-model parity, and OpenLane Sky130 sign-off.
> - **Verified end-to-end on Qwen2** (below): near-FP16 accuracy at ~4 bits/value.
>
> | | |
> |---|---|
> | Retired TurboQuant+ datapath (archived, full history) | branch [`legacy/turboquant-plus`](../../tree/legacy/turboquant-plus) |
> | Algorithm spec + reference model + golden vectors | `../channelquant/` (frozen contract v0.2) |
> | Per-milestone lab notebook | [`NOTES.md`](NOTES.md) |

---

## TL;DR

| | |
|---|---|
| **What** | Streaming compress/decompress engine for transformer KV-cache tensors |
| **Why** | Cuts off-chip LPDDR5X KV-cache bandwidth ~3.8× (near-lossless), enabling longer context in the same memory budget |
| **How** | ChannelQuant — per-channel INT4 keys (grouped, G=128) + per-token INT4 values + static top-k FP16 outlier-channel isolation (CQ-4+) |
| **K/V asymmetry** | K: per-channel scale over a token group (the GQA-critical axis); V: per-token scale |
| **Tiers** | CQ-8 (per-token INT8 K+V), CQ-4 (per-channel INT4 K / per-token INT4 V), CQ-4+ (CQ-4 with k=2 FP16 outlier channels), **CQ-3-rot** (CQ-4+ keys + WHT-rotated per-token **INT3** values — flat 3.0 b/val, ~4.8×; see [`docs/wht_value_rotation.md`](docs/wht_value_rotation.md)) |
| **Verified** | RTL bit-exact vs golden (`sim_kpath`/`sim_top`), 3-way Python↔C++↔SV parity, **all CI gates green** incl. Sky130 sign-off |
| **Accuracy** | HellaSwag acc_norm within ~0.4–0.8 pt of FP16 (CQ-4+ tier) on Qwen2-0.5B/1.5B (see below) |
| **Status** | RTL complete through Sky130 physical sign-off (all CI gates green). 16nm (Lambda) sign-off is future work; full-chip tape-out target Summer 2027 via TSMC University Program |

---

## How ChannelQuant works

The GQA accuracy problem is that a few **fixed key channels** carry most of the
quant error. ChannelQuant scales **per channel** on the key path (so those
channels get their own scale) and isolates the worst top-k as FP16 outliers:

**Key path — per-channel INT4 (`cq_key_path`)**
1. Buffer a group of **G=128** key tokens (`residual_buffer`).
2. Take the per-channel max over the group (`amax_unit`, key mode) and freeze
   **D per-channel FP16 scales** (`scale_bank`).
3. Quantize each keep-channel to **INT4**; the top-k outlier channels (CQ-4+,
   k=2 from a static calibrated ROM mask) are held **FP16** instead.

**Value path — per-token INT4 (`cq_value_path`)**
- Per-token amax → FP16 scale → INT4 (INT8 for the CQ-8 tier). No grouping.
- **CQ-3-rot (branch):** a fixed Walsh-Hadamard rotation of each value row before the per-token
  amax/INT3 quant (`wht_unit` + `cq_wht_value`) drops values to a flat **3.0 bits/value**,
  near-lossless (idea: Abhiram Bandi + Chaithu Talasila). Keys are untouched. Hardware runs
  "Path B": store rotated, sum A·V in rotated space, undo the rotation once on the MatE output
  (`wht_inverse_out`). Reference + RTL bit-exact on real Qwen (348,160/348,160). See
  [`docs/wht_value_rotation.md`](docs/wht_value_rotation.md).

**Unified per-channel SRAM record** `{tag, D×FP16 field, D×INT4 code}`
- Keep channel → `{group scale, INT4 code}`; **outlier channel → `{raw FP16, code +1}`**
  so decompress `code · field` widens the FP16 exactly — no separate sidecar region
  and no read-side mask. Read-back reuses the same per-channel dequant, tag-muxed
  against the value dequant.

**Area/timing:** each compute core (scale / quant / dequant) carries an fp16
divider, so instead of D parallel units the datapath **serializes one shared unit**
across the D channels (a single divide cone is what stalled place-and-route). This
is bit-exact with the behavioral oracle and place-and-routes at a real clock.

---

## Accuracy — verified end-to-end on Qwen2

HellaSwag `acc_norm`, n=1000, ChannelQuant K̂/V̂ inserted into the model's KV path
(reproduced this repo via the frozen `../channelquant` reference):

| Model | FP16 | CQ-4 (Δ) | CQ-4+ (Δ) | bits/value |
|---|---|---|---|---|
| Qwen2-0.5B (D=64) | 0.4260 | 0.4170 (−0.009) | 0.4220 (−0.004) | ~4.19 / 4.38 |
| Qwen2-1.5B (D=128) | 0.5210 | 0.5050 (−0.016) | **0.5130 (−0.008)** | ~4.13 / 4.22 |

Both tiers clear the ≤0.02 acceptance gate at **~4 bits/value (≈3.8× KV
compression)**; the CQ-4+ outlier lane earns its keep at D=128. Combined with the
ACU precision controller (INT8/FP16-routed S·V) the system holds accuracy at FP16
(no measurable loss on Qwen2-0.5B).

---

## How this fits in LonghornSilicon

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (16FFC)       │
│                                                                      │
│   ┌──────────────────┐                                               │
│   │  ACU (block 1)   │  Q·Kᵀ scores                                  │
│   │  precision       │──────────────────┐                            │
│   │  controller      │                   ▼                           │
│   │  INT8 vs FP16    │          ┌────────────────────┐               │
│   │  gate per tile   │          │ Token Importance    │               │
│   │  + INT8/FP16 MAC │          │ Unit (block 3)      │               │
│   └────────┬─────────┘          └─────────┬──────────┘               │
│            │  K, V                        │ tier signals              │
│            ▼                              ▼                           │
│   ┌─────────────────────────┐                                        │
│   │  KV Cache Engine        │  ChannelQuant compress on writes,      │
│   │  (this repo)            │  decompress on reads:                  │
│   │                         │  K → per-channel INT4 (+outlier FP16)  │
│   │                         │  V → per-token INT4                     │
│   └─────────────┬───────────┘                                        │
│                 ▼                                                     │
│   ┌─────────────────────────┐   ┌──────────────────────┐             │
│   │ Memory Hierarchy Ctrl.  │◀─▶│ Off-chip LPDDR5X      │             │
│   │ (block 4)               │   │ (cold KV + weights)   │             │
│   └─────────────────────────┘   └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

| Block | This repo? | Role |
|---|---|---|
| **ACU (Attention Compute Unit)** | no ([repo](https://github.com/LonghornSilicon/adaptive-precision-attention)) | Decides INT8 vs FP16 per tile, runs the MAC array |
| **KV Cache Engine** | **this repo** | ChannelQuant compress on write, decompress on read |
| **Token Importance Unit** | not yet | Tracks attention weight per cached token → keep / demote / evict |
| **Memory Hierarchy Controller** | not yet | Routes between on-die SRAM and off-chip LPDDR5X (direct; no eDRAM tier) |

The two live blocks coordinate at attention time: the KVE decompresses K/V → the ACU
computes Q·Kᵀ scores → the precision controller routes INT8/FP16 → the MAC array
runs the matmul.

---

## What's in this repo

```
kv-cache-engine/
├── rtl/
│   ├── kv_cache_engine.sv        # Top: AXI-Lite CSR + AXI-Stream, ChannelQuant FSM + SRAM
│   ├── cq_key_path.sv            # Grouped per-channel INT4 key codec (serialized)
│   ├── cq_value_path.sv          # Per-token INT4/INT8 value codec (serialized)
│   ├── cq_units_syn.sv           # Synthesizable fp16 cores: scale / quant / dequant
│   ├── cq_units.sv, cq_fp_pkg.sv # Behavioral `real` oracle (for the parity TBs)
│   ├── amax_unit.sv              # Per-token / per-channel max reduction
│   ├── residual_buffer.sv        # G-token group hold (key path)
│   ├── scale_bank.sv             # D per-channel scale bank (key path)
│   ├── sram_controller.sv        # Behavioral SRAM (reg array)
│   ├── tb/                       # sim, sim_realdata, sim_cq, sim_amax, sim_vpath,
│   │                             #   sim_kpath, sim_top, sim_syn  (+ vendored golden vectors)
│   ├── constraints/, *.tcl, synth.ys, Makefile
│   └── KEYPATH_HANDOFF.md, TEARDOWN.md, NOTES pointers
├── openlane/kv_cache_engine/     # LibreLane / OpenROAD Sky130 flow (+ src/ symlinks)
├── sw/reference_model/           # channelquant_ref.{hpp,cpp} (ChannelQuant C++ ref) + tests
├── docs/                         # ISA spec, reference-model API, sw overview, CI docs
├── NOTES.md                      # dated lab notebook (every parity/synth result)
└── .github/workflows/ci.yml      # thin caller → shared block-ci reusable workflow
```

The retired TurboQuant+ modules (`rotation_unit`, `qjl_unit`, `quantizer`,
`packer`, `decompressor`, `norm_unit`) live on branch `legacy/turboquant-plus`.

---

## Verification & results

**RTL (this host, iverilog 12.0 / yosys):**
- `make sim_top` — per-token INT4 V **and** grouped CQ-4+ keys **bit-exact** through
  the AXI FSM + SRAM (D=64, G=64, k=2).
- `make sim_kpath` — 6/6 bit-exact (serialized key path: scale + INT4 payload + K̂ +
  sidecar, full and partial groups).
- `make sim sim_realdata sim_vpath sim_amax sim_syn sim_cq` — all green.
- `yosys proc; check` on the top — **0 "conflicting with a constant", 0 latches, 0
  CHECK problems, no `real`.**

**CI gates (all green):**

| Gate | What it does | Status |
|---|---|---|
| 1. RTL functional verification | Directed + replay + parity iverilog TBs | ✅ |
| 3. RTL synthesis (Yosys) | Synth + FF-count assertion | ✅ |
| 4. Formal equivalence | RTL ≡ post-synth netlist (Yosys induction) | ✅ |
| 5. Reference model tests | C++ + Python bit-exact (3-way parity) | ✅ |
| 6. OpenLane Sky130 sign-off | Full Sky130 PnR + DRC/LVS | ✅ |
| 2 / 7 / 8 | coverage / paper / Cadence 16FFC | disabled |

The synthesis/formal/OpenLane gates run a small **flop-based gate proxy** of the
default params (the SRAM and residual buffer are behavioral flip-flops, no Sky130
macro); the real head-dim / group / depth are set per-instantiation (every TB
overrides them). See the FF-count note in `.github/workflows/ci.yml`.

---

## Reproduce

Toolchain: **iverilog 12.0** + **yosys** (CPU-only). On a fresh host see the
per-host EDA-env notes; `. rtl/eda-env.sh` puts both on PATH.

```sh
cd rtl
make sim_top      # top-level ChannelQuant end-to-end (per-token V + grouped keys), bit-exact
make sim_kpath    # grouped per-channel INT4 key path, 6/6 bit-exact
make sim_cq       # golden-vector parity, all 9 vectors (behavioral oracle)
make sim sim_realdata sim_vpath sim_amax sim_syn   # the rest of the board

# reference-model parity (C++ + Python):
cd ../sw/reference_model && make test-all

# synthesis / Sky130 sign-off:
cd ../../rtl && yosys -s synth.ys
cd ../openlane/kv_cache_engine && librelane --docker-no-tty --dockerized config.json
```

End-to-end accuracy on Qwen2 is reproduced from the frozen `../channelquant`
reference (`analysis/c23_headline.py`, HellaSwag); the algorithm accuracy claims
live in that repo's contract.

---

## Register map (AXI-Lite, ISA v0.2)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | `CTRL` | RW | bit[0]: soft_reset, bit[1]: enable |
| `0x04` | `STATUS` | R | bit[0]: idle, sram_full |
| `0x08` | `INFO_DIM` | R | head dim D |
| `0x0C` | `INFO_TIER` | R | 0=CQ-8, 1=CQ-4, 2=CQ-4+ |
| `0x10` | `INFO_GROUP` | R | key group size G (contract §3.1) |
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

Full ISA specification: [`docs/isa/kv_cache_engine_isa.pdf`](docs/isa/kv_cache_engine_isa.pdf).

---

## Status & roadmap

- [x] Codec pivot TurboQuant+ → ChannelQuant (algorithm de-risked in `../channelquant`)
- [x] Synthesizable fp16 compute cores (scale / quant / dequant), bit-exact vs oracle
- [x] Per-token value path + grouped per-channel INT4 key path (serialized)
- [x] Outlier-channel lane (CQ-4+) + static ROM mask
- [x] Top-level integration (AXI-Lite CSR + AXI-Stream), unified per-channel SRAM record
- [x] Directed / replay / parity / top-stream testbenches — all green, bit-exact
- [x] 3-way Python↔C++↔SV reference parity
- [x] Yosys synthesis + FF-count + formal RTL≡netlist equivalence (CI green)
- [x] OpenLane Sky130 sign-off (CI green)
- [x] End-to-end accuracy on Qwen2-0.5B / 1.5B (near-FP16 at ~4 bits)
- [ ] Partial-group flush (g<G) top stream-framing (datapath already supports it)
- [ ] TSMC 16FFC sign-off on Cadence (waiting on PDK access)
- [ ] ZCU102/104 FPGA prototype (Vivado, when board arrives)
- [ ] Integration with Token Importance Unit, Memory Hierarchy Controller
- [ ] Full-chip tape-out via TSMC University Program shuttle (Lambda 16nm track, target Summer 2027)

---

## Citation

```bibtex
@misc{kv_cache_engine_2026,
  title  = {KV Cache Engine: A Streaming Silicon Implementation of ChannelQuant
            (Per-Channel INT4) KV-Cache Compression},
  author = {LonghornSilicon},
  year   = {2026},
  url    = {https://github.com/LonghornSilicon/kv-cache-engine}
}
```

## Acknowledgments

The ChannelQuant codec follows the per-channel-key / per-token-value + outlier
recipe of **KIVI** (Liu et al., ICML 2024) and **KVQuant** (Hooper et al., 2024);
this block contributes the streaming silicon implementation. The open hardware
flow uses [Yosys](https://github.com/YosysHQ/yosys),
[OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD),
[LibreLane](https://github.com/librelane/librelane), and the
[SkyWater Sky130 PDK](https://github.com/google/skywater-pdk).
