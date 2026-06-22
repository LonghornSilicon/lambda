# KV Cache Engine

This is the **KV Cache Engine** block of the LonghornSilicon LLM inference
accelerator — block 2 of four targeting TSMC 16FFC tape-out. It is a streaming
compress-on-write / decompress-on-read engine for transformer KV-cache tensors,
sitting between the ACU and the memory hierarchy.

> ## 🔧 Revamp in progress — codec: TurboQuant+ → ChannelQuant
>
> **The block stays; the codec it implements is being replaced.** TurboQuant+
> (PolarQuant + QJL + Walsh–Hadamard rotation) was **retired 2026-06-22**: it
> reaches ~3.5× compression but with a **−0.10 HellaSwag acc_norm collapse on
> GQA** models (0.316 vs 0.420 FP16 on Qwen2-0.5B). Root cause: KV quant error
> on GQA is dominated by a few fixed high-magnitude **key channels**, and the
> rotation step delocalizes that error so no per-token protection catches it.
>
> The successor codec is **ChannelQuant** — per-channel-key INT4 / per-token-value
> INT4 / static outlier-channel isolation (the KIVI/KVQuant recipe). It reaches
> **~3.6–3.8× near-lossless** (7B: 0.604 vs 0.612 FP16) where naive INT4 collapses
> to chance. The algorithm is prior art (KIVI ICML'24, KVQuant 2024); **the
> contribution of this block is the streaming silicon implementation.**
>
> | | |
> |---|---|
> | Revamp plan (this block) | [`findings/channelquant_block_revamp.md`](findings/channelquant_block_revamp.md) |
> | Algorithm spec + reference model + golden vectors | `../channelquant/` |
> | Retired TurboQuant+ datapath (archived, full history) | branch [`legacy/turboquant-plus`](../../tree/legacy/turboquant-plus) |
>
> **Status of the revamp:** datapath teardown started (new `amax_unit`,
> `residual_buffer`, `scale_bank` skeletons added; `rotation_unit`/`qjl_unit`
> scheduled for removal — see the manifest in [`rtl/TEARDOWN.md`](rtl/TEARDOWN.md)).
> The TurboQuant+ documentation below the line is retained until the ChannelQuant
> datapath replaces it.
>
> ⚠️ Everything below describes the **TurboQuant+ datapath now being torn down** —
> compression figures, register map, and verified-FF counts are the predecessor's,
> not yet ChannelQuant's.

---

## (predecessor) KV Cache Engine — TurboQuant+

A hardware-verified KV cache compression engine using
[TurboQuant+](https://github.com/themoddedcube/turboquant-plus)
that compresses key vectors to **4.25 bits per value** and value vectors
to **~3.0 bpv** at runtime, giving a **3--5x DRAM bandwidth reduction**
for KV cache traffic with bounded reconstruction error.

---

## TL;DR

| | |
|---|---|
| **What** | Streaming compress/decompress engine for transformer KV cache tensors |
| **Why** | Cuts KV-cache DRAM bandwidth 3--5x, enabling longer context in the same memory budget |
| **How** | TurboQuant+ turbo4: PolarQuant (3-bit Lloyd-Max) + QJL (1-bit sign projection) |
| **K/V asymmetry** | K gets PolarQuant+QJL (inner product preservation), V gets PolarQuant only (MSE-optimized) |
| **Hardware cost** | ~6400 FFs estimated (full pipeline); ~93 FFs current top-level |
| **Verified** | Directed + replay testbenches, 120 Python + 64 C++ reference model tests |
| **Status** | Tape-out target Q3/Q4 2026 via TSMC University Program 16FFC |

---

## How this improves on standard KV caching

### The problem: KV cache is the memory bottleneck

In autoregressive transformer inference, every generated token must
attend to all previous tokens' key and value vectors. For long contexts
(4K--128K tokens), the KV cache dominates both on-chip SRAM capacity
and off-chip DRAM bandwidth. At FP16, a 4096-token cache with
dimension 64 consumes 1 MB — and that scales linearly with sequence
length, batch size, and model depth.

### Prior work: quantization helps but loses accuracy

Naive INT8 or INT4 quantization reduces storage but introduces
systematic bias in attention scores (keys) and accumulated values.
Published schemes (GEAR, RotateKV, Lexico) address this with
varying quality/complexity tradeoffs, but none provide a clean
hardware-friendly pipeline with asymmetric K/V treatment.

### Our contribution: TurboQuant+ with asymmetric K/V compression

TurboQuant+ uses a two-stage approach optimized differently for
keys and values:

**Key path (4.25 bpv):**
1. **Normalize** — extract L2 norm as metadata
2. **Rotate** — Walsh-Hadamard Transform + random sign flips
   (makes coordinates i.i.d. Gaussian, regardless of input distribution)
3. **Quantize** — 3-bit optimal Lloyd-Max scalar quantization
4. **QJL project** — 1-bit Quantized Johnson-Lindenstrauss on the
   residual (preserves inner products for attention score accuracy)

**Value path (~3.0 bpv):**
Same pipeline minus QJL — values only need low MSE, not inner
product preservation.

The rotation step is the key insight: WHT is O(n log n), self-inverse
(same hardware for compress and decompress), and makes Lloyd-Max
quantization near-optimal regardless of the input vector's coordinate
distribution.

### Why this matters for the hardware budget

| | Uncompressed (FP16) | This work (TurboQuant+ turbo4) |
|---|---|---|
| Key storage per vector (D=64) | 1024 bits | 288 bits (3.56x) |
| Value storage per vector | 1024 bits | 208 bits (4.92x) |
| DRAM bandwidth per token | 1.0x | **~0.28x** average |
| On-chip SRAM capacity | 1.0x | **~3.5x** effective |
| Additional silicon | none | ~6400 FFs (~0.5% of a typical attention tile) |

---

## How this fits in LonghornSilicon

The KV cache engine is one of four blocks in the
LonghornSilicon LLM inference accelerator:

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (16FFC)       │
│                                                                      │
│   ┌──────────────────┐                                               │
│   │  ACU             │ Q·K^T scores                                  │
│   │  (block 1)       │─────────────────┐                             │
│   │                  │                  ▼                             │
│   │  precision_      │         ┌────────────────────┐                │
│   │  controller      │         │  Token Importance   │                │
│   │  ────────────    │         │  Unit (block 3)     │                │
│   │  INT8 vs FP16    │         │                     │                │
│   │  gate per tile   │         │  Per-token weight   │                │
│   │                  │         │  → keep / evict     │                │
│   │  + INT8/FP16     │         └─────────┬──────────┘                │
│   │  MAC array       │                   │ tier signals               │
│   └────────┬─────────┘                   │                           │
│            │  K, V                       ▼                           │
│            │               ┌─────────────────────────┐               │
│            └──────────────▶│  KV Cache Engine         │               │
│                            │  (this repo)             │               │
│                            │                          │               │
│                            │  TurboQuant+ compress    │               │
│                            │  on writes, decompress   │               │
│                            │  on reads                │               │
│                            └─────────────┬───────────┘               │
│                                          │                           │
│                           ┌──────────────┴──────────────┐            │
│                           ▼                              ▼            │
│             ┌─────────────────────────┐   ┌──────────────────────┐   │
│             │  Memory Hierarchy Ctrl. │   │  Off-chip LPDDR5     │   │
│             │  (block 4)              │◀─▶│  (cold KV + model    │   │
│             │                         │   │   weights)            │   │
│             │  L1 SRAM (hottest KV)   │   └──────────────────────┘   │
│             │  L2 eDRAM 8-32 MB       │                              │
│             │  SHIELD refresh control │                              │
│             └─────────────────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
```

| Block | This repo? | Role |
|---|---|---|
| **ACU (Attention Compute Unit)** | no ([repo](https://github.com/LonghornSilicon/adaptive-precision-attention)) | Decides INT8 vs FP16 per tile, runs the MAC array |
| **KV Cache Engine** | **this repo** | Compresses K/V on write, decompresses on read (TurboQuant+ turbo4) |
| **Token Importance Unit** | not yet | Tracks attention weight per cached token → keep / demote / evict |
| **Memory Hierarchy Controller** | not yet | Routes between L1 SRAM / L2 eDRAM / off-chip LPDDR5 |

The two blocks coordinate at attention time: the KV cache engine
decompresses K/V → the ACU computes Q·K^T scores → the precision
controller routes INT8/FP16 → the MAC array runs the matmul.

---

## What's in this repo

```
kv-cache-engine/
├── analysis/                  # Python: algorithm exploration, test-vector generation
├── rtl/
│   ├── kv_cache_engine.sv          # Top-level module (AXI-Lite + AXI-Stream)
│   ├── norm_unit.sv                # L2 norm: adder tree + integer sqrt
│   ├── rotation_unit.sv            # WHT butterfly + random sign flips
│   ├── quantizer.sv                # 3-bit Lloyd-Max nearest-centroid
│   ├── qjl_unit.sv                 # QJL 1-bit sign projection (K only)
│   ├── packer.sv                   # Bit-pack/unpack for SRAM
│   ├── sram_controller.sv          # Behavioral SRAM (reg array, dual-port)
│   ├── decompressor.sv             # Reverse compression pipeline
│   ├── tb/                         # Self-checking + replay testbenches
│   ├── constraints/                # SDC: 16FFC (800 MHz) + Sky130 (80 MHz)
│   ├── genus.tcl, innovus.tcl      # Cadence flow (waiting on TSMC PDK)
│   ├── synth.ys                    # Yosys synthesis script
│   └── Makefile                    # Targets: sim, sim_realdata, testvectors, synth
├── openlane/
│   └── kv_cache_engine/            # LibreLane / OpenROAD flow targeting Sky130A
│       └── config.json             # Design config (80 MHz target)
├── sw/
│   ├── README.md                   # Top-level orientation for the compiler team
│   └── reference_model/
│       ├── kv_cache_engine_ref.{hpp,cpp,py}   # Bit-exact reference (120+64 tests)
│       ├── test_*.{cpp,py}                    # C++ and Python test suites
│       └── Makefile                           # test, test-all, shared, static
├── docs/
│   ├── isa/kv_cache_engine_isa.{tex,pdf}      # ISA spec (kv-isa-0.1)
│   ├── reference_model_api.{tex,pdf}          # Formal C++ / C / Python API reference
│   ├── sw_overview.{tex,pdf}                  # Compiler-team orientation PDF
│   ├── ci_overview.md                         # CI pipeline reference
│   ├── ci_setup.md                            # GitHub Actions runner setup
│   └── chamber_setup.md                       # End-to-end Cadence chamber walkthrough
├── .github/workflows/ci.yml      # CI: RTL verif → synth → Sky130 signoff → ref tests
└── README.md (this file)
```

---

## Results

### Compression quality (algorithmic)

- **120 Python + 64 C++ reference model tests pass** — round-trip
  compress/decompress, K/V asymmetry, streaming vs batch agreement,
  C API compatibility, RTL replay from hex vectors.
- **Compression ratio**: K = 3.56x (4.25 bpv), V = 4.92x (~3.0 bpv),
  combined >= 3x on canonical trace.
- **Reconstruction**: bounded MSE below threshold for both K and V paths.

### RTL verification

- **14 directed testbench cases** — all states, reset recovery,
  pipeline stall, centroid bucket coverage.
- **Replay testbench** — hex-file driven, bit-exact match against
  reference model outputs.
- **Python / C++ / SV three-way parity** — compress AND decompress
  paths verified bit-exact.

### Hardware (current top-level, Yosys)

| Metric | Value |
|---|---|
| Frequency target (Sky130) | **80 MHz** (12.5 ns) |
| Frequency target (16FFC) | **800 MHz** (1.25 ns) |
| Top-level FFs (Yosys, current wiring) | ~93 (sram_controller + FSM) |
| Estimated full-pipeline FFs | ~6400 (all sub-modules wired) |
| Yosys synthesis | Clean (zero warnings) |

---

## Reproduce

### Functional verification (~30 sec)

```sh
cd rtl
make testvectors   # generate hex vectors from Python reference
make sim           # directed testbench — should print ALL TESTS PASSED
make sim_realdata  # replay testbench — should print ALL TESTS PASSED
```

### Reference model tests (~10 sec)

```sh
cd sw/reference_model
make test-all      # builds C++ tests, runs them, runs Python tests
```

### Yosys synthesis (~10 sec)

```sh
cd rtl
yosys -s synth.ys  # or: make synth
# Reports cell/FF counts after generic synth + NAND mapping
```

### Sky130 OpenLane signoff (~5-10 min)

Requires Docker (~25 GB free) and `pip install librelane`:

```sh
cd openlane/kv_cache_engine
librelane --docker-no-tty --dockerized config.json
# Final metrics at runs/<latest>/final/metrics.json
```

### Cadence flow (TSMC 16FFC, when PDK is provisioned)

```sh
cd rtl
genus -files genus.tcl -log reports/genus.log
innovus -files innovus.tcl -log reports/innovus.log
```

PDK paths are at the top of each Tcl file. The end-to-end walkthrough
for fresh-chamber sign-off is in [`docs/chamber_setup.md`](docs/chamber_setup.md).

---

## CI / CD

The thin caller at [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
delegates to the
[LonghornSilicon shared `block-ci` reusable workflow](https://github.com/LonghornSilicon/.github/blob/main/.github/workflows/block-ci.yml),
which defines 8 gates for every block in the organization:

| Gate | Runner | What it does |
|---|---|---|
| 1. RTL functional verification | GitHub Ubuntu | Directed + replay iverilog testbenches |
| 2. Line coverage gate | GitHub Ubuntu | Verilator-based line coverage (disabled for now) |
| 3. RTL synthesis (Yosys) | GitHub Ubuntu | Synth + FF-count assertion (expected: 72) |
| 4. Formal equivalence | GitHub Ubuntu | RTL ≡ post-synth netlist via Yosys |
| 5. Reference model tests | GitHub Ubuntu | C++ + Python bit-exact verification |
| 6. OpenLane Sky130 sign-off | GitHub Ubuntu | Full Sky130 PnR; **fails if any violation** |
| 7. Paper build | GitHub Ubuntu | pdflatex compile (disabled — no paper) |
| 8. Cadence 16FFC sign-off | Self-hosted | Genus + Innovus (disabled until PDK + licenses) |

Detailed CI walkthrough: [`docs/ci_overview.md`](docs/ci_overview.md).
Runner setup: [`docs/ci_setup.md`](docs/ci_setup.md).

---

## Register map (AXI-Lite)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | `CTRL` | RW | bit[0]: soft_reset, bit[1]: enable |
| `0x04` | `STATUS` | R | bit[0]: idle, bit[3]: sram_full |
| `0x08` | `INFO_DIM` | R | VECTOR_DIM (64) |
| `0x0C` | `INFO_PQ_BITS` | R | PQ quantization bits (3) |
| `0x10` | `INFO_QJL_BITS` | R | QJL projection bits (1) |
| `0x14` | `INFO_SRAM_DEPTH` | R | SRAM entries (16) |
| `0x18` | `INFO_CR_K` | R | Key compression ratio (8.8 fixed-point) |
| `0x1C` | `INFO_CR_V` | R | Value compression ratio (8.8 fixed-point) |
| `0x20` | `INFO_VERSION` | R | ISA version (0x00010000) |
| `0x24` | `OCCUPANCY` | R | Valid SRAM entries |
| `0x28` | `WRITE_ADDR` | RW | Target write address |
| `0x2C` | `READ_ADDR` | RW | Target read address |
| `0x30` | `KV_SELECT` | RW | 0=key, 1=value |
| `0x34` | `IRQ_MASK` | RW | Interrupt enable mask |
| `0x38` | `IRQ_STATUS` | R/W1C | Interrupt pending status |

Full ISA specification: [`docs/isa/kv_cache_engine_isa.pdf`](docs/isa/kv_cache_engine_isa.pdf).

---

## Status & roadmap

- [x] Algorithm selection (TurboQuant+ turbo4, asymmetric K/V)
- [x] Python golden reference model (120 tests)
- [x] C++ bit-exact reference model (64 tests)
- [x] SystemVerilog RTL sub-modules (norm, rotation, quantizer, QJL, packer, decompressor)
- [x] Top-level integration with AXI-Lite + AXI-Stream
- [x] Directed + replay testbenches (ALL TESTS PASSED)
- [x] Yosys synthesis (clean)
- [x] OpenLane Sky130 config + CI
- [x] ISA specification (kv-isa-0.1)
- [x] Cadence scripts (Genus + Innovus + MMMC)
- [ ] **Wire full compression pipeline into top-level FSM**
- [ ] **OpenLane Sky130 sign-off (zero violations, all corners)**
- [ ] **TSMC 16FFC sign-off on Cadence (waiting on PDK access)**
- [ ] **ZCU102/104 FPGA prototype (Vivado, when board arrives)**
- [ ] Integration with Precision Controller, Token Importance Unit, Memory Hierarchy Controller
- [ ] Full-chip tape-out via TSMC University Program shuttle (target Q3/Q4 2026)

---

## Citation

```bibtex
@misc{kv_cache_engine_2026,
  title  = {KV Cache Engine: Hardware TurboQuant+ Compression for Transformer KV Caches},
  author = {LonghornSilicon},
  year   = {2026},
  url    = {https://github.com/LonghornSilicon/kv-cache-engine}
}
```

## Acknowledgments

This work uses [TurboQuant+](https://github.com/themoddedcube/turboquant-plus)
for the compression algorithm, built on the QJL framework
(Zandieh et al., 2024) and PolarQuant (Lin et al., 2024). The open
hardware flow uses
[Yosys](https://github.com/YosysHQ/yosys),
[OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD),
[LibreLane](https://github.com/librelane/librelane), and the
[SkyWater Sky130 PDK](https://github.com/google/skywater-pdk).
