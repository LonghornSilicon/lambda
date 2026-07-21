# Lambda ACU — SSCS Chipathon 2026 submission

**LonghornSilicon "Lambda" decode-attention datapath (Attention Compute Unit).**
An open-source accelerator for LLM-inference **attention + KV-cache**, hardened
for the Chipathon 2026 GF180MCU shuttle as a **multi-macro** design stitched into
the workshop padring.

- **Track:** A (open-source digital, RTL→GDS) — candidate for the Track-D digital
  showcase.
- **Shuttle PDK:** GF180MCU (GlobalFoundries 180 nm).
- **Flow:** LibreLane **Classic** per block macro → **Chip** flow into the
  chipathon-2026 workshop padring.
- **Verification:** cocotb (Python), RTL and gate-level against gf180 cells.

> **Status: honest skeleton.** The submission *harness* (workshop-slot
> `chip_core`, serial host loader, per-macro LibreLane configs, cocotb harness)
> is real and the smoke test passes. The datapath macros are dropped into their
> reserved instantiation sites as their RTL is hardened for GF180. Five blocks
> already have real, Sky130-signed RTL (copied in under `rtl/blocks/`); two are
> still being written. See [Block → macro mapping](#block--macro-mapping) and
> [What's still needed](#whats-still-needed). Every stub is marked `TODO`.

---

## Decode-attention datapath

One decode step (single query token attending over the KV cache):

```
   host ── SPI ──► Q, K, V  (INT8, head-dim D=128)
                     │
        ┌────────────┴─────────────────────────────────────────────┐
        │  kve            store/stream K,V  (ChannelQuant KV codec)  │
        │  mate_qkt       scores  = Q·Kᵀ                            │
        │  precision_ctrl decide INT8 vs FP16 per tile (d_fp16)      │
        │  vecu_softmax   P = softmax(scores)                       │
        │  tiu            H2O heavy-hitter scoring / eviction pick    │
        │  mate_pv / mate_pv_fp16   o = Σ P·V  (INT8 signed / FP16)  │
        └────────────┬─────────────────────────────────────────────┘
                     │
   host ◄─ SPI ──── o  (attention output row, D-wide)
```

The datapath is much wider (D=128 INT8 lanes for Q/K/V) than the ~20 bidir pads
the workshop slot provides, so tensors are **streamed** in and the result
streamed out over a narrow serial (SPI) link — see [Host serial protocol](#host-serial-protocol).

## Block → macro mapping

Each block is hardened as its **own GF180 LibreLane macro**, then instantiated in
`rtl/lambda_acu.sv` and stitched into the padring via `rtl/chip_core.sv`.

GF180 status below is **measured** — see [`docs/gf180_gls_report.md`](docs/gf180_gls_report.md)
for full per-macro signoff + the gate-level end-to-end match numbers.

| Macro (`DESIGN_NAME`)   | Function                                   | GF180 hardened (LibreLane Classic)                 | GF180 gate-level match vs reference | LibreLane config |
|-------------------------|--------------------------------------------|----------------------------------------------------|-------------------------------------|------------------|
| `precision_controller`  | per-tile INT8-vs-FP16 decision (no divide) | ✅ **clean** (6/6, all corners); 20 911 µm², ~40 MHz | ✅ gate decision matches, discriminates | `precision_controller.yaml` ✅ |
| `token_importance_unit` | H2O heavy-hitter scoring / eviction        | ✅ setup/hold/DRC/LVS/antenna clean; slew@ss only; 35 966 µm², ~58 MHz | ✅ keep-tier + evict victim match | `token_importance_unit.yaml` ✅ |
| `mate_pv`               | INT8 P·V token-reduction MAC               | ✅ setup/hold/DRC/LVS/antenna clean; slew@ss only; 170 348 µm², ~36 MHz | ✅ **INT32 bit-exact** vs matmul_int8 | `mate_pv.yaml` ✅ |
| `mate_pv_fp16`          | FP16 P·V MAC (fp32 internal accumulate)    | ✅ setup/hold/DRC/LVS/antenna clean; slew@ss only; 508 460 µm², ~6.7 MHz | ✅ **rel_err 1.7e-4** (< 5e-3) | `mate_pv_fp16.yaml` ✅ |
| `kv_cache_engine` (kve) | ChannelQuant KV-cache codec (store/stream) | 🟠 codec hardens as logic; KV **store now on a REAL `gf180mcu_fd_ip_sram` macro** — placed/timed/routed-clean/PSM-connected, bit-exact round-trip; DRC/LVS of the macro PDN vias still open | store round-trip bit-exact thru real SRAM | `kve.yaml` + `kve_store_gf180.yaml` ✅ |
| `mate_qkt`              | Q·Kᵀ decode scoring (fp16, fp32 accum)     | ✅ setup/hold/DRC/LVS/antenna clean; slew@ss only; 904 834 µm², ~6.7 MHz | ✅ **scores rel_err 0.0** (< 5e-3) → gate | `mate_qkt.yaml` ✅ |
| `vecu_softmax` (pipelined) | decode online softmax (exp-LUT)         | ✅ setup/hold/DRC/LVS/antenna clean; **ss +19.2 ns** (pipelined); slew@ss; 1 486 490 µm², ~3.6 MHz | ✅ **softmax 2.1e-4 / attn-out 5.4e-4** (within tol) | `vecu_softmax.yaml` ✅ |

All seven compute blocks are **re-hardened on GF180MCU** (LibreLane 3.0.5 Classic;
clocks re-timed — GF180 180 nm is much slower). The **full compute datapath
Q·Kᵀ → softmax → P·V** (plus the ACU gate and TIU) passes a **GF180 gate-level
end-to-end** check (INT bit-exact / FP16 rel_err < 5e-3 / softmax within LUT tol)
on a real Qwen tile (`tb/tb_gls_e2e.sv`, `make test-gls-e2e`). Signoff is
**setup + hold + DRC + LVS + antenna clean across the whole datapath at every
corner** (`vecu_softmax` was pipelined to close the last ss-corner setup miss —
now ss +19.2 ns); the one remaining physical item is slow-corner (`ss`)
max-transition on the big fp16 / register-array blocks. **KVE store — now on real
SRAM:** the KV store was refactored behind a swappable `kv_sram` interface and
backed by real **`gf180mcu_fd_ip_sram`** hard macros — a 512-word store round-trips
KV records **bit-exact** through the real macro (`make test-kv-sram-gf180`), and
hardens with **4 SRAM macros placed, timed, routed-clean, antenna-clean, and PDN
power-connected** (`kve_store_gf180.yaml`). The remaining open item is clean
**Magic-DRC + LVS of the macro PDN vias** (electrically connected, geometry needs
refinement) — see [`docs/gf180_gls_report.md`](docs/gf180_gls_report.md) §4.
Provenance (source repo, branch, commit) is in
[`rtl/blocks/PROVENANCE.md`](rtl/blocks/PROVENANCE.md).

## Repository layout

```
chipathon-lambda-acu/
├── README.md                     # this file (submission doc)
├── rtl/
│   ├── chip_core.sv              # workshop-slot override (verbatim pad interface)
│   ├── lambda_acu.sv             # ACU top: SPI loader + macro instantiation sites
│   ├── spi_loader.sv             # serial host loader (SKELETON + documented protocol)
│   └── blocks/                   # real block RTL, copied from sibling repos
│       ├── PROVENANCE.md
│       ├── kve/*.sv              # KV-cache engine (16 files)
│       ├── token_importance_unit.sv
│       ├── precision_controller.sv
│       ├── mate_pv.sv
│       └── mate_pv_fp16.sv
├── librelane/                    # one Classic-flow config per macro (GF180)
│   ├── kve.yaml                        (real)
│   ├── token_importance_unit.yaml      (real)
│   ├── precision_controller.yaml       (real)
│   ├── mate_pv.yaml                    (real)
│   ├── mate_pv_fp16.yaml               (real)
│   ├── mate_qkt.yaml                   (real)
│   └── vecu_softmax.yaml               (real)
├── tb/                           # cocotb harness (mirrors the template)
│   ├── Makefile                  # dispatcher: make test-smoke / test-all
│   ├── Makefile.cocotb           # cocotb standard include
│   ├── timescale.v               # 1ns/1ps for GL sim against gf180 cells
│   ├── chip_core_wrap.sv         # pins pad widths to SLOT_WORKSHOP for elaboration
│   └── test_smoke.py             # elaborates chip_core + 1 SPI START frame
└── docs/
    ├── build.md                  # per-macro LibreLane + padring integration steps
    └── architecture.md           # datapath / pad-map / diagram descriptions
```

## Host serial protocol

The workshop slot gives the core **1 input pad + 20 bidir pads**. The host talks
to the ACU over a 4-wire **SPI slave** on the first four bidir pads; the rest are
debug/observation. Full contract lives in the header of
[`rtl/spi_loader.sv`](rtl/spi_loader.sv).

Pad map (`rtl/chip_core.sv`):

| bidir pad | dir    | signal     |
|-----------|--------|------------|
| `[0]`     | in     | `spi_sclk` |
| `[1]`     | in     | `spi_cs_n` |
| `[2]`     | in     | `spi_mosi` |
| `[3]`     | out    | `spi_miso` |
| `[19:4]`  | out    | observation (8-bit heartbeat + busy/done) |
| `input[0]`| in     | spare (reserved external strobe) |
| `analog[59:0]` | —  | pass-through, unconnected at core level |

Frame (SPI mode 0, MSB-first): `CMD` byte, then `ADDR[15:8]`, `ADDR[7:0]`, then
streamed `DATA` (internal address auto-increments). Commands: `0x01 WRITE`,
`0x02 READ`, `0x03 START`, `0x04 STATUS`. Address map (CTRL/STATUS/SEQ_LEN/
HEAD_DIM CSRs + Q/K/V/OUT tensor streaming regions) is in the loader header. The
loader, byte fabric, and START→busy/done handshake are **real and exercised by
the smoke test**; the wide Q/K/V/OUT streaming into the actual macro buffers is
`TODO` (wired as the macros land).

## GF180 multi-macro → padring build plan

1. **Per-macro harden (LibreLane Classic).** Run each `librelane/<macro>.yaml`
   through the Classic flow → `GDS / LEF / .lib / .nl.v` views. See
   [`docs/build.md`](docs/build.md).
2. **Per-macro verify (cocotb).** RTL sim, then gate-level sim against the gf180
   stdcell + primitives verilog (`test-*-gl` pattern from the template Makefile).
3. **Integrate into the workshop padring.** Drop `rtl/chip_core.sv` into the
   padring fork (below), merge each hardened macro into `librelane/config.yaml`'s
   `MACROS:` dict + `PDN_MACRO_CONNECTIONS`, then run
   `SLOT=workshop make librelane` (Chip flow + Magic DRC + Netgen LVS).

### Padring fork (integration target)

**`Mauricio-xx/chipathon-2026-gf180mcu-padring`** — the Chipathon 2026 workshop
padring fork of `wafer-space/gf180mcu-project-template`. It adds a native
LibreLane **`workshop`** slot (a port of Juan Moya's `padring_gf180`) with
`chip_id` + `wafer.space` logo macros and the workshop pad ring.

Integration contract (from the fork's README, "Use the workshop slot for your
own RTL"):

- **Workshop slot pad budget:** `NUM_INPUT_PADS=1`, `NUM_BIDIR_PADS=20`,
  `NUM_ANALOG_PADS=60`, plus `clk`, `rst_n`, and 4/4 DVDD/DVSS. Die 2935×2935 µm.
- **Plug-in point:** replace the fork's `src/chip_core.sv` with **our**
  `rtl/chip_core.sv`, **keeping the exact port list** (the pad interface —
  `input_in/pu/pd`, `bidir_in/out/oe/cs/sl/ie/pu/pd`, `analog`, parameterized by
  `NUM_INPUT_PADS/NUM_BIDIR_PADS/NUM_ANALOG_PADS`). Ours is copied verbatim from
  the fork's `chip_core.sv` / the template `chip_core_multi.sv`, so it drops in.
- **Add our sources + macros:** add `rtl/lambda_acu.sv`, `rtl/spi_loader.sv`, and
  the hardened block netlists to the fork's `librelane/config.yaml`
  (`VERILOG_FILES` + `MACROS:`), then `SLOT=workshop make librelane`.
- The padring itself stays fixed; only `chip_core` + the macro list change. Full
  step-by-step in [`docs/build.md`](docs/build.md).

(An equivalent fork `Jekk1213/chipathon-2026-gf180mcu-padring` exists; the
`Mauricio-xx` one is the more recently updated and is what the GF180 LibreLane
example notebooks target.)

## What's still needed

To turn this skeleton into a real, tapeout-ready submission:

1. **GF180 signoff polish + real KVE SRAM** — Stage 1 + 2 hardened all seven
   compute blocks on GF180 with a passing gate-level end-to-end covering the full
   Q·Kᵀ→softmax→P·V datapath (see [`docs/gf180_gls_report.md`](docs/gf180_gls_report.md));
   setup/hold/DRC/LVS/antenna are clean. Remaining: close the **slow-corner
   (`ss`) max-transition** violations (driver up-sizing / multi-corner slew
   repair) on the fp16 blocks that still show them, and **integrate the
   `gf180mcu_fd_ip_sram` hard macro** into `kv_cache_engine` (today it hardens
   with flip-flop register-array storage at the `SRAM_DEPTH=2` proxy — not real
   KV capacity). This SRAM macro is the one remaining honest hole in the
   gate-level datapath coverage.
3. **Padring-fork integration + full datapath wiring** — clone
   `Mauricio-xx/chipathon-2026-gf180mcu-padring`, drop in `rtl/chip_core.sv`,
   merge the macros into `librelane/config.yaml`, and complete `lambda_acu.sv`:
   wire the SPI Q/K/V/OUT streaming buffers to the real macro stream ports and
   replace the placeholder sequencer with the real block-chain FSM. Then
   `SLOT=workshop make librelane` to GDS.

## Credits & license

RTL blocks are LonghornSilicon designs (see `rtl/blocks/PROVENANCE.md`). Padring
and pad layout are Apache-2.0 (wafer.space template + Juan Moya's `padring_gf180`,
via the chipathon-2026 padring fork). This repo is intended Apache-2.0 to match.
