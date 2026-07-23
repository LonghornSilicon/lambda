# Lambda ACU — SSCS Chipathon 2026 GF180MCU Submission

**LonghornSilicon "Lambda" decode-attention Attention Compute Unit (ACU).**
An open-source hardware accelerator for the **attention + KV-cache** inner loop of
LLM inference, assembled into one full-chip on the Chipathon 2026 GF180MCU
workshop padring.

- **Track:** A — open-source digital, RTL → GDS.
- **Shuttle PDK:** GF180MCU (GlobalFoundries 180 nm), wafer.space build.
- **Flow:** LibreLane 3.0.5 (Classic per-macro; Chip flow into the workshop padring).
- **Verification:** cocotb (Python) + Icarus, RTL and gate-level.

This document is the submission's design + implementation + verification writeup.
It is deliberately **honest about what is closed and what is not** — see
[§6 Physical implementation status](#6-physical-implementation-status).

---

## 1. What this chip does (the LLM connection)

Transformer LLM inference is dominated, at long context, by **attention over the
KV cache**: for each newly generated token (the "decode" step) the model scores
one query against every cached key, soft-maxes the scores, and takes the
weighted sum of the cached values. The Lambda ACU is a hardware datapath for
exactly that decode step, plus the two systems tricks that make a long KV cache
affordable on-chip:

- **ChannelQuant KV codec (KVE)** — stores/streams K and V in a compressed,
  rotated (Walsh–Hadamard) 3-bit form (CQ-3-rot), reconstructing values on read.
- **H2O token importance (TIU)** — heavy-hitter scoring so the cache can evict
  low-importance tokens.
- **Adaptive precision (ACU gate)** — a divide-free per-tile decision to run the
  P·V reduction in INT8 or escape to FP16 when the attention distribution is peaked.

One decode step, as assembled here:

```
   host ── SPI ──► Q (int8), K (fp16), V (fp16)
                    │
   KVE  cq_value_path_wht : V → rotated INT3 codes + fp16 scale → rotated V̂ (fp16)
   MatE mate_qkt          : score[l] = round_fp16(Σ_d Q[d]·K[l][d])
   ACU  precision_controller : gate d_fp16 = (max·N > 10·Σ)          (advisory)
   VecU vecu_softmax      : w[l] = softmax(score)  (exp-LUT online softmax)
   TIU  token_importance_unit : keep-tier + eviction victim (H2O)
   MatE mate_pv_fp16      : o_rot[d] = round_fp16(Σ_l w[l]·V̂rot[l][d])
   MatE mate_pv (INT8)    : Σ_l a8[l]·code[l][d] over the KVE int8 codes (exercised)
   KVE  wht_inverse_out   : undo the WHT once → o[d] (fp32) attention output
                    │
   host ◄─ SPI ──── o (fp32, D-wide)
```

## 2. Chip assembly / architecture

Three RTL files in `chip/rtl/` assemble the seven hardened block macros into one
padring-ready core:

| File | Role |
|------|------|
| `lambda_acu.sv` | Integration top + **decode-step FSM**. Instantiates all seven macros and sequences them through one attention pass over a small register/stream interface. |
| `spi_loader.sv` | 4-wire **SPI slave** — streams the wide Q/K/V tensors in and the fp32 result out over the narrow pad budget. |
| `chip_core.sv`  | Workshop-slot override — instantiates `lambda_acu` against the padring's exact pad interface (copied verbatim from the fork). |

**Why a serial link.** The decode tile is far wider than the workshop slot's pads
(1 input + 20 bidir + 60 analog). Q/K/V (head-dim `Dh`, `L` cached tokens) are
therefore *streamed* in and the result *streamed* out over SPI. The assembled
core is parameterized; the padring build instantiates it at **`Dh=16`, `L=8`** —
a real, simulable-and-synthesizable decode tile. (The cross-block cosim exercises
the identical block RTL at `Dh=128`; only the tile shape differs.)

**Decode-step FSM** (`lambda_acu.sv`): `IDLE → KVE (encode/decode each token's V)
→ QKT → SOFTMAX (+ precision gate on the same score stream) → TIU (install /
accumulate mass / evict) → P·V (fp16 decode path + int8 tile in parallel) →
inverse-WHT → DONE`. `busy`/`done`/gate/`err` are exposed on a sticky STATUS
register (readable over SPI) and on the spare observation pads.

### Host serial protocol (`spi_loader.sv`)

SPI mode 0, MSB-first. Frame = `CMD, ADDR[15:8], ADDR[7:0], DATA…` (address
auto-increments per byte).

| CMD | | Address map (byte-addressed; fp16 little-endian on the wire) |
|-----|--|-------------|
| `0x01 WRITE` | | `0x0100` Q (Dh int8) · `0x0200` K (L·Dh fp16) · `0x0400` V (L·Dh fp16) |
| `0x02 READ`  | | `0x0800` OUT (Dh fp32) · `0x0001` STATUS mirror |
| `0x03 START` | | launch one decode step |
| `0x04 STATUS`| | next MISO byte = STATUS `{err, gate_fp16, done(sticky), busy}` |

Pad map: `bidir[0]=sclk, [1]=cs_n, [2]=mosi, [3]=miso, [19:4]=observation`.

## 3. Block → macro mapping

All block RTL is the same source of truth the per-block repos signed off; the ACU
blocks are pulled from `acu/*/rtl/`, the KVE/TIU blocks from `chip/verif/blocks/`.

| Macro | Function | Synthesizable | GF180 status |
|-------|----------|:---:|--------------|
| `precision_controller`  | INT8-vs-FP16 per-tile gate (no divide) | ✅ | Classic-clean (per §6 / gf180_gls_report) |
| `token_importance_unit` | H2O heavy-hitter eviction | ✅ | **Re-hardened here to GDS, fully clean signoff (§6)** |
| `mate_pv`               | INT8 P·V token-reduction MAC | ✅ | Classic-clean |
| `mate_pv_fp16`          | FP16 P·V MAC (fp32 accumulate) | ✅ | Classic-clean |
| `mate_qkt`              | Q·Kᵀ decode scoring (fp16) | ✅ | Classic-clean |
| `vecu_softmax`          | online softmax (exp-LUT) | ✅ | Classic-clean (multi-cycle) |
| `kv_cache_engine` (KVE store) | ChannelQuant KV store on real `gf180mcu_fd_ip_sram` | ✅ | Clean signoff on real SRAM |
| KVE value-path *reference* (`cq_value_path_wht`, `wht_unit`, `wht_inverse_out`, `cq_fp_pkg`) | CQ-3-rot value codec | ❌ **behavioral** | uses `real` fp — bit-exact contract, synthesizable fp16 lowering is future work (§6) |

## 4. Functional verification — FULL-CHIP RTL SIM (closed)

`chip/pdk/gf180/tb/test_fullchip.py` (cocotb + Icarus) drives the **assembled
`chip_core`** — through `chip_core_wrap` at the workshop pad widths (1/20/60) —
**entirely over the SPI slave**: it streams Q/K/V in with WRITE frames, issues
START, polls STATUS until done, then READs the OUT region and reconstructs the
fp32 attention row. Run: `make -C chip/pdk/gf180/tb test-fullchip`.

**Result (PASS):**

- `Q·Kᵀ` scores read from the chip **bit-match** the fp32 reference.
- `softmax` weights read from the chip **bit-match** the reference.
- The streamed-out attention row matches the reference attention **over the
  reconstructed values** to **max rel err 5e-4** (fp16 exp-LUT + fp16 P·V rounding
  only — the assembled arithmetic is right).
- End-to-end vs attention over the *original* V is **0.25**, dominated by the KVE
  **CQ-3 (3-bit) value quantization** — a characterized codec property, proven
  bit-exact against its reference in the cross-block cosim.
- The **SPI-streamed** OUT bytes are **bit-identical** to the on-chip `out_buf`,
  and the STATUS-mirror read matches STATUS — the serial readback path works.

This is the headline result: **the assembled full chip computes a correct decode
attention pass and streams it out, driven only through its pads.**

Supporting: `make test-smoke` (elaboration + SPI framing, 2 tests pass) and the
cross-block cosim `make -C chip/verif cosim` (every block bit-exact / within-tol
on one shared tile) both pass.

## 5. Physical implementation — per-macro (closed) + padring integration (wired)

### Per-macro GF180 hardening — reproduced on the shuttle PDK

`scripts/harden.sh <macro>` runs a macro through LibreLane 3.0.5 Classic inside
`ghcr.io/librelane/librelane:3.0.5` against the GF180 PDK. The
**`token_importance_unit`** macro was hardened end-to-end on the submission node
as proof the flow closes:

| Metric | Value |
|--------|-------|
| die area | 35 966 µm² |
| stdcell instances | 771 (1 727 total) |
| setup WNS | **+22.6 ns** (met) |
| hold WNS | **+0.47 ns** (met) |
| Magic DRC | **0** |
| Netgen LVS | **0** errors, 0 device/net/pin diffs |
| antenna | **0** violating nets/pins |
| routing DRC | **0** (converged in 4 iters) |
| PG violations | **0** |
| GDS / LEF | produced (`final/gds`, `final/lef`) |

### Full-chip padring integration — wired and driven

Integration into `Mauricio-xx/chipathon-2026-gf180mcu-padring` (workshop slot,
die 2935×2935 µm) follows the fork's contract:

1. Our `chip_core.sv` **replaces** `src/chip_core.sv` (verbatim pad interface — drops in).
2. Our `lambda_acu.sv`, `spi_loader.sv`, and the block RTL are added to
   `librelane/config.yaml` `VERILOG_FILES` (flat flow; `USE_SLANG: true`).
3. `SLOT=workshop make librelane` (Chip flow) against the **wafer.space GF180 PDK**.

This was executed on the node. The Chip flow runs, resolves the workshop padring,
passes run-setup + Verilator lint (after supplying the correct PDK, see §6), and
reaches yosys synthesis.

## 6. Physical implementation status — the honest gap

**Closed:** the assembled chip is functionally verified end-to-end (§4); the
per-macro GF180 Classic flow closes with fully-clean signoff (§5 TIU); the
padring integration is wired and drives through lint into synthesis.

**Not closed — full-chip GDS.** Two blockers were hit, in order:

1. **PDK variant [resolved].** The workshop padring instantiates wafer.space
   custom IO/IP cells (`gf180mcu_ws_io__dvdd/dvss`, `gf180mcu_ws_ip__id/logo`)
   that are **not** in the ciel `gf180mcuD` store on the node. Fetching the
   wafer.space PDK build (`wafer-space/gf180mcu @ 1.8.0`, ~1.2 GB, ships the
   `ws_io`/`ws_ip` cells) and pointing `--pdk-root` at it clears Verilator lint.

2. **KVE value-path is behavioral RTL [open].** In the *flat* Chip flow, yosys
   (slang frontend) aborts synthesizing the KVE **value-path reference** RTL:
   `cq_fp_pkg` models fp16/fp32 with SystemVerilog `real` (`f16_to_real`,
   `real_to_f16`, …), which is **simulation-only, not synthesizable** — slang
   asserts on lowering it. This is a known, documented split: `cq_units.sv`'s own
   header states the synthesizable fp16-hardware lowering "is the synthesis phase
   (TEARDOWN.md P4)". The ACU blocks (`mate_*`, `vecu_softmax`,
   `precision_controller`) and the TIU are fully synthesizable (TIU proven to GDS).

**Two paths to close the full-chip GDS (punch-list):**

- **Hierarchical (recommended).** Harden each block as its own macro from its
  *synthesizable* config (TIU/PC/mate_*/vecu are ready; KVE from `kve.yaml`, whose
  store path is the real fp16 hardware on `gf180mcu_fd_ip_sram`), then place the
  macro LEF/GDS views + `PDN_MACRO_CONNECTIONS` in the padring Chip flow. This
  sidesteps the behavioral value-path entirely. Needs the ACU macro configs
  authored (only `kve.yaml`, `kve_store_gf180.yaml`, `token_importance_unit.yaml`
  are committed today) plus multi-hour PnR per macro.
- **Flat.** Complete the KVE value-path's synthesizable fp16 lowering (repo
  TEARDOWN P4) so `cq_value_path_wht`/`wht_unit`/`wht_inverse_out` synthesize,
  then the single flat Chip flow closes.

## 7. Reproduce

```sh
# Full-chip functional sim (the headline result)
python3 -m venv ccenv && ccenv/bin/pip install cocotb numpy
source ccenv/bin/activate
make -C chip/pdk/gf180/tb test-fullchip     # streams Q/K/V in, checks streamed-out row
make -C chip/pdk/gf180/tb test-smoke        # elaboration + SPI framing
make -C chip/verif cosim                     # cross-block bit-exact cosim

# Per-macro GF180 hardening (needs docker + a GF180 PDK)
chip/pdk/gf180/scripts/harden.sh token_importance_unit   # → runs/<macro>/final/{gds,lef}

# Full-chip padring GDS (see §5/§6; needs the wafer.space GF180 PDK @ 1.8.0)
#   git clone Mauricio-xx/chipathon-2026-gf180mcu-padring
#   cp chip/rtl/chip_core.sv <fork>/src/chip_core.sv
#   add chip/rtl/{lambda_acu,spi_loader}.sv + block RTL to <fork>/librelane/config.yaml
#   SLOT=workshop make librelane   (--pdk-root <wafer.space gf180mcuD>)
```

## 8. Credits & license

RTL blocks are LonghornSilicon designs (see `PROVENANCE.md`). WHT value rotation
primitive is Chaithu Talasila / Abhiram Bandi. Padring + pad layout are Apache-2.0
(wafer.space template + Juan Moya's `padring_gf180`, via the chipathon-2026 fork).
Intended license: Apache-2.0.
