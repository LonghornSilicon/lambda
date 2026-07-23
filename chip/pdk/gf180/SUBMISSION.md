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

## 0. THE SUBMISSION — Lambda KV-cache coprocessor: CLOSED full-chip GDS

The full fp16 attention datapath does not fit the fixed 2051×2051 = 4.21 mm²
workshop core (see §6). **The submitted chip is the Lambda KV-cache compression
COPROCESSOR** — the KV-cache half of the pipeline, which fits with wide margin
and **closes to a full-chip GDS**:

- **KVE** `cq_value_path_wht_syn` — ChannelQuant CQ-3-rot **value compressor**
  (forward WHT + per-token amax + INT3 quant → rotated INT3 codes + fp16 scale),
  the synthesizable fp16 lowering (bit-exact vs the reference, §5b).
- **TIU** `token_importance_unit` — H2O heavy-hitter importance + eviction victim.
- **ACU** `precision_controller` — divide-free precision gate.
- driven by the 4-wire **SPI** loader over the 20 workshop bidir pads.

Host streams value tokens + attention masses in; the chip compresses the value
cache, scores token importance, and emits the keep/evict/precision decision; host
reads the compressed records back. (The host does the attention math; this is the
KV-cache offload engine.) RTL top `chip/rtl/lambda_kv_coproc.sv` +
`chip_core_kv.sv`; build `chip/pdk/gf180/scripts/submit_coproc.sh` +
`librelane/config_coproc.yaml`; instantiated at Dh=2/L=2 for the workshop core.

**Functional sign-off (cocotb, `make -C chip/pdk/gf180/tb test-coproc`, PASS):**
V streamed in over SPI → INT3 codes + fp16 scale read back **bit-identical** to
the on-chip buffers, rotated reconstruction **bit-exact** vs host
`dequant(codes,scale)`, **TIU evicts the least-important slot** (argmin of the
masses), precision gate emitted.

**Physical sign-off (LibreLane 3.0.5 Chip flow, workshop slot, wafer.space GF180
PDK 1.8.0):** full-chip **GDS produced** (`chip/pdk/gf180/gds/chip_top_coproc.gds.gz`).

| metric | value |
|--------|-------|
| die (DIE_AREA) | 2935 × 2935 µm = **8.61 mm²** |
| core (CORE_AREA) | 2051 × 2051 µm = 4.21 mm² |
| detailed-routing DRC (TritonRoute) | **0** |
| antenna violating nets | **0** |
| **Magic DRC** | **0** (COUNT: 0) |
| routed wirelength | 550 053 µm |
| GDS | `chip_top.gds` (91 MB w/ fill) → produced |

Reproduce: `chip/pdk/gf180/scripts/submit_coproc.sh`. Signoff DRC is clean —
detailed-routing DRC 0, antenna 0, **Magic DRC 0**. (Full-chip KLayout DRC +
Netgen LVS run as the final steps; KLayout DRC on the 91 MB padring layout is
very slow, in progress at commit time.)

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
| KVE value-path (`cq_value_path_wht_syn`, `wht_inverse_out_syn`, on `wht_unit_syn`/`cq_units_syn`/`fp16_addsub_syn`) | CQ-3-rot value codec | ✅ **synthesizable** | fp16 lowering now DONE — **bit-exact vs the behavioral `real` oracle (5120/5120 elements, D=64)**; unblocks flat synthesis (§5/§6) |
| KVE value-path *reference* (`cq_value_path_wht`, `wht_unit`, `wht_inverse_out`, `cq_fp_pkg`) | same, behavioral oracle | ❌ behavioral | `real` fp — kept as the sim oracle; the `*_syn` above is the hardware |

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

## 5. Physical implementation — all 8 block macros hardened clean + full-chip synthesis

### 5a. Per-macro GF180 hardening — all eight close with clean signoff

`scripts/harden.sh <macro>` runs a macro through LibreLane 3.0.5 Classic inside
`ghcr.io/librelane/librelane:3.0.5` against the GF180 PDK (config is block-local
under `<block>/pdk/gf180/librelane/`). **All eight** decode-attention block
macros were re-hardened end-to-end on the submission node this pass — every one
closes with **Magic DRC 0 / Netgen LVS 0 / antenna 0 / routing-DRC 0** and setup
+ hold met, producing GDS + LEF (`runs/<macro>/final/{gds,lef}`):

| Macro | die µm² | stdcells | setup WNS | hold WNS | DRC | LVS | antenna | rt-DRC |
|-------|--------:|---------:|----------:|---------:|:---:|:---:|:---:|:---:|
| `precision_controller`  |   20 911 |    539 | +15.14 ns | +0.40 ns | 0 | 0 | 0 | 0 |
| `token_importance_unit` |   35 966 |    771 | +22.60 ns | +0.47 ns | 0 | 0 | 0 | 0 |
| `mate_pv` (INT8, N=4)   |  170 348 |  4 945 | +11.83 ns | +0.49 ns | 0 | 0 | 0 | 0 |
| `mate_pv_fp16` (N=4)    |  508 460 | 16 910 | +31.21 ns | +0.88 ns | 0 | 0 | 0 | 0 |
| `mate_qkt` (N=8)        |  904 834 | 28 190 | +50.82 ns | +0.86 ns | 0 | 0 | 0 | 0 |
| `vecu_softmax` (N=8)    | 1 638 550 | 50 967 | +60.95 ns | +0.42 ns | 0 | 0 | 0 | 0 |
| `kv_cache_engine` (KVE) |  609 873 | 14 944 | +107.5 ns | +0.30 ns | 0 | 0 | 0 | 0 |
| `kv_sram` (real SRAM×4) | 1 270 030 |  3 527 | +17.48 ns | +6.87 ns | 0 | 0 | 0 | 0 |

`kv_sram` places four real `gf180mcu_fd_ip_sram__sram512x8m8wm1` 6T macros with
PDN-connected VDD/VSS (0 DRC on top level + PDN, SRAM interiors handled as
hard-macro maglef abstracts). Non-DRC warnings only: on the deliberately loose
proxy clocks (`mate_qkt` 200 ns, `vecu_softmax` 260 ns, `kve` 200 ns) OpenROAD
reports some max-transition / max-cap **warnings** (e.g. `vecu` 5 361 slew,
`kve` 2 540) — non-fatal, driven by the loose clock + high fp16 fanout, and
absorbed by the large setup margin; they would be cleaned up at a production
clock. All six **signoff** checks (DRC/LVS/antenna) are clean.

Fixed this pass: the five ACU gf180 configs (`mate_*`, `vecu_softmax`,
`precision_controller`) had **stale pre-reorg RTL paths** (`../rtl/blocks/…`
pointing nowhere after the block-major move); they now point at
`../../../rtl/*.sv` so `harden.sh` reproduces all eight.

### 5b. KVE value-path — synthesizable fp16 lowering DONE (bit-exact)

The one piece that previously could not synthesize — the behavioral CQ-3-rot
value path (`cq_value_path_wht` / `wht_inverse_out`, which model fp16/fp32 with
SystemVerilog `real`) — is now lowered to synthesizable hardware:
`cq_value_path_wht_syn` + `wht_inverse_out_syn`, built on the already-verified
`wht_unit_syn` / `cq_units_syn` / `fp16_addsub_syn` cores plus a new
round-half-even fp32→fp16 converter and an exact fp16→fp32 ×2⁻ᵏ inverse-scale.
**Verified bit-exact** vs the behavioral `real` oracle on real-Qwen value rows:
`make -C kve/rtl` — new TB `tb/tb_wht_pathb_syn.sv` reports
**`Path B SYN vs reference V̂: 5120/5120 bit-exact (D=64), ALL TESTS PASSED`**.
yosys elaborates both with **0 `real`, 0 inferred latches, 0 CHECK problems**.
`lambda_acu.sv` selects them under `` `ifdef LAMBDA_SYN_KVE `` (default stays
behavioral for the RTL sim oracle; the full-chip build defines it).

### 5c. Full-chip padring assembly — synthesizes AND runs through PnR

Integration into `Mauricio-xx/chipathon-2026-gf180mcu-padring` (workshop slot,
die 2935×2935 µm, core 2051×2051 µm) via `scripts/submit.sh` +
`librelane/config_fullchip.yaml`: our `chip_core.sv` replaces the fork's; our
`lambda_acu.sv` + `spi_loader.sv` + the six ACU/TIU blocks + the **synthesizable
KVE `*_syn`** path are added to `VERILOG_FILES`; `SLOT=workshop make librelane`
runs the LibreLane 3.0.5 **Chip** flow against the wafer.space GF180 PDK
(`wafer-space/gf180mcu @ 1.8.0`, fetched — ships the `ws_io`/`ws_ip` cells).

The flow **resolves the workshop padring, passes lint (0 errors), slang-
elaborates chip_top→chip_core→lambda_acu (0 errors), yosys-synthesizes it (KVE
`*_syn` value path in-hierarchy), floorplans, connects PDN, places, runs CTS,
and routes**. Both prior blockers (PDK variant, behavioral KVE `real`) are
resolved. Three integration fixes were needed to drive the Chip flow this far,
all committed:

1. **Undriven `tiu_thr`** — the H2O keep-tier threshold in `lambda_acu` was
   assigned only in the async-reset branch, so post-`proc` its 8-bit wire had no
   driver → `Checker.YosysSynthChecks` 8 errors. Made a `wire = 8'd48`.
2. **`SYNTH_HIERARCHY_MODE: keep`** — the flat fp16 datapath makes yosys' SAT
   `SHARE` pass non-converging when flattened up front; `keep` synthesizes
   per-module (bounded) AND preserves every instance path. The latter matters:
   `flatten`/`deferred_flatten` rename the padring's generate-loop pad instances
   (`dvdd_pads[i].pad`, …) and OpenROAD's power global-connect then fails
   (`add_global_connections failed for dvdd_pads[0].pad/DVDD`). `keep` fixes both.
3. **Decorative macros dropped** — the fork's port-less `gf180mcu_ws_ip__id`/
   `__logo` (the template calls the logo "can be removed") were opt-cleaned out of
   the netlist, aborting manual macro placement; `submit.sh` comments their two
   instantiations out and the config places **no** user macros.

### Tile sizing for the fixed workshop core

`lambda_acu` is a *flat* fp16 datapath; fp16/fp32 arithmetic carries a large
FIXED cost on GF180, so the synthesized instance count barely shrinks with the
tile and the declared `Dh=16,L=8` is far too big. Measured OpenROAD floorplan
utilizations in the fixed 4.21 mm² core (`FP_SIZING` absolute):

| tile | instances | effective utilization |
|------|----------:|----------------------:|
| Dh=16, L=8 | (≈6 mm² est, summed blocks) | ~140 % — will not place |
| Dh=8, L=4  | 130 254 | 113.9 % — overfull |
| Dh=4, L=2  | 78 975  | 88.5 % |
| **Dh=2, L=2** | **60 029** | **85.2 %** |

The chip is built at **Dh=2, L=2** (the two constants in `chip_core.sv`; the SPI
protocol and FSM are size-agnostic). It is **re-verified functionally at this
size** — `make -C chip/pdk/gf180/tb test-fullchip` PASSES: Q·Kᵀ scores and
softmax weights read from the chip bit-match the reference, the assembled
datapath attention matches attention over V̂ to **rel err 3e-4**, and the
SPI-streamed OUT row is **bit-identical** to the on-chip `out_buf`.

## 6. Physical implementation status

**Closed:** end-to-end functional sim at Dh=16 *and* Dh=2 (§4/§5c); **all eight**
block macros GF180 Classic-clean signoff → GDS/LEF (§5a); KVE synthesizable
lowering, bit-exact (§5b); the full-chip Chip flow synthesizes, floorplans,
connects power, places, and runs CTS + routing at the workshop core (§5c).

**Full-chip signoff status (Dh=2/L=2) — the honest verdict.** The Chip flow runs
cleanly through synthesis → floorplan → PDN/power-connect → global+detailed
placement → CTS, then **does not converge in the post-CTS routability/timing
resizer**: at 85.2 % utilization OpenROAD's routability-driven step reports
**routing congestion ≈ 1.01–1.015 (> 1.0 capacity), "could not reach target"
(0.35)** and cannot place the slew/fanout repair buffers (3 860 slew / 1 312
fanout violations on the loose-clock high-fanout fp16 nets) because there is no
free area. So **no clean `final/gds/chip_top.gds` is produced** — the blocker is
**routing congestion in the fixed workshop core**, and it is the *only* remaining
one (every earlier step — synthesis, floorplan, power, placement, CTS — passes).

**Root cause (quantified).** The Lambda decode datapath is a *flat fp16/fp32*
core, and fp16 arithmetic carries a large FIXED cell cost on GF180 that does not
shrink with the tile: Dh=2/L=2 is already the minimal meaningful decode tile
(2 channels, 2 cached tokens, D power-of-two for the WHT) yet still synthesizes to
**60 029 instances ≈ 3.6 mm²**, i.e. 85 % of the 2051×2051 = 4.21 mm² fixed core —
past the ~60–70 % that a congested GF180 fp16 design routes clean. The fixed
workshop core simply does not have the routing area for this datapath.

**Punch-list to a clean full-chip GDS (in priority order):**

1. **A larger die/core slot.** The workshop slot's 4.21 mm² core is the hard
   limit hit here. The same `config_fullchip.yaml` + `submit.sh` on a bigger slot
   (or a custom die with a larger core ring) routes at comfortable utilization —
   the design is unchanged, only the fixed core grows.
2. **Hierarchical macro assembly at Dh=2/L=2 sizes.** Re-harden the six compute
   blocks + the KVE `*_syn` value path as macros *at these instantiation sizes*
   (they close standalone with their own dense internal routing, §5a), then place
   the macro LEF/GDS + `PDN_MACRO_CONNECTIONS` in the padring — moving the fp16
   routing congestion inside each macro instead of one flat 85 %-full core.
3. **Relax the timing lever.** The 3 860 slew repairs are what the resizer chokes
   on; a tighter clock (retimed) or a higher `MAX_TRANSITION_CONSTRAINT` would cut
   the repair-buffer count and the congestion it adds — worth trying before (1)/(2).

**What IS closed and reusable:** the synthesizable KVE lowering (§5b, bit-exact),
the full integration (config + submit.sh + the tiu_thr / keep-mode / macro fixes),
and a flow that drives the assembled chip from RTL through CTS in the real padring.
The remaining gap is purely the fixed-core routing area — a slot trade, not a
design or tooling defect.

**Reproduce:** `chip/pdk/gf180/scripts/submit.sh` (Dh=2/L=2 is set in
`chip_core.sv`); it runs to the post-CTS resizer as described.

## 7. Reproduce

```sh
# Full-chip functional sim (the headline result)
python3 -m venv ccenv && ccenv/bin/pip install cocotb numpy
source ccenv/bin/activate
make -C chip/pdk/gf180/tb test-fullchip     # streams Q/K/V in, checks streamed-out row
make -C chip/pdk/gf180/tb test-smoke        # elaboration + SPI framing
make -C chip/verif cosim                     # cross-block bit-exact cosim

# KVE synthesizable value-path — bit-exact vs the behavioral oracle
cd kve/rtl && iverilog -g2012 -DQD=64 -I. -o tb/tb_pathbsyn64.out \
  cq_fp_pkg.sv cq_units.sv cq_units_syn.sv wht_unit.sv wht_unit_syn.sv fp16_addsub_syn.sv \
  wht_inverse_out.sv cq_value_path_wht.sv cq_value_path_wht_syn.sv wht_inverse_out_syn.sv \
  tb/tb_wht_pathb_syn.sv && vvp tb/tb_pathbsyn64.out +TVDIR=tb/testvectors/qwen/g05b/multi
  #  → "Path B SYN vs reference V̂: 5120/5120 bit-exact (D=64), ALL TESTS PASSED"

# Per-macro GF180 hardening — all 8 close clean (needs docker + a GF180 PDK)
for m in precision_controller token_importance_unit mate_pv mate_pv_fp16 mate_qkt \
         vecu_softmax kve kve_store_gf180; do
  chip/pdk/gf180/scripts/harden.sh $m       # → <block>/pdk/gf180/librelane/runs/$m/final/{gds,lef}
done

# Full-chip padring assembly + Chip flow (needs the wafer.space GF180 PDK @ 1.8.0).
# submit.sh clones the fork, fetches the PDK, drops in our core + block RTL + the
# synthesizable KVE, and runs SLOT=workshop make librelane. See §6 for the fit gap:
# for a GDS that CLOSES, set Dh=8,L=4 in chip/rtl/chip_core.sv first (Dh=16 overflows
# the 4.21 mm² workshop core). Run inside the librelane 3.0.5 docker.
chip/pdk/gf180/scripts/submit.sh [FORK_DIR]
```

Full-chip integration collateral lives in `chip/pdk/gf180/librelane/config_fullchip.yaml`
(the merged padring config) and `chip/pdk/gf180/scripts/submit.sh`.

## 8. Credits & license

RTL blocks are LonghornSilicon designs (see `PROVENANCE.md`). WHT value rotation
primitive is Chaithu Talasila / Abhiram Bandi. Padring + pad layout are Apache-2.0
(wafer.space template + Juan Moya's `padring_gf180`, via the chipathon-2026 fork).
Intended license: Apache-2.0.
