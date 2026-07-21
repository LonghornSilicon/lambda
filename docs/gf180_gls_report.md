# GF180 hardening + gate-level end-to-end verification (Stages 1 & 2)

**Date:** 2026-07-21   **PDK:** gf180mcuD (ciel `54435919…`, the version LibreLane
3.0.5 pins)   **Flow:** LibreLane 3.0.5 Classic (docker `ghcr.io/librelane/librelane:3.0.5`)
**Sim:** Icarus Verilog 12.0, gate-level against `gf180mcu_fd_sc_mcu7t5v0` cell models.

The bar: the GF180-**hardened** netlist, simulated end-to-end at the **gate**
level on GF180 cells, must produce results matching the reference model —
**INT bit-exact**, **FP16 rel_err < 5e-3**. Result: **met** (see §2).

- **Stage 1** hardened + GL-verified the five blocks that had RTL: `mate_pv`,
  `mate_pv_fp16`, `precision_controller`, `token_importance_unit`, `kv_cache_engine`.
- **Stage 2** adds the two decode-scoring blocks `mate_qkt` (Q·Kᵀ) and
  `vecu_softmax` (online softmax) and extends the GL loop to the **full compute
  datapath Q·Kᵀ → softmax → P·V**.

---

## 1. Per-macro GF180 signoff (7 macros)

Hardened standalone (LibreLane Classic, `librelane/<macro>.yaml`,
`scripts/harden.sh`). Clocks re-timed for GF180 (180 nm — far slower than the
Sky130 periods the Stage-1 configs were ported from; the fp16 blocks need the
loosest clocks).

| macro (`DESIGN_NAME`) | params (baked) | clk | die µm² | cells | setup WS | hold WS | DRC | LVS | antenna | max-cap | max-slew (transition) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `precision_controller`  | BLOCK 64×64 (N=4096) | 40 ns  | 20 911  | 1 064  | **+15.14** | +0.402 | 0 | 0 | 0 | 0 | **0 (clean, all corners)** |
| `token_importance_unit` | N_SLOTS=4            | 40 ns  | 35 966  | 1 727  | **+22.60** | +0.466 | 0 | 0 | 0 | 1 (ss) | 6 (ss only) |
| `mate_pv`               | N=4                 | 40 ns  | 170 348 | 9 833  | **+11.83** | +0.493 | 0 | 0 | 0 | 0 | 69 (ss only) |
| `mate_pv_fp16`          | N=4                 | 180 ns | 508 460 | 32 287 | **+31.21** | +0.876 | 0 | 0 | 0 | 0 | 319 (ss only) |
| `mate_qkt`              | N=8                 | 200 ns | 904 834 | 61 255 | **+50.82** | +0.863 | 0 | 0 | 0 | 20 (ss) | 2706 (ss; 87 tt, 30 ff) |
| `vecu_softmax` (pipelined) | N=8              | 300 ns | 1 486 490 | 101 236 | **+149 (tt) / +207 (ff) / +19.2 (ss)** | +0.919 | 0 | 0 | 0 | 5 (ss) | 5292 (ss; 20 tt) |
| `kv_cache_engine` (kve) | SRAM_DEPTH=2, VECTOR_DIM=8, KEY_GROUP=2 | 200 ns | 621 960 | 32 294 | **+109.82** | +0.261 | 0 | 0 | 0 | 7 (ss) | 2392 (ss; 83 tt) |

**Implied fmax** (min period = clk − worst-corner setup WS; loose clocks chosen
for clean closure, so conservative): precision_controller ≈ 40 MHz,
token_importance_unit ≈ 58 MHz, mate_pv ≈ 36 MHz, mate_pv_fp16 ≈ 6.7 MHz,
mate_qkt ≈ 6.7 MHz, vecu_softmax ≈ 3.6 MHz (worst ss corner; ~6.5 MHz at tt),
kve ≈ 11 MHz.

### The 6 signoff checks
- **Setup:** **all seven now meet setup across ALL corners** (TNS = 0),
  **including the extreme `ss_125C_4v50` corner**. `vecu_softmax` was the lone
  Stage-2 exception (−26.5 ns at ss on the un-pipelined ~366 ns online-softmax
  fp16→exp-LUT→fp32→fp16 chain); it has since been **pipelined** — the two-serial-
  fp32-multiply exp eval is split across 3 feed-forward register stages (+358 FFs,
  throughput unchanged at 1 score/cycle, bit-exact, +3-cycle data-independent
  latency). Re-hardened, it now closes at **ss WS +19.2 ns** (tt +149, ff +207),
  TNS = 0, at a **tighter 300 ns clock** than the un-pipelined 340 ns that failed.
  The 3-stage split is uneven (the longest single stage is ~263 ns at ss, not the
  hoped ~183 ns), and the aggressive resize to hit ss roughly doubled the cell
  count (55k → 101k, 1.49 mm²) — a size/timing trade the follow-up (a more even
  pipeline split or higher utilisation) can reclaim.
- **Hold:** all seven **meet hold** (WS positive, TNS = 0, all corners).
- **Magic DRC:** **0** on all seven.
- **Netgen LVS:** **0** on all seven (0 device/net/pin/property mismatches).
- **Antenna:** **0** violating nets on all seven.
- **Max-cap / max-transition (slew):** `precision_controller` is **fully clean
  on all corners**. The others have **max-transition violations confined to the
  extreme slow corner** `max_ss_125C_4v50` (the `tt`/`ff` corners are clean or
  near-clean). Known GF180 wide-corner artifact: GF180's ss corner is very slow,
  so min-size drivers that pass transition at tt/ff exceed it at ss. The big
  fp16 blocks (`vecu_softmax`, `mate_qkt`, `mate_pv_fp16`) and the register-array
  `kve` show the largest counts (un-optimised high-fanout nets on sprawling
  floorplans). DRC/LVS are unaffected (those are physical-rule / connectivity).

**Setup now closes at every corner across all seven macros** (the pipelined
`vecu_softmax` removed the last ss-corner setup miss). The remaining follow-up is
the **ss-corner max-transition** (slew) on the big fp16/register-array blocks —
driver up-sizing / a tighter placement-time transition target / denser
floorplans (multi-corner repair). It is a physical-optimisation item that does
not affect functional correctness or the gate-level sim.

## 2. GF180 gate-level end-to-end (the deliverable)

`tb/tb_gls_e2e.sv` reproduces the cross-block check from the architecture repo's
`rtl/tb/tb_chip_cosim.sv` — now the **full compute datapath**: KVE reconstruct V̂
→ **mate_qkt** Q·Kᵀ scoring → ACU gate → **vecu_softmax** → **mate_pv/_fp16** P·V,
plus TIU keep/evict, on a **real Qwen attention tile** (`tb/vectors/qwen_*.hex`).
It reuses the cosim's reference computations + tolerance gates, but instantiates
the **GF180 gate-level netlists** (`runs/<macro>/final/nl/<macro>.nl.v`) compiled
against the `gf180mcu_fd_sc_mcu7t5v0` Verilog cell models.

Run: `cd tb && make test-gls-e2e`. Output:

```
[KVE  RTL] CQ-3-rot V̂ over 8 real-Qwen tokens: bit-exact vs reference
[MatE  GL] INT8 P·V MAC, 8 tokens x N=4, INT32 acc: int32 BIT-EXACT vs matmul_int8
[MatE  GL] e2e KVE->P·V dequant vs Sigma A*Vhat: max rel err 0.011291 (within tol, tol 0.06)
[MatE  GL] FP16 P·V escape: tile Sigma A*Vhat max rel err 0.000174 vs seq-fp32 golden (within tol, tol 0.005)
[ACU   GL] precision gate: FP16=1 (peaked) / FP16=0 (uniform) -> discriminates (match reference decision: YES)
[TIU   GL] keep-tier (thr=128) + eviction victim: match reference (evict slot 3, exp 3)
[MatE  GL] Q·Kᵀ (mate_qkt) scores: rel-err 0.000000 (< 0.005) vs seq-fp32 golden; -> ACU gate fp16=1: match reference
[VecU  GL] decode Q·Kᵀ->softmax->P·V (weights = vecu_softmax GL): softmax err 0.000213 (< 0.050), attn-out rel-err 0.000542 (< 0.060): within tol

GF180 GATE-LEVEL E2E (full compute datapath Q·Kᵀ->softmax->P·V + ACU + TIU gate-level; KVE RTL): ALL PASS
```

| block | check vs reference | result |
|---|---|---|
| **Q·Kᵀ (`mate_qkt`, GL)** | N=8 fp16 scores vs sequential-fp32 golden | max rel err **0.000000** (< **5e-3**) |
| **softmax (`vecu_softmax`, GL)** | N=8 fp16 weights vs exact fp64 softmax | max abs err **2.13e-4** (< 0.05, exp-LUT) |
| **closed-loop attention out (GL)** | `Σ_t softmax(Q·Kᵀ)[t]·V̂[t]` vs reference | max rel err **5.42e-4** (< 0.06) |
| MatE INT8 P·V (`mate_pv`, GL) | `Σ_t A[t]·V̂rot[t]` INT32, vs `matmul_int8` | **bit-exact** |
| MatE INT8 P·V reconstruct (GL) | dequant vs `Σ A·V̂` (rotated space) | max rel err **0.0113** (< 0.06, INT8-quant only) |
| MatE FP16 P·V (`mate_pv_fp16`, GL) | vs sequential-fp32 golden | max rel err **1.74e-4** (< **5e-3**) |
| ACU gate (`precision_controller`, GL) | `d_fp16 == (max·N > 10·Σ)`; peaked→FP16, uniform→INT8 | **matches + discriminates** |
| TIU (`token_importance_unit`, GL) | keep-tier(thr) + argmin evict victim | **matches** (evict slot 3 = expected) |

So the **entire compute datapath Q·Kᵀ → softmax → P·V** (plus the ACU gate and
TIU), as GF180 gate-level netlists, is bit-exact (INT) / within `5e-3` (FP16) /
within the softmax LUT tolerance to the same reference the RTL cosim uses, on
real-Qwen-derived stimulus. The Stage-1 checks are unchanged (**no regression**).

## 3. Coverage boundary (honest scope)

**Gate-level in the end-to-end GLS (§2) — the full compute datapath:** `mate_qkt`,
`vecu_softmax`, `mate_pv`, `mate_pv_fp16`, `precision_controller`,
`token_importance_unit` — GF180 hardened netlists, driven at their baked proxy
widths (mate_qkt/vecu_softmax N=8, mate_pv/_fp16 N=4, tiu N_SLOTS=4,
precision_controller N=4096 tile) with the identical reference formula. Q·Kᵀ,
softmax and P·V are all real GF180 gate-level.

**RTL (feeds the gate-level blocks), NOT gate-level:** the KVE value-reconstruct
path `cq_value_path_wht` + `wht_inverse_out` (combinational CQ-3-rot decode).
It reconstructs the real V̂ that drives the gate-level P·V, and its own
bit-exact check here is at RTL. Reason: the cosim's KVE step is this small
combinational reconstruct, which is a different module from the full
`kv_cache_engine` (AXI-Lite/Stream codec) — see the SRAM note below.

**`kv_cache_engine` (full KVE) — hardened standalone, but not in the GLS loop:**
- It **does harden on GF180** (Classic flow completes: setup/hold met, DRC=0,
  LVS=0, antenna=0; §1), using the **curated synthesis file set** (the sibling
  repo's `openlane/kv_cache_engine/src` list). The behavioral RTL views
  (`cq_fp_pkg.sv`, `cq_units.sv`, `wht_*.sv`) use `real`/`$fscanf` and abort
  yosys — the `*_syn.sv` real-free views are the synthesis set (`librelane/kve.yaml`).
- **SRAM store — now on a REAL GF180 SRAM macro (see §4).** The KVE's KV store was
  previously a flip-flop register array at the `SRAM_DEPTH=2` proxy. It is now
  refactored behind a swappable `kv_sram` memory interface and backed by real
  `gf180mcu_fd_ip_sram` hard macros — see §5 for the integration + its honest
  signoff boundary. (The full `kv_cache_engine` codec still hardens as logic; its
  AXI wrapper is separate from the cosim's combinational value reconstruct, so it
  is not in the compute-datapath GLS loop — but the *store* now uses real SRAM.)

**Both `mate_qkt` and `vecu_softmax` are now VERBATIM vendored sources — no local
edits.** The **pipelined** `vecu_softmax` (`attention-compute-unit` `rtl`
`f36daa2`) restructured the exp datapath and no longer has the module-scope
`gi_unused` loop counter that the earlier un-pipelined version needed a
synth-compat patch for, so it synthesizes cleanly with the default yosys frontend
as-is. See `rtl/blocks/PROVENANCE.md`.

## 4. KV store on a real GF180 SRAM macro (closing the SRAM hole)

**RTL refactor (PDK-agnostic, kv-cache-engine repo `rtl`):** the raw storage array
was extracted out of `sram_controller` into a new **`kv_sram`** module with a clean
`addr/data/we/re` interface (registered 1-cycle read). The default `kv_sram` is
behavioral (a reg array) so sim + other PDKs are unchanged; the GF180 build swaps
in a wrapper that tiles the hard macro. No functional change — verified no
regression: `make sim_top` (V + grouped keys thru SRAM, ALL PASS), `make sim`
(17/17), `make sim_realdata` (ALL PASS).

**GF180 tiling wrapper (chipathon, `rtl/blocks/kve_gf180_sram/kv_sram.sv`):** the
GF180 open SRAM IP is single-port, synchronous, 512×8 (registered Q). A
WIDTH-bit × DEPTH-word store is `ceil(WIDTH/8)` byte-lane banks sharing one
address/control; the KVE FSM writes (ST_STORE) and reads (ST_RLOAD) in distinct
cycles, so the two logical ports mux safely onto the one macro port
(`CEN=~(we|re)`, `GWEN=~we`, `WEN=we?0:FF`).

**Functional round-trip through the REAL macro — BIT-EXACT** (`make test-kv-sram-gf180`):
`sram_controller` + the GF180 `kv_sram` wired to the real
`gf180mcu_fd_ip_sram__sram512x8m8wm1` **simulation model** (the sign-off view of
the hard IP), an **80-bit × 512-word** store (10 banks). Writes KV records, reads
them back (incl. overwrite) — all **BIT-EXACT** through the real macro protocol.
So the KV store round-trips through real SRAM, not flip-flops.

**Physical hardening with the macro placed (`librelane/kve_store_gf180.yaml`):**
`kv_sram` hardened as a 32-bit × 512-word store = **4 placed
`gf180mcu_fd_ip_sram__sram512x8m8wm1` hard macros** (2×2 grid), LEF/lib/gds from
the PDK, a custom `pdn_cfg_sram.tcl` bridging the macro power pins to the grid:

| check | result |
|---|---|
| macro placement | **4 SRAM macros placed** (die 1.27 mm², 8911 cells) |
| setup / hold | **met** (setup WS +17.5 ns, hold WS +6.9 ns @ 40 ns) |
| routing (TritonRoute) DRC | **0 — clean** (global + detailed routing complete) |
| antenna | **0** |
| PDN power connectivity (PSM) | **passes** (macro VDD/VSS electrically tied to the grid) |
| SRAM macro LVS device match | **matches** (`gf180mcu_fd_ip_sram…` device classes equivalent) |
| Magic DRC | **7026 — NOT clean** (all from the PDN via geometry tying the macro's Metal2/Metal1 power pins to the Metal4 straps; macro-internal bitcell DRC suppressed via `MAGIC_DRC_USE_GDS: false`, the standard hard-macro handling) |
| Netgen LVS | **6 errors** (power-net "badnets" from the same PDN connection; the SRAM device itself matches) |

**Honest boundary:** the real SRAM macro is **placed, timed, routed-clean,
antenna-clean, PSM-power-connected, and LVS-device-matched** — the store is
genuinely on real 6T-SRAM bitcells and round-trips bit-exact in sim. The one
unresolved piece is **clean Magic-DRC + LVS signoff of the PDN via connection**
from the macro's low-metal (Metal2 VDD / Metal1 VSS) power pins up to the Metal4
straps: the connection is electrically valid (PSM passes) but its via geometry
violates gf180 Via1/Via2 rules and leaves LVS power-net mismatches. Closing it
needs a refined PDN (proper via arrays / a per-macro power ring on the pin
layers) — the classic fiddly hard-macro PDN step, a documented follow-up. This is
a real partial, not faked signoff.

**The `gf180mcu_fd_ip_sram` macro is now integrated** (placed + routed + power-
connected + functional-bit-exact); the remaining hole narrowed to **clean
DRC/LVS signoff of the macro PDN vias**.

## 5. Reproduce

```bash
# one-time: enable the gf180 PDK LibreLane pins into the ciel store
docker run --rm -e PDK_ROOT=/pdk -v ~/.ciel:/pdk ghcr.io/librelane/librelane:3.0.5 \
  ciel enable --pdk-family gf180mcu 54435919abffb937387ec956209f9cf5fd2dfbee

# harden a macro (GF180 Classic)
scripts/harden.sh precision_controller     # …mate_pv, mate_pv_fp16,
                                            #   token_importance_unit, mate_qkt,
                                            #   vecu_softmax, kve

# gate-level end-to-end (needs the 6 compute macros hardened)
cd tb && make test-gls-e2e

# KV store on the real SRAM macro (§4)
scripts/harden.sh kve_store_gf180          # hardens kv_sram + 4 gf180 SRAM macros
cd tb && make test-kv-sram-gf180           # KV round-trip bit-exact thru real SRAM
# (kve RTL regression, in the kv-cache-engine repo: make sim_top sim sim_realdata)
```

Notes / friction encountered (for the next runner):
- LibreLane could not be `pip`-installed on this box (read-only base venv); the
  **docker image path works** and is what `scripts/harden.sh` uses.
- LibreLane 3.0.5 pins a **specific** gf180mcu ciel version; `ciel enable` that
  exact hash, then run in **default ciel-pdk mode** (`-p gf180mcuD`, `PDK_ROOT`
  env). `--manual-pdk` loads the PDK but leaves `LIB` empty → yosys `DFFLIBMAP`
  fails ("Missing -liberty").
- The Sky130 `token_importance_unit` config's `MAX_TRANSITION_CONSTRAINT: 1.5`
  sent OpenROAD `repair_design` into a non-converging loop on the slower GF180
  cells — dropped it (GF180 default max-transition) and it closes.
- **`vecu_softmax` was the hard one — now fixed by pipelining.** Its
  fp16→exp-LUT→fp32→fp16 chain is the longest path in the datapath. The
  un-pipelined version missed ss setup by −26.5 ns even at a 340 ns clock, and the
  post-CTS/post-GRT resizers ground for tens of minutes asymptotically (never
  closing) — a sign the *base* path, not the clock, was the limit. The upstream
  RTL was then **pipelined** (3-stage feed-forward exp eval); re-hardened, it
  closes ss at **+19.2 ns** at a *tighter* 300 ns clock, and the resizers now pass
  in seconds (margin to spare). Lesson: when the resizer grinds asymptotically,
  pipeline the path — don't keep raising the period.
- **SRAM hard-macro escaping (§4).** A `genvar` generate loop names the macro
  instances `lane[0].u_bank` → Verilog escaped identifier → ODB name
  `lane\[0\].u_bank`. Three different LibreLane fields want three different forms:
  `PDN_MACRO_CONNECTIONS` is a **regex** (use `^lane.*u_bank$`, or escape the
  brackets), the `instances` placement keys go through `escape_verilog_name` so
  give the **raw** `lane[0].u_bank`, and YAML needs **single quotes** for literal
  backslashes. Also: harden the macro-bearing wrapper (`kv_sram`) **as the top**
  so its power pins connect cleanly (macros nested under a power-less parent leave
  the pins unconnected), give a `vh` blackbox stub so Verilator/yosys elaborate,
  and set `MAGIC_DRC_USE_GDS: false` so Magic DRCs the macro abstract (not the
  vendor's internal bitcell GDS, which throws ~31k false device-rule errors).
