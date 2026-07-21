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
| `vecu_softmax`          | N=8                 | 340 ns | 779 671 | 54 800 | **+140.0 (tt) / −26.5 (ss)** | +0.856 | 0 | 0 | 0 | 2 (ss) | 2117 (ss; 22 tt) |
| `kv_cache_engine` (kve) | SRAM_DEPTH=2, VECTOR_DIM=8, KEY_GROUP=2 | 200 ns | 621 960 | 32 294 | **+109.82** | +0.261 | 0 | 0 | 0 | 7 (ss) | 2392 (ss; 83 tt) |

**Implied fmax** (min period = clk − setup WS; loose clocks chosen for clean
closure, so conservative): precision_controller ≈ 40 MHz, token_importance_unit
≈ 58 MHz, mate_pv ≈ 36 MHz, mate_pv_fp16 ≈ 6.7 MHz, mate_qkt ≈ 6.7 MHz,
vecu_softmax ≈ 5.0 MHz **at the tt corner** (ss does not close — see below),
kve ≈ 11 MHz.

### The 6 signoff checks
- **Setup:** six of seven **meet setup across all corners** with positive margin
  (TNS = 0). The **one honest exception is `vecu_softmax`**: it meets setup at
  the `tt` (+140 ns) and `ff` (+213 ns) corners by a wide margin but **misses at
  the extreme `ss_125C_4v50` corner by −26.5 ns** (TNS −696 ns, 32 endpoints).
  The failing path is the online-softmax fp16→exp-LUT→fp32→fp16 chain, which is
  ~366 ns post-route at the ss corner — the longest single combinational path in
  the whole datapath. Even at a 340 ns clock (2.7× the fp16 P·V period) the ss
  corner + post-route parasitics push it over. Closing it needs **pipelining the
  exp/normalize path** (the right fix) or a ≥400 ns clock + denser floorplan; a
  documented follow-up. The netlist is functionally valid and is used in the GL
  e2e (§2) — timing closure at ss is orthogonal to logical correctness.
- **Hold:** all seven **meet hold** (WS positive, TNS = 0, all corners).
- **Magic DRC:** **0** on all seven.
- **Netgen LVS:** **0** on all seven (0 device/net/pin/property mismatches).
- **Antenna:** **0** violating nets on all seven.
- **Max-cap / max-transition (slew):** `precision_controller` is **fully clean
  on all corners**. The others have **max-transition violations confined to the
  extreme slow corner** `max_ss_125C_4v50` (the `tt`/`ff` corners are clean or
  near-clean). Known GF180 wide-corner artifact: GF180's ss corner is very slow,
  so min-size drivers that pass transition at tt/ff exceed it at ss. The big
  fp16 blocks (`mate_qkt`, `vecu_softmax`, `mate_pv_fp16`) and the register-array
  `kve` show the largest counts (un-optimised high-fanout nets on sprawling
  floorplans). DRC/LVS are unaffected (those are physical-rule / connectivity).

Closing the ss-corner setup + transition fully needs pipelining the long fp16
paths, driver up-sizing, and denser floorplans (multi-corner repair) — follow-ups
that do not affect functional correctness or the gate-level sim.

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
- **SRAM boundary (as requested, stated honestly):** at the gate-proxy
  `SRAM_DEPTH=2`, the KV storage (`scale_bank` / `residual_buffer`) synthesizes
  to **flip-flop register arrays** — the netlist contains **2114 DFF cells and
  zero `gf180mcu_fd_ip_sram` macro instances**. This is a **SIM/area proxy, not
  real KV capacity.** A real-capacity KVE needs the `gf180mcu_fd_ip_sram` hard
  macro (the PDK **does ship it** — `libs.ref/gf180mcu_fd_ip_sram` is installed)
  wired into the flow as a `MACRO` with PDN connections. That integration is
  **TODO** and is why the full `kv_cache_engine` is not in the gate-level GLS
  loop (its AXI interface + register-array storage differ from the cosim's
  combinational reconstruct). The register-array proxy is also what drives its
  2392 ss-corner transition violations (§1).

**The `gf180mcu_fd_ip_sram` macro integration is now the ONE remaining honest
hole** in gate-level datapath coverage. The entire *compute* datapath
(Q·Kᵀ → softmax → P·V, plus ACU + TIU) is GF180 gate-level verified end to end;
the only piece not on real gate-level SRAM is the KV *store* in `kv_cache_engine`.

**`vecu_softmax` — one documented RTL synth-compat patch:** the vendored source
declares its fp16-subnormal-normalize loop counter (`gi_unused`) at module scope
but uses it only inside `function automatic fp16_to_fp32`; the default yosys
frontend latch-infers it → 320 multiple-driver check errors. Moved the
declaration into that function (per-call) — **behaviorally + simulation
identical**, the only divergence from the byte-identical vendored source, flagged
inline and in `rtl/blocks/PROVENANCE.md`. `mate_qkt` is verbatim.

## 4. Reproduce

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
- **`vecu_softmax` is the hard one.** Its fp16→exp-LUT→fp32→fp16 chain is the
  longest path in the datapath; the post-CTS/post-GRT resizers grind for tens of
  minutes asymptotically closing it, and even at a 340 ns clock the ss corner is
  −26.5 ns post-route (tt/ff pass by >140 ns). The real fix is to **pipeline the
  exp/normalize path**, not to keep raising the period. Its netlist is logically
  correct and passes the GL e2e regardless of ss-corner timing.
