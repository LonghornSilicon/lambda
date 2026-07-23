# PDK Holes Audit — Sky130 (flagship) & GF180 (shuttle)

**Living document (started 2026-07-21).** Honest, exhaustive gap register for the decode
attention datapath across PDKs. Update it as holes close — do not let it go stale.

**PDK roles:**
- **Sky130 (130nm)** = the **flagship** — a real, manufacturable PDK; our closest-to-actual-chip
  proof. Priority for completeness.
- **GF180MCU** = the **chipathon shuttle** target (SSCS Chipathon 2026).
- **ASAP7 (7nm)** = predictive **research bracket** only (16nm-FinFET proxy), not a tapeout.

## Master matrix (decode attention datapath)

| Block | RTL | Sky130 signoff | GF180 signoff | ASAP7 | cosim / GLS |
|---|---|---|---|---|---|
| `precision_controller` | ✅ | ✅ 80 MHz | ✅ | ✅ | ✅ RTL + GF180 GLS |
| `mate_pv` (INT8 P·V) | ✅ | ✅ 71 MHz | ✅ | ✅ | ✅ RTL + GF180 GLS |
| `mate_pv_fp16` (FP16 P·V) | ✅ | ✅ 12 MHz | ✅ | ✅ | ✅ RTL + GF180 GLS |
| `token_importance_unit` | ✅ | ✅ (multi-corner) | ✅ | — | ✅ RTL + GF180 GLS |
| `mate_qkt` (Q·Kᵀ) | ✅ | ✅ **signed off** 9-corner (DRC/LVS/antenna 0; residual ss slew) | ✅ | — | ✅ RTL + GF180 GLS |
| `vecu_softmax` | ✅ | ✅ **signed off** 9-corner, 105 ns/9.5 MHz (DRC/LVS/antenna/max-cap 0; residual ss slew) | ✅ (ss fixed) | — | ✅ RTL + GF180 GLS |
| `kv_cache_engine` (KVE) | ✅ | ✅ **signed off** 9-corner (5/6 clean; ss-corner reset-tree cap/slew tracked) — 0.236 mm², ~24 MHz ss; `SRAM_DEPTH=2` flop proxy [committed in `kve/pdk/sky130/openlane/kv_cache_engine/results/`, run tag `sky130_signoff`] | ✅ real `gf180mcu_fd_ip_sram` (4 macros); **DRC 0 / LVS 0**, setup +17.5/hold +6.9; bit-exact round-trip + full GLS e2e pass | — | ✅ RTL; GLS via combinational reconstruct |
| RoPE | ❌ **no RTL** | — | — | — | reference stand-in (pre-RoPE'd tiles) |
| RMSNorm | ❌ **no RTL** | — | — | — | reference stand-in |
| `lambda_acu` top + decode FSM | ❌ **stub only** | — | — | — | testbench-stitched; no integrated top |
| MatE 8×8 GEMM/FFN systolic · full VecU · MSC · LSU · HIF | ❌ no RTL (off-shuttle scope) | — | — | — | off-chip / spec-only |

## 🔴 130nm (Sky130) flagship holes — PRIORITY

The flagship-parity backfill is **done** — `mate_qkt` and `vecu_softmax` now have committed
Sky130 sign-offs, so 130nm covers the full datapath (no longer trailing GF180):

1. ~~**`mate_qkt` — no Sky130 sign-off.**~~ **RESOLVED** — Sky130A 9-corner sign-off committed
   (`acu/mate/pdk/sky130/openlane/mate_qkt/`): DRC/LVS/antenna 0, setup/hold met all corners
   (residual ss max-slew only). The flagship now does Q·Kᵀ scoring in 130nm silicon terms.
2. ~~**`vecu_softmax` — no Sky130 sign-off.**~~ **RESOLVED** — Sky130A 9-corner sign-off committed
   (`acu/vecu/pdk/sky130/openlane/vecu_softmax/`, multi-cycle revision `2c458aa`): 105 ns / 9.5 MHz,
   DRC/LVS/antenna/max-cap 0 (residual ss slew only).
3. ~~**KVE — no committed Sky130 sign-off.**~~ **RESOLVED 2026-07-21** — Sky130A 9-corner sign-off
   committed (in `kve/pdk/sky130/openlane/kv_cache_engine/results/`, run tag `sky130_signoff`): setup/hold/DRC/LVS/antenna
   all 0; residual **max-cap 5 + max-slew 1503 at the ss corner only**, from the high-fanout async
   `rst_n` tree over the ~1500-flop array (functionally clean — async reset, recovery/removal +90 ns;
   it's the tracked ss-corner physical-opt item, reset-tree buffering). Also fixed a real config bug
   (`DESIGN_REPAIR_MAX_SLEW_PCT=0` had *disabled* slew repair; restored 20% → max-cap 61→5,
   max-slew 5042→1503). 0.236 mm², ~24 MHz ss. **Still open:** the `SRAM_DEPTH=2` flop proxy below.
4. **KVE storage is a `SRAM_DEPTH=2` flop proxy on Sky130 too** — no real KV capacity at 130nm
   (same hole as GF180). → *Sky130 SRAM macro (sky130 OpenRAM/DFFRAM) or documented proxy.*
5. **No Sky130 integrated top.** Blocks are signed off individually; there is no Sky130 GDSII of
   the integrated ACU datapath, and the cosim is RTL-only. → *Phase-3 `lambda_acu` top → Sky130.*

## 🟠 GF180 (shuttle) holes

1. ~~**KVE `gf180mcu_fd_ip_sram` macro**~~ — **RESOLVED 2026-07-22 (chipathon `rtl` `5514cb0`).**
   KV storage backed by 4 real `gf180mcu_fd_ip_sram__sram512x8m8wm1` macros (32b×512, die 1.27 mm²)
   via a PDK-agnostic `kv_sram` interface (kve `rtl` `f6ed2db`). **Fully clean GF180 sign-off:
   Magic-DRC 0, Netgen-LVS 0, setup +17.5 ns, hold +6.9 ns, antenna 0.** The PDN close: connect the
   SRAM's **Metal3** power pins to the Metal4 straps with a legal **Via3** (the old Metal1/Metal2
   route forced illegal Via1/Via2 stacks) → DRC 7026→8, LVS 6→0; then a documented **DRC-view-only
   maglef** widening one sub-min-width pin in the *vendor SRAM abstract* (0.11 µm vs 0.28 µm — an
   abstract artifact; the vendor's real GDS is DRC-clean and LVS ran on the real device = 0) → DRC 8→0.
   **Independently re-verified:** bit-exact SRAM round-trip + full GLS e2e **ALL PASS**. Residual
   ss-corner slew/cap (54/12) on mux/control logic tracked separately (DRC/LVS-independent).
2. ~~**`vecu_softmax` area**~~ — **RESOLVED 2026-07-22 (GF180 re-harden done; no area drop).** The RTL
   was rebalanced to a multi-cycle datapath (one fp32 op/cycle, `2c458aa`) and GF180-re-hardened: it
   **closes ss at +60.9 ns @ 260 ns** with normal resizing (all corners meet setup/hold, bit-exact,
   cosim `ALL BLOCKS PASS`). But the predicted area drop did **not** happen — the multi-cycle version
   is **111,253 cells / 1.64 mm², ~10% *larger*** than the 3-stage's 101,236 / 1.49 mm²; the 1.49 mm²
   was largely *inherent*, not ss-close resize bloat (`chip/pdk/gf180/docs/gf180_gls_report.md` §1).
   The win is timing robustness, not area.
3. **ss-corner max-transition (slew)** on the large fp16 / register-array blocks (`mate_pv_fp16`,
   `vecu_softmax`, `kve`). Setup/hold/DRC/LVS unaffected. → physical-opt (driver upsizing / slew
   repair).
4. **No integrated `lambda_acu` top hardened on GF180.** Blocks are standalone macros; the
   padring stitching (`chip_core.sv` override + SPI loader → `chipathon-2026-gf180mcu-padring`)
   is not done. → Phase 5.
5. **Full-chip cocotb GL** (chip_core through the padring) not run — only per-block + the
   datapath GLS.

## ⚪ Cross-cutting (both PDKs)

- **RoPE + RMSNorm** have no RTL — the cosim uses pre-RoPE'd Qwen tiles, so the on-die raw-Q/K
  path doesn't exist yet. Needed for a self-contained chip-top.
- **`lambda_acu` integration top + decode-step FSM** is a stub — the datapath is stitched by the
  testbench, not a single RTL module.
- **Projections (QKV/output), FFN GEMMs, logits** — off-chip by design for this shuttle (the
  general 8×8 systolic GEMM engine was never RTL).

## Priority ranking (as of 2026-07-21)

1. ~~**Flagship parity (Sky130):**~~ ✅ **done** — KVE, `mate_qkt`, and `vecu_softmax` all have
   committed Sky130 sign-offs; the flagship covers the full datapath and no longer trails the shuttle.
2. ~~GF180 KVE real SRAM macro + `vecu_softmax` area~~ — ✅ **done** — KVE real `gf180mcu_fd_ip_sram`
   macro signed off; `vecu_softmax` GF180 re-hardened (area **not** reclaimed — see GF180 hole #2).
3. `lambda_acu` integration top + decode FSM (both PDKs) → Phase 3.
4. RoPE / RMSNorm RTL for the chip-top.
5. ss-corner slew physical-opt; full-chip padring GL.
