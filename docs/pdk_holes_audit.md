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
| `mate_qkt` (Q·Kᵀ) | ✅ | ❌ **none** (Yosys smoke only) | ✅ | — | ✅ RTL + GF180 GLS |
| `vecu_softmax` | ✅ | ❌ **none** | ✅ (ss fixed) | — | ✅ RTL + GF180 GLS |
| `kv_cache_engine` (KVE) | ✅ | ✅ **signed off** 9-corner (5/6 clean; ss-corner reset-tree cap/slew tracked) — 0.236 mm², ~24 MHz ss; `SRAM_DEPTH=2` flop proxy [merged to `rtl` `d4143c1`] | ✅ real `gf180mcu_fd_ip_sram` (4 macros); **DRC 0 / LVS 0**, setup +17.5/hold +6.9; bit-exact round-trip + full GLS e2e pass | — | ✅ RTL; GLS via combinational reconstruct |
| RoPE | ❌ **no RTL** | — | — | — | reference stand-in (pre-RoPE'd tiles) |
| RMSNorm | ❌ **no RTL** | — | — | — | reference stand-in |
| `lambda_acu` top + decode FSM | ❌ **stub only** | — | — | — | testbench-stitched; no integrated top |
| MatE 8×8 GEMM/FFN systolic · full VecU · MSC · LSU · HIF | ❌ no RTL (off-shuttle scope) | — | — | — | off-chip / spec-only |

## 🔴 130nm (Sky130) flagship holes — PRIORITY

The flagship is currently **less complete than GF180** for the datapath, because `mate_qkt` and
`vecu_softmax` were advanced on GF180 with Sky130 sign-off skipped. To keep 130nm as the flagship:

1. **`mate_qkt` — no Sky130 sign-off.** Only a Yosys smoke + GF180 hardening exist. The flagship
   cannot do Q·Kᵀ scoring in 130nm silicon terms. → *Backfill a Sky130 OpenLane sign-off.*
2. **`vecu_softmax` — no Sky130 sign-off.** Only GF180. Flagship missing softmax. → *Backfill
   Sky130 (use the pipelined + area-reclaimed version once it lands).*
3. ~~**KVE — no committed Sky130 sign-off.**~~ **RESOLVED 2026-07-21** — Sky130A 9-corner sign-off
   committed (`sky130-kve-signoff` `440c7b5`, pending merge to `rtl`): setup/hold/DRC/LVS/antenna
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
2. **`vecu_softmax` area** — the ss-close resize ~2×'d cells (→1.49 mm²). **RTL rebalanced 2026-07-21**
   (`attention-compute-unit` `rtl` `2c458aa`, architecture `rtl` `d837b42`): converted to a
   multi-cycle datapath (one fp32 op/cycle, longest path ~one fp32 op vs two → closes ss at a
   faster clock with *normal* resizing, no cell-cloning blow-up; +FFs but combinational area
   returns toward the ~55k base). Bit-exact, cosim re-confirmed `ALL BLOCKS PASS`. → **GF180
   re-harden pending** (queued behind the KVE SRAM agent, which owns the chipathon repo) to confirm
   the actual area drop + that ss still closes.
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

1. **Flagship parity (Sky130):** ~~KVE Sky130 run~~ ✅ done; still backfill Sky130 sign-off for
   `mate_qkt` + `vecu_softmax` (use the area-reclaimed softmax). *The flagship should not trail the shuttle.*
2. GF180 KVE real SRAM macro + `vecu_softmax` area — *in progress.*
3. `lambda_acu` integration top + decode FSM (both PDKs) → Phase 3.
4. RoPE / RMSNorm RTL for the chip-top.
5. ss-corner slew physical-opt; full-chip padring GL.
