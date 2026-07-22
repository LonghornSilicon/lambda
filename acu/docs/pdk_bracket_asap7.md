# PDK bracket — Sky130 (130 nm planar) vs ASAP7 (7 nm FinFET) on the same RTL

**One line:** the three signed-off APA logic tiles, taken through a second PDK —
predictive 7 nm FinFET ASAP7 — so we can bracket each block between a real planar
node and a FinFET node on *identical* RTL. The FinFET side is the point: it is a far
better device-family proxy for the chip's **TSMC 16 nm FinFET** target than planar
Sky130, even though the numbers themselves are predictive.

---

## Read this first — what these numbers are and are NOT

- **ASAP7 is a PREDICTIVE / RESEARCH PDK (ASU + ARM), not a foundry PDK.** Its rule
  deck is realistic-*shaped* but **not manufacturable**; its cell timing and power are
  model-based (NLDM), not silicon-calibrated. Treat every ASAP7 number here as a
  **research estimate**, not sign-off.
- **"Sign-off" on ASAP7 means "passes the predictive rules and closes timing"** — WNS/
  TNS = 0 setup and hold, DRC-clean on the predictive deck. It does **not** mean
  tape-out-ready. The Sky130 column, by contrast, is a real OpenLane GDSII sign-off.
- **Area de-scale is RESOLVED: the ASAP7 µm² below are REAL, not the 16×-inflated
  raw draw.** ASAP7's *original* GDS/LEF are intentionally drawn at 4× real dimensions
  (so raw areas are 16× too large). This ORFS platform ships the **de-scaled 1× LEFs**.
  Verified directly, not assumed: the placement site `asap7sc7p5t` is `SIZE 0.054 BY
  0.270` µm — exactly the *real* ASAP7 54 nm CPP and 270 nm 7.5-track cell height. So
  **no /16 is applied or needed.** Independent cross-check: mate_pv shrinks ≈ 91× from
  Sky130, which is physically sane for 130 nm → 7 nm; a raw 16×-inflated ASAP7 number
  would imply an impossible ≈ 6× shrink. Both signals agree → the areas are real.
- **Power is the softest metric here** — doubly so because (a) ASAP7 power models are
  predictive and (b) each node is measured at its *own* closing frequency, which differ
  by ~15–28×. Compare power with care; the energy-per-cycle column (power ÷ fmax) is the
  fairer cross-node view, but it too inherits the predictive-power softness.
- **No SRAM / no IO cells** are involved — all three tiles are pure logic, which is
  exactly why they suit a predictive PDK. (ASAP7's SRAM/IO story is weak; irrelevant here.)

---

## Method (apples-to-apples with the Sky130 sign-off)

- Same RTL files as the Sky130 runs (`rtl/mate_pv.sv`, `rtl/mate_pv_fp16.sv`,
  `rtl/precision_controller.sv`), **unmodified**.
- Same **N = 4** head-dim proxy for the two MAC tiles (matches `SYNTH_PARAMETERS:["N=4"]`
  in the OpenLane configs); precision_controller carries no parameter override on either
  node. FF counts match across nodes (323 / 315 / 30) — a good same-netlist cross-check.
- Same floorplan-utilisation intent (core util 50 / 45 / 50 %, place density 60/55/60 %).
- Tool: **OpenROAD-flow-scripts**, built-in `asap7` platform (RVT primary Vt, NLDM),
  run in the `openroad/orfs` docker image. Corners: setup SS 0.63 V/100 °C, hold FF
  0.77 V/25 °C, nominal TT 0.70 V/0 °C. Full flow synth→floorplan→place→CTS→route→final.
- fmax found the Sky130 way: start loose, tighten the clock until WNS just reaches 0.
  Configs + closing reports live in `orfs/asap7/<block>/` (see `orfs/asap7/README.md`).

---

## The bracket

Area is reported two ways: **cell area** = placed standard-cell area excluding fill
(the PDK-neutral logic-area metric; matches Sky130's `design__instance__area__stdcell`),
and **core area** = the placed core incl. fill/whitespace at the util above. Power is at
each node's closing fmax.

### INT8 P·V MAC — `mate_pv` (N=4)
| Metric | Sky130 — 130 nm planar (sign-off) | ASAP7 — 7 nm FinFET (predictive) | Sky130 → ASAP7 |
|---|---|---|---|
| fmax (closes, WNS=0) | **71 MHz** (14.0 ns) | **2.00 GHz** (500 ps, +18 ps slack) | **28.2× faster** |
| Cell area (excl. fill) | 39 948 µm² | **440 µm²** | 90.8× smaller |
| Core area (incl. fill) | 66 802 µm² | 792 µm² | 84.4× smaller |
| Power @ fmax | 5.53 mW @ 71 MHz | 5.65 mW @ 2.00 GHz | ~same power, 28× the speed |
| Energy / cycle (P÷f) | 77.9 pJ | **2.83 pJ** | 27.6× less |
| Leakage | 68 nW | 321 nW | see FinFET note |
| Flip-flops | 323 | 323 | identical netlist |

### FP16 P·V MAC — `mate_pv_fp16` (N=4)
| Metric | Sky130 — 130 nm planar (sign-off) | ASAP7 — 7 nm FinFET (predictive) | Sky130 → ASAP7 |
|---|---|---|---|
| fmax (closes, WNS=0) | **11.76 MHz** (85 ns) | **286 MHz** (3500 ps, +84 ps slack) | **24.3× faster** |
| Cell area (excl. fill) | 111 462 µm² | **1 753 µm²** | 63.6× smaller |
| Core area (incl. fill) | 206 443 µm² | 3 931 µm² | 52.5× smaller |
| Power @ fmax | 10.44 mW @ 11.76 MHz | 23.87 mW @ 286 MHz | (very different f — see E/cyc) |
| Energy / cycle (P÷f) | 888 pJ | **83.5 pJ** | 10.6× less |
| Leakage | 201 nW | 1 273 nW | see FinFET note |
| Flip-flops | 315 | 315 | identical netlist |

### Precision controller — `precision_controller`
| Metric | Sky130 — 130 nm planar (sign-off) | ASAP7 — 7 nm FinFET (predictive) | Sky130 → ASAP7 |
|---|---|---|---|
| fmax (closes, WNS=0) | **80 MHz** (12.5 ns) | **1.18 GHz** (850 ps, +24 ps slack) | **14.7× faster** |
| Cell area (excl. fill) | 3 438 µm² | **56 µm²** | 61.0× smaller |
| Core area (incl. fill) | 5 816 µm² | 98 µm² | 59.6× smaller |
| Power @ fmax | 0.330 mW @ 80 MHz | 0.196 mW @ 1.18 GHz | lower power, 15× the speed |
| Energy / cycle (P÷f) | 4.12 pJ | **0.17 pJ** | 24.7× less |
| Leakage | 6.3 nW | 39 nW | see FinFET note |
| Flip-flops | 30 | 30 | identical netlist |

> fmax speedups differ by block (14.7× / 24.3× / 28.2×) because each has a different
> critical-path *character*: precision_controller's limiter is one wide (24-bit) constant-
> multiply-and-compare — a long carry chain that the FinFET node's wire RC helps least,
> so it gains the least; the pipelined INT8 MAC (short mult→reg / add→reg stages) gains
> the most. That block-dependent spread is itself a useful FinFET-vs-planar signal.

---

## INT8-vs-FP16 delta — at *each* node

This is the number MatE cost-model docs care about: how much does the FP16 datapath cost
over INT8, and does that ratio move between nodes?

| Delta (FP16 relative to INT8) | Sky130 (130 nm planar) | ASAP7 (7 nm FinFET) |
|---|---|---|
| Cell area (excl. fill) | **2.79×** | **3.99×** |
| Core area (incl. fill) | 3.09× | 4.97× |
| fmax advantage of INT8 | 6.04× | 6.99× |
| Energy / cycle | 11.4× | 29.5× *(softest — see caveat)* |

**Finding:** the FP16 datapath's **area penalty over INT8 grows on the FinFET node**
(2.8× → 4.0× in cell area). The fmax gap is roughly node-invariant (~6–7×, INT8 faster),
consistent with the FP path being a fixed extra pipeline depth rather than a wire-limited
structure. The energy-per-cycle gap widening (11× → 30×) is directionally consistent but
sits on the **softest** (predictive-power) footing, so treat it as a trend, not a spec.
Practical read for the ACU: on a FinFET-class node the incentive to gate to INT8 (the
whole point of the precision controller) is **larger**, not smaller, than the Sky130
bracket suggested.

---

## Why ASAP7 is a better 16 nm proxy than Sky130 (despite being predictive)

The chip targets **TSMC 16 nm FinFET**. Sky130 is 130 nm **planar bulk**; ASAP7 is 7 nm
**FinFET**. Neither is 16 nm, but the *device family* is what carries the qualitative
behavior, and only ASAP7 is FinFET:

1. **Leakage regime.** On planar Sky130, leakage is a rounding error (6–200 nW here) and
   the flow barely thinks about it. On the FinFET node it is a **first-class citizen**:
   ASAP7 leakage is 5–6× the Sky130 leakage in *absolute* terms despite 60–90× less area
   — i.e. leakage *per area* is ~300–500× higher. 16 nm shares this regime qualitatively
   (leakage/Vt-mix is a real design axis), so ASAP7 exercises the leakage-vs-speed trade
   that Sky130 lets you ignore.
2. **Fin quantization.** FinFET drive strength comes in **discrete fin-count steps**, not
   the continuous transistor-width knob of planar. Sizing/buffering therefore snaps to
   quantized drives — we hit this directly: the fast FinFET min-delays plus quantized
   hold-buffer drive produced a hold-fixing sensitivity (a "hold storm" at loose hold
   margin) that simply does not appear on Sky130. 16 nm buffering behaves the same way.
3. **Wire-dominated delay + resistance-aware routing.** At sub-20 nm the interconnect RC,
   not the gate, sets a lot of the delay; the ASAP7 flow routes resistance-aware over a
   thin 7 nm BEOL. That is the regime 16 nm lives in; Sky130's fat planar BEOL is far more
   forgiving and hides it.
4. **Steeper subthreshold slope / lower DIBL** (FinFET electrostatics) shift the timing-vs-
   voltage curve toward the 16 nm shape rather than the planar one.

So the ASAP7 column should be read as **"what the FinFET device family does to this RTL,"**
directionally toward 16 nm — with the standing caveat that the *absolute* values are
predictive and uncalibrated, and only a real 16 nm PDK gives sign-off numbers.

---

## Provenance / reproduce

- Configs + closing reports: `orfs/asap7/{mate_pv,mate_pv_fp16,precision_controller}/`
  (`config.mk`, `constraint.sdc`, `results_asap7/*_metrics.json`, `6_finish.rpt`, GDS, webp).
- Driver + full method + the two flow gotchas worked around: `orfs/asap7/README.md`.
- Sky130 sign-off metrics (the left column): `openlane/<block>/results/sky130_*_signoff_metrics.json`.
- Tool: `openroad/orfs` docker, built-in `asap7` platform (de-scaled 1× LEFs, RVT/NLDM).
