# Sky130 OpenLane / LibreLane flow — `mate_qkt`

End-to-end open-source RTL → GDSII flow for `mate_qkt`, the **Q·Kᵀ score tile**
(INT8 query × FP16 key IEEE-754 binary16 multiply, binary32 internal
accumulator, round-to-fp16 scores), targeting SkyWater **Sky130A**. Same flow and
tuning family as `../mate_pv_fp16` (the signed-off FP16 P·V tile), so the two FP16
blocks reach GDSII the same way and their area/power can be diffed directly.

> 130 nm Sky130 proxy, used for 16 nm estimates — Lambda targets TSMC 16 nm.

This is one of the two **flagship-parity backfills** (audit priority #1): `mate_qkt`
and `vecu_softmax` had GF180 sign-off but no Sky130, so the 130 nm flagship could not
score Q·Kᵀ or do softmax in silicon terms. This closes the Q·Kᵀ half.

## Run it

Requires Docker (~25 GB free disk); this run used LibreLane **3.0.5** on the
Sky130A PDK (ciel version `8afc8346`).

```sh
# enable the Sky130A PDK once (ciel manages PDK_ROOT):
ciel enable --pdk-family sky130 8afc8346a57fe1ab7934ba5a6056ea8b43078e71

cd openlane/mate_qkt
librelane --docker-no-tty --dockerized --pdk sky130A config.json
```

## Config

Based on `mate_pv_fp16/config.json` (same FP16 datapath family):

- `CLOCK_PERIOD` **80 ns (12.5 MHz)** — `mate_qkt` accumulates each head-dim
  channel's INT8·FP16 product into a **single-cycle fp32 feedback recurrence**
  (`acc ← fp32_add(acc, int8×fp16_prod)`), so the fp32 adder (align → 28-bit
  add/sub → leading-zero-normalise → round-to-nearest-even) plus the product is on
  the critical path and cannot be pipelined without dropping below 1 channel/cycle.
  The worst reg-to-reg path is **72.5 ns at the slow `ss_100C_1v60` corner**, so
  80 ns closes with **+7.5 ns** of slack — the clean-close point (a hair faster than
  the P·V tile's 76 ns fp32 path because there is no separate prod-register stage).
- `SYNTH_PARAMETERS: ["N=4"]` — 4 score lanes for the physical proxy (same as the
  P·V tiles). Functional default is N=8 (**≈ 2× area / cells**).
- `FP_CORE_UTIL 45` / `PL_TARGET_DENSITY_PCT 55`, `DESIGN_REPAIR_MAX_CAP_PCT 60` +
  `RUN_POST_GRT_DESIGN_REPAIR` — the fp32 datapath's high-fanout exponent/shift
  broadcasts need extra max-cap repair headroom + a post-route repair pass to drive
  **Max-Cap to zero** (same recipe as `mate_pv_fp16`).
- `PL_/GRT_RESIZER_HOLD_SLACK_MARGIN 0.6` — the tighter 80 ns CTS left one hold path
  marginally negative at 0.4 ns margin; 0.6 ns re-fixes it (setup has +7.5 ns to
  spare). Hold closes at **+0.29 ns**.

`src/mate_qkt.sv` is the block top (kept in sync with `rtl/mate_qkt.sv`).

## Sign-off — Sky130A (N=4 proxy)

**All six physical checks are zero, multi-corner across all 9 IPVT corners**
(`{min,nom,max} × {tt_025C_1v80, ss_100C_1v60, ff_n40C_1v95}`). Committed under
`results/` (`mate_qkt.gds` + `mate_qkt.png` render + `sky130_signoff_metrics.json`).

| check | value |
|---|---|
| setup violations | **0** (WNS 0, all 9 corners; worst slack +7.5 ns @ ss) |
| hold violations | **0** (WNS 0; worst slack +0.29 ns) |
| Magic DRC / KLayout DRC | **0 / 0** |
| LVS (Netgen, incl. device diff) | **0** |
| antenna | **0** |
| **max-cap** | **0** (all 9 corners) |

Also clean: power-grid IR = 0, XOR diff = 0. Reported honestly (not one of the six):
`max_slew = 674`, `max_fanout = 3` — residual slew is at the ss corner only (the
FP16 family's known register-tree transition item; setup/hold/DRC/LVS unaffected),
the same class of residual as `mate_pv_fp16` (734).

| metric | value | notes |
|---|---|---|
| **fmax** | 12.5 MHz (80 ns constrained) | OpenSTA confirms all paths meet at every corner; the ss reg-to-reg path is 72.5 ns → ~13.8 MHz intrinsic. Not pipelined (single-cycle fp32 accumulate), like the P·V FP16 tile. |
| setup / hold WS | +7.5 / +0.29 ns | worst across all corners (ss / fast). Zero setup and hold violations. |
| die area | 198,901 µm² (0.199 mm²) | floorplanner: std-cell area ÷ target util + IO margin. |
| std-cell area | 99,950 µm² (12,973 cells) | fp32 adder + INT8·FP16 multiplier + fp16 round logic. |
| sequential (FF) | 315 | 4×32 fp32 acc + 4×32 fp32 prod + 4×16 fp16 score + control. |
| core utilization | 54.5 % | |
| total power | ~12.4 mW | OpenSTA estimate @ 1.8 V / 12.5 MHz, default toggle (no workload VCD) — an estimate, not measured. |

**Derivation caveats:** (1) **N=4 proxy** (4 lanes); the functional block is N=8, so
≈ 2× area / cells / power. (2) 130 nm Sky130A, 1.8 V, timing at the slow corner.
(3) power is at this block's own 80 ns clock under an assumed toggle rate.

## mate_qkt (Q·Kᵀ) vs mate_pv_fp16 (P·V) — same FP16 family

| quantity | `mate_qkt` (Q·Kᵀ) | `mate_pv_fp16` (P·V) |
|---|---|---|
| std-cell area | 99,950 µm² | 111,462 µm² |
| cells | 12,973 | 14,583 |
| flip-flops | 315 | 315 |
| die area | 198,901 µm² | 221,987 µm² |
| fmax (constrained) | 12.5 MHz (80 ns) | 11.8 MHz (85 ns) |
| power @ own clock | ~12.4 mW | 10.4 mW |

Both are single-cycle fp32-accumulate FP16 tiles; `mate_qkt` is slightly smaller and
a touch faster (INT8 query multiplicand + no separate prod-register stage).
