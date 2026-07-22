# Sky130 OpenLane / LibreLane flow — `mate_pv_fp16`

End-to-end open-source RTL → GDSII flow for `mate_pv_fp16`, the **FP16 P·V MAC
tile** (IEEE-754 binary16 multiply, binary32 internal accumulator), targeting
SkyWater Sky130A. Same flow and tuning family as `../mate_pv` (the signed-off
INT8 tile), so the two blocks reach GDSII the same way and their area/power can
be diffed directly.

> 130 nm Sky130 proxy, used for 16 nm estimates — Lambda targets TSMC 16 nm.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane` (this run used
LibreLane **3.0.5**).

```sh
cd openlane/mate_pv_fp16
librelane --docker-no-tty --dockerized config.json
```

(This session drove the same 3.0.5 image directly via Docker, mounting the cached
Sky130A PDK; the committed `config.json` is the flow input either way.)

## Config

Based on `mate_pv/config.json`, with the differences a **full IEEE-754 fp32
datapath** (vs a plain int32 add) needs to close the Sky130 SS corner:

- `CLOCK_PERIOD` **85 ns (11.8 MHz)** — the accumulator is a *single-cycle
  feedback recurrence* (`acc ← fp32_add(acc, prod)`), so the whole fp32 adder
  (align barrel-shift → 28-bit add/sub → leading-zero-normalise barrel-shift →
  round-to-nearest-even) is on the critical path and **cannot be pipelined
  without dropping below 1 token/cycle**. That path is ~63 ns pre-route / ~74 ns
  post-route at the slow corner — ~4× the INT8 tile's int32 add, which is the
  whole reason FP16 clocks 6× slower than INT8's 14 ns.
- `SYNTH_PARAMETERS: ["N=4"]` — 4 lanes for the physical proxy (same as the INT8
  tile). Functional default is N=8; the bit-exact sim uses N=8 (**627 FFs**).
- `FP_CORE_UTIL 45` / `PL_TARGET_DENSITY_PCT 55` — the fp32 datapath is ~2.7×
  the INT8 std-cell area, so it floorplans a bit looser to route and buffer
  cleanly.
- `DESIGN_REPAIR_MAX_CAP_PCT / GRT_DESIGN_REPAIR_MAX_CAP_PCT 60`,
  `GRT_DESIGN_REPAIR_MAX_WIRE_LENGTH 150`, `RUN_POST_GRT_DESIGN_REPAIR true` —
  extra max-cap repair headroom + a post-route repair pass. The big combinational
  fp32 adder has many high-fanout nets (exponent-difference / shift-control
  broadcasts); the default single pre-route repair left a handful of resizer
  buffers marginally over the SS-corner cap limit, so a post-route repair pass
  with margin drives Max-Cap to zero.
- `CTS_CLK_MAX_WIRE_LENGTH 100` + generous `*_RESIZER_HOLD_SLACK_MARGIN` — the
  last max-cap offender was the CTS clock-root buffer; splitting the clock wire
  clears it, and the raised hold-repair margin re-fixes the small hold violations
  that the clock-tree change introduces (setup has ~8.5 ns of slack to spare).

`src/mate_pv_fp16.sv` is the block top (kept in sync with `rtl/mate_pv_fp16.sv`).

## Sign-off — Sky130A (N=4 proxy)

All six physical checks are **zero**. Committed under `results/`
(`mate_pv_fp16.gds` + `mate_pv_fp16.png` render + the signoff metrics json).

| metric | value | where it comes from |
|---|---|---|
| **fmax** | 11.8 MHz (85 ns) | the *constrained* `CLOCK_PERIOD` OpenSTA confirms all paths meet at every PVT corner — **not** a measured max. Worst setup slack is +8.4 ns (the single-cycle fp32 adder dominates but has margin at 85 ns). |
| **setup / hold WS** | +8.4 / +0.01 ns | worst timing slack across all corners, OpenSTA at the slow `ss_100C_1v60` / fast corners. Zero setup and hold violations (the tight +0.01 ns hold is post hold-buffer repair — positive at every corner). |
| **die area** | 465.8 × 476.5 µm = 221,987 µm² | floorplanner sizes the die from *std-cell area ÷ target utilization* + IO margin; tracks cell count and packing. |
| **std-cell area** | 111,462 µm² (14,583 cells) | Σ of the `sky130_fd_sc_hd` cell areas after synth + resizer buffers — the fp32 adder + fp16 multiplier + fp16/fp32 rounding logic. **≈ 2.8× the INT8 tile** (39,948 µm²). |
| **sequential (FF)** | 315 | flip-flops: 4×32 fp32 acc + 4×32 fp32 prod_reg + 4×16 fp16 c_data + 3 control, minus a few constant-optimised bits. Essentially equal to INT8's 323. |
| **core utilization** | ≈ 54 % | fraction of the core area filled by std cells. |
| **total power** | 10.44 mW (4.60 internal + 5.84 switching + ~0 leakage) | OpenSTA estimate at 1.8 V / 11.8 MHz under an **assumed default toggle rate** (no workload VCD) — an estimate, not measured. At the INT8 tile's 71 MHz this would scale up ~6×; the fair cross-tile number is **energy/op** (below). |

**Derivation caveats:** (1) these are the **N=4 proxy** (4 lanes); the functional
block is **N=8**, so ≈ **2×** the area / cells / power. (2) 130 nm Sky130A, 1.8 V,
timing at the slow corner. (3) power is at the FP16 block's own 85 ns clock — see
the FP16-vs-INT8 delta note.

## FP16 vs INT8 (`mate_pv`) — the measured delta

| quantity | INT8 `mate_pv` | FP16 `mate_pv_fp16` | ratio |
|---|---|---|---|
| std-cell area | 39,948 µm² | 111,462 µm² | **2.79×** |
| cells | 4,508 | 14,583 | 3.24× |
| flip-flops | 323 | 315 | 0.98× (≈ equal) |
| die area | 75,660 µm² | 221,987 µm² | 2.93× |
| fmax | 71.4 MHz | 11.8 MHz | **0.16× (6.1× slower)** |
| power @ own clock | 5.53 mW | 10.44 mW | 1.89× |
| energy / token-cycle | 0.077 nJ | 0.89 nJ | **≈ 11.5×** |

FP16 costs ≈ 2.8× the area and ≈ 11.5× the energy per MAC of INT8, runs ≈ 6× slower
(single-cycle fp32 accumulate), and uses ≈ the same flip-flop count.

## Porting to 16 nm (TSMC N16) — rough estimates ⚠️

Lambda targets TSMC N16; Sky130 (130 nm) is the open proxy we can actually run.
Scaling 130 nm → 16 nm with first-order node ratios (**crude — see disclaimer**):

| quantity | Sky130 (N=4 proxy) | ×ratio (130→16 nm) | 16 nm estimate (N=4) | full block (N=8, ≈2×) |
|---|---|---|---|---|
| std-cell area | 111,462 µm² | ÷ ~40 (area shrink) | ~2,800 µm² | ~5,600 µm² |
| fmax | 11.8 MHz | × ~6 (FO4 delay) | ~70 MHz | ~70 MHz |
| dynamic power / op | (energy 0.88 nJ/op @ N=4) | ÷ ~8 (lower C, 1.8→~0.8 V) | order-of-magnitude lower | ~2× the N=4 figure |

Ratio basis identical to the INT8 tile's README: linear feature ratio 130/16 ≈ 8×;
area ≈ square of cell-pitch ratio, de-rated for FinFET/routing → ~40×; FO4 delay
~70–100 ps @ 130 nm vs ~12–15 ps @ N16 → ~6×; dynamic power ∝ C·V²·f.

> ⚠️ **Disclaimer.** These 16 nm figures are **order-of-magnitude extrapolations
> from node ratios, not a 16 nm implementation.** Real N16 numbers depend on the
> actual PDK, cell library, Vdd, corner set, and P&R — none of which we have (that
> is *why* Sky130 is the proxy). Treat as sanity-check magnitude only, ±2×+.
