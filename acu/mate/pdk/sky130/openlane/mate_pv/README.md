# Sky130 OpenLane / LibreLane flow — `mate_pv`

End-to-end open-source RTL → GDSII flow for `mate_pv`, the INT8 P·V MAC tile
(INT32 accumulator), targeting SkyWater Sky130A. Same flow and tuning as
`../precision_controller` (the signed-off reference), so the two blocks reach
GDSII the same way.

> 130nm Sky130 proxy, used for 16nm estimates — Lambda targets TSMC 16nm.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/mate_pv
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`. First
invocation downloads the Sky130A PDK (~500 MB via Ciel) and the LibreLane
Docker image (~6 GB); subsequent runs reuse both caches.

## Config

Based on `precision_controller/config.json`, with the differences a real MAC (vs
a trivial comparator) needs to close the Sky130 SS corner:

- `CLOCK_PERIOD` **14 ns (71 MHz)** — the signed-off clock after the RTL was
  pipelined (`prod_reg` registers the multiply, splitting the old ~15 ns
  `mult→add` path into two ~7 ns stages). The un-pipelined block only closed at
  25 ns (40 MHz); pipelining nearly doubled fmax.
- `SYNTH_PARAMETERS: ["N=4"]` — synthesize **4 lanes** for the physical proxy
  run (same pattern as TIU's `N_SLOTS=4`); halves `a_data`'s fanout tree, which
  drove the Max-Cap violations. The functional RTL default is N=8; the bit-exact
  sim and the `expected-ff-count` synth gate use N=8 (**643 FFs**).
- `IO_DELAY_CONSTRAINT: 2` (vs 5) — inputs feed straight into the arithmetic.

`src/mate_pv.sv` is the block top (kept in sync with `rtl/mate_pv.sv`).

## Sign-off — clean at 71 MHz, Sky130A (N=4 proxy)

All six physical checks are **zero**. Committed under `results/`
(`mate_pv.gds` + `mate_pv.png` render + `sky130_71MHz_signoff_metrics.json`).

| metric | value | where it comes from |
|---|---|---|
| **fmax** | 71.4 MHz (14 ns) | the *constrained* `CLOCK_PERIOD` that OpenSTA confirms all paths meet at every PVT corner — **not** a measured max. Worst setup slack is +0.83 ns, so true max is a bit higher. |
| **setup / hold WS** | +0.83 / +0.22 ns | worst timing slack across all corners (margin to failure), from OpenSTA at the slow `ss_100C_1v60` / fast corners. |
| **die area** | 269.8 × 280.5 µm = 75,660 µm² | the floorplanner sizes the die from *std-cell area ÷ target utilization* + IO margin (`FP_CORE_UTIL`), so it tracks cell count and how tightly you pack, not a constant. |
| **std-cell area** | 39,948 µm² (4508 cells) | sum of the library-defined areas of each `sky130_fd_sc_hd` cell after synth + the resizer's buffers — the silicon the logic occupies, excluding routing whitespace. |
| **sequential (FF)** | 323 | flip-flops; matches the N=4 RTL: 4×32 acc + 4×32 c_data + 4×16 prod_reg + 3 control. |
| **core utilization** | 59.8 % | fraction of the core area filled by std cells. |
| **total power** | 5.5 mW (3.25 internal + 2.28 switching + ~0 leakage) | OpenSTA estimate = Σ(net cap × V² × activity) + cell internal + leakage, at 1.8 V / 71 MHz under an **assumed default toggle rate** (no workload VCD) — an estimate, not measured. |

**Two derivation caveats:** (1) these are the **N=4 proxy** (4 lanes); the full
functional block is **N=8**, so ≈ **2×** the area / cells / power. (2) 130 nm
Sky130A, 1.8 V, timing at the slow corner.

## Porting to 16 nm (TSMC N16) — rough estimates ⚠️

Lambda targets TSMC N16; Sky130 (130 nm) is the open proxy we can actually run.
Scaling 130 nm → 16 nm with first-order node ratios (**crude — see disclaimer**):

| quantity | Sky130 (N=4 proxy) | ×ratio (130→16 nm) | 16 nm estimate (N=4) | full block (N=8, ≈2×) |
|---|---|---|---|---|
| std-cell area | 39,948 µm² | ÷ ~40 (area shrink) | ~1,000 µm² | ~2,000 µm² |
| fmax | 71 MHz | × ~6 (FO4 delay) | ~430 MHz | ~430 MHz |
| dynamic power / op | (2.3 mW @ 71 MHz) | ÷ ~8 (lower C, 1.8→~0.8 V) | order-of-magnitude lower | ~2× the N=4 figure |

Ratio basis: linear feature ratio 130/16 ≈ 8×; area ≈ square of the cell-pitch
ratio, de-rated for FinFET/routing overhead → ~30–50× (we use ~40×); gate (FO4)
delay ~70–100 ps @ 130 nm vs ~12–15 ps @ N16 → ~6×; dynamic power ∝ C·V²·f with
C down ~8× and V² down ~5×.

> ⚠️ **Disclaimer.** These 16 nm figures are **order-of-magnitude extrapolations
> from node ratios, not a 16 nm implementation.** Real N16 numbers depend on the
> actual PDK, standard-cell library, Vdd, corner set, and P&R — none of which we
> have (that is *why* Sky130 is the proxy). Post-Dennard scaling breaks simple
> V/area/power rules, so treat these as a sanity-check magnitude only, ±2×+.

The `openlane-sky130` CI gate re-runs this config and asserts the six checks are
zero, same as every other block.
