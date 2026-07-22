# `mate_pv_fp16` — synthesizable FP16 P·V MAC tile (RTL)

**Status:** RTL complete + pipelined, bit-exact to the reference's FP16 semantics, Yosys-clean (315 FFs N=4 / 627 FFs N=8, no latches), GDSII-signed-off on Sky130A at 11.8 MHz.
**Home:** `rtl/mate_pv_fp16.sv` (+ `rtl/tb/tb_mate_pv_fp16.sv`, `rtl/tb/gen_mate_pv_fp16_vectors.py`).
**One line:** the FP16 datapath of the MatE P·V tile — the piece that previously existed only as a **"tolerance-only" behavioral note**, now real synthesizable IEEE-754 RTL so its Sky130 area/power can be measured and diffed against the INT8 `mate_pv`.

## Why this exists

The MatE MAC array has two datapaths: the INT8 tile (`precision_controller.d_fp16 == 0`,
implemented in `mate_pv.sv`) and the FP16 tile. Only the INT8 tile had synthesizable RTL;
the FP16 tile was **tolerance-only** in the reference (`MAC_ARRAY_DESIGN.md`), so every MatE
doc that quoted an "FP16 P·V area/power delta" had to write **"TBD pending re-synthesis."**
This block fills that gap: a real IEEE-754 binary16 P·V vector-MAC with an fp32 internal
accumulator, taken to GDSII in the same OpenLane/Sky130 flow as the INT8 tile, so the delta
is a measured number instead of a placeholder.

## What it computes (bit-exact contract)

Realises the ACU MAC-array reference `sw/reference_model/mac_array_ref.{hpp,py}`
`matmul_fp16` for M=1 (one attention row), `fp16_accumulator_bits = 32`:

```
o[n] = round_fp16( Σ_t  A[t] · V[t][n] )       A,V ∈ fp16 → o ∈ fp16
```

- **fp16 × fp16 → fp32, exact.** Two binary16 significands are ≤11 bits each, so the product
  is ≤22 bits — it fits fp32's 24-bit significand with no rounding, and its magnitude
  (2⁻⁴⁸ … 2³¹) is always an fp32 *normal*. So the multiply never loses a bit.
- **fp32 accumulate, round-to-nearest-even at every add.** The token-dim reduction runs in a
  full IEEE-754 binary32 adder (align → add/sub → renormalise → RTNE), matching the
  reference's fp32 internal accumulator.
- **round to fp16 exactly once**, on the emitted result (RTNE, overflow → inf,
  underflow → subnormal/0) — matching the reference's single output round.

### Accumulation order (the one honest subtlety)

The reference's Python/C++ model performs the fp32 reduction with numpy's BLAS `@`, whose
pairwise/blocked summation order is **machine-dependent and not itself canonical**. A
streaming token-reduction MAC accumulates **sequentially** in token-arrival order — that is
the hardware. This RTL is **bit-exact to a sequential-fp32 golden**, which is in turn
bit-exact to numpy's *sequential* float32 accumulation over 4000+ random rows. Sequential
fp32 agrees with numpy BLAS `@` on **~99.9 % of output lanes**; the ~0.1 % that differ do so
by a few ULP on long reductions — comfortably inside the FP16 path's documented
`rel_err < 5e-3` tolerance, an inherent reduction-order artifact rather than a correctness
gap. (The `fp32_to_fp16` rounder alone was checked bit-exact to `numpy.float16` over 200 k
values, including the overflow-to-inf and subnormal round-half boundaries.)

## Interface (house streaming style — identical to `mate_pv`)

One token per clock, `s_valid=1`; scalar fp16 A-code on `a_data`, the N-wide packed fp16
V-row on `v_data`; `s_last=1` on the final token. `c_valid` pulses when the row's Σ is
ready, with the N fp16 results on `c_data`; accumulators auto-reset (to +0.0) for the next
row. **Pipelined:** the per-lane fp16×fp16→fp32 product is registered (`prod_reg`) between
the multiply and the fp32 accumulate — same split as the INT8 tile — so the result is
emitted **2 cycles after `s_last`** (value identical, just delayed; downstream uses the
`c_valid` handshake). Throughput 1 token/cycle. `N` = head-dim lane count (default 8).

The one interface difference from `mate_pv`: operands and results are **16-bit fp16** (INT8
tile: 8-bit codes in, int32 out), because the FP16 contract rounds the result back to fp16.
The internal accumulator is 32-bit fp32.

## Verification

- **Bit-exact:** `make sim_mate_fp16` — `mate_pv_fp16` vs the sequential-fp32 golden on **17
  rows, 17/17, 0 errors**, covering: zeros (+0/−0), signed cancellation to exact 0,
  fp16-subnormal inputs, a 520- and a 600-token row (long fp32 reduction), overflow-to-±inf
  (Σ > 65 504 → fp16 inf), inf/NaN operand propagation (inf+finite, 0·inf → NaN), and a
  subnormal-**output** row (products in the 2⁻²⁴…2⁻¹⁴ band). An additional randomized
  stress of **1000 rows** (K up to 600, random incl. subnormals) also passes 0 errors.
- **Synthesis:** Yosys — **315 FFs (N=4) / 627 FFs (N=8)**, no latches (asserted
  `t:$dlatch`-free). FF count is essentially identical to the INT8 tile (INT8 N=4 = 323):
  the FP16 tile's wider `prod_reg` (fp32, +16b/lane) is offset by its narrower `c_data`
  (fp16, −16b/lane).

## GDSII — Sky130A (N=4 proxy)

`openlane/mate_pv_fp16/` reaches GDSII (LibreLane 3.0.5) with **all six sign-off checks
zero** (setup / hold / DRC / LVS / antenna / Max-Cap) — see `openlane/mate_pv_fp16/README.md`
for each metric's derivation and the rough 16 nm (TSMC N16) estimates. Clock **85 ns
(11.8 MHz)**; the physical run synthesizes `N=4` lanes (proxy, like the INT8 tile).
Committed outputs (`gds` + `png` + signoff metrics json) are under
`openlane/mate_pv_fp16/results/`.

### Why so much slower than the INT8 tile (85 ns vs 14 ns)

The accumulator is a **single-cycle feedback recurrence** — `acc ← fp32_add(acc, prod)` —
so the full IEEE-754 fp32 adder (align barrel-shift → 28-bit add/sub → leading-zero
normalise barrel-shift → RTNE) sits on the critical path and **cannot be pipelined without
dropping below 1 token/cycle**. That fp32 adder is ~4× the logic depth of the INT8 tile's
int32 add, which is the whole fmax gap. (The normaliser is priority-encoded + single
barrel-shift, not a shift-by-1 ripple — that alone took the Sky130 path from ~121 ns to
~63 ns pre-route.)

## FP16-vs-INT8 delta — the number the docs called "TBD"

Both tiles, **N=4 proxy, Sky130A, slow-corner sign-off** (INT8 from `openlane/mate_pv`):

| metric | INT8 `mate_pv` | FP16 `mate_pv_fp16` | delta |
|---|---|---|---|
| std-cell area | 39,948 µm² (4,508 cells) | 111,462 µm² (14,583 cells) | **≈ 2.8× area, 3.2× cells** |
| sequential (FF) | 323 | 315 | **−8 (≈ equal)** |
| die area | 75,660 µm² | 221,987 µm² | ≈ 2.9× |
| fmax (signoff clock) | 71.4 MHz (14 ns) | 11.8 MHz (85 ns) | **≈ 6.1× slower** |
| total power | 5.53 mW @ 71.4 MHz | 10.44 mW @ 11.8 MHz | 1.9× (at each block's clock) |
| energy / token-cycle | 0.077 nJ | 0.89 nJ | **≈ 11.5× energy/op** |

**Headline:** the FP16 P·V tile costs **≈ 2.8× the std-cell area, ≈ 3.2× the cell count, and
≈ 11.5× the energy per MAC** of the INT8 tile, runs **≈ 6× slower** (single-cycle fp32
accumulate), and uses **essentially the same number of flip-flops**. Power in raw mW is only
1.9× because the FP16 tile is clocked 6× slower; frequency-normalised (energy per operation)
is the fair comparison and shows the ≈ 11.5× cost — consistent in direction with the reference
cost model's 2.5 pJ (fp16) vs 0.5 pJ (int8) per-MAC ratio, larger in silicon because of the
full fp32 accumulate + fp16 round hardware.

> All figures are the **N=4 physical proxy**; the functional block is N=8 (≈ 2× area / cells
> / power, same fmax). 130 nm Sky130A, 1.8 V, slow corner — a proxy for the TSMC N16 target;
> see the openlane README for 16 nm extrapolation with disclaimers.

## Still open

- fmax is bounded by the single-cycle fp32 accumulate. A higher clock would need a
  redundant/carry-save accumulator or a multi-cycle accumulate with < 1 token/cycle
  throughput — a larger redesign, out of scope for measuring the area/power delta.
