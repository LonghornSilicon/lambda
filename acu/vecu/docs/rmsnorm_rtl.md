# `rmsnorm` — synthesizable VecU RMSNorm slice (RTL)

**Status:** RTL complete + micro-sequenced, **bit-exact to the RMSNorm LUT golden**
(`rtl/tb/tb_rmsnorm.sv`: 8 committed corner rows + 300-row random stress, **0
mismatches**), **Yosys-clean** (951 FFs at D=16, **no latches** — `t:$dlatch = 0`
asserted). **GF180 hardened to GDSII** (LibreLane 3.0.5, clean 5-check sign-off —
see below); Sky130 in progress.
**Home:** `rtl/rmsnorm.sv` (+ `rtl/tb/tb_rmsnorm.sv`, `rtl/tb/gen_rmsnorm_vectors.py`,
golden `sw/reference_model/rmsnorm_ref.py`).
**One line:** RMS-normalizes a decode hidden-state vector with a learned per-channel
gain — the pre-attention / pre-MLP norm the chip-top raw path needs.

## What it computes (the hardware algorithm)

For a `D`-element fp16 vector `x` with a learned per-channel gain `g`:

```
y_i = x_i * rsqrt(mean(x^2) + eps) * g_i
```

- **fp32 sum-of-squares.** Each `x_i` is widened to fp32, squared with `fp32_mul`,
  and summed in an fp32 accumulator (`fp32_add`) — like the MatE accumulator, so the
  error is the rsqrt LUT, not fp16 sum drift. `mean = ss * (1/D)` (`1/D` a folded
  fp32 constant), `v = mean + eps` (`eps = 2^-16`).
- **rsqrt LUT + linear interp.** Decompose `v = 2^E * f`, `f ∈ [1,2)`. A 64-entry
  LUT `R[j] = round_fp16(1/sqrt(1 + j/64))` (top 6 mantissa bits index it, low 17
  bits interpolate) gives `rsqrt(f)`; the exponent is applied as
  `2^(-floor(E/2))`, with an extra `1/sqrt(2)` when `E` is odd
  (`rsqrt(v) = rsqrt(f) * 2^(-E/2)`). The fp16 rounding of the rsqrt table **is**
  the LUT approximation. (Subnormal/zero `v` clamps to `rsqrt(eps)`.)
- **fp32 emit, one fp16 rounding.** `y_i = round_fp16( x_i * scale * g_i )`, both
  products in fp32.

The IEEE primitives (`fp16↔fp32`, `fp32_add/sub/mul`, `fp32_to_fp16`) are the same
datapath as `mate_pv_fp16` / `vecu_softmax`.

## Accuracy (the two bars, kept separate)

| bar | what it proves | result |
|---|---|---|
| **(a) RTL bit-exact to the LUT-golden** | the RTL matches its spec exactly | `make sim_rmsnorm`: 8 corner rows + 300 random, **0 mismatches** |
| **(b) LUT-golden vs exact fp64 RMSNorm** | the LUT is accurate enough | **max rel-err ≈ 7.9e-4** to the vector peak; rsqrt fp16-quant alone ≤ **2.4e-4** over [1,2). Run `python sw/reference_model/rmsnorm_ref.py`. |

Well within the `rel_err < 5e-3` bar.

## Interface (house streaming style)

- **LOAD** — one fp16 element per clock on `s_data` and its gain on `g_data`,
  `s_valid=1`, `s_last=1` on the final element (`D` total). Both are buffered.
- **EMIT** — after the pipeline finishes, the block streams the `D` normalized
  elements on `y_data`/`y_valid`, `y_last` on the final one; the consumer waits on
  the `y_valid` handshake.
- `busy` high from the first element until the last output.

## Timing structure — micro-sequenced

At most **one fp32 op per register-to-register path** (like `vecu_softmax`), so the
GF180 ss corner closes with **normal** resizing:

| phase | per-item cycle sequence |
|---|---|
| ACC (per elem, 2 cyc)  | `sq = x*x` · `ss += sq` |
| PREP (rsqrt sequence)  | `mean = ss/D` · `v = mean+eps` · decompose+LUT · `diff = hi-lo` · `term = frac*diff` · `r = lo+term` · `r *= 2^(-k)` · `scale = odd ? r*1/√2 : r` |
| EMIT (per elem, 3 cyc) | `t1 = x*scale` · `t2 = t1*g` · `y = fp16(t2)` |

Op sequence is fixed → bit-exact to `rmsnorm_ref.py`.

## Synthesis

Yosys `synth_rmsnorm.ys` (`make synth_rmsnorm`): **951 FFs** (D=16; the x + g
buffers dominate), **no latches** (`select -assert-count 0 t:$dlatch`), ~88.4k
NAND2-equivalent cells (abc `-fast -g NAND` lower bound). Function locals are
default-initialised so no transient latch inference.

## Harden — GDSII sign-off

### GF180MCU (gf180mcuD, LibreLane 3.0.5 Classic) — CLOSED
Results: `pdk/gf180/librelane/results/rmsnorm/` (GDS, metrics, render, `SIGNOFF.md`).

| | value |
|---|---|
| Die area | 1095180 µm² (1.095 mm²), 69989 insts, 48.9 % util |
| Clock (loose) | 260 ns; fmax(ss) ~6.7 MHz (crit path ~149 ns) |
| Setup / Hold WNS | 0 / 0 (all corners) |
| Magic DRC / route DRC | 0 / 0 |
| LVS | clean (netgen "Circuits match uniquely") |
| Antenna | 0 nets / 0 pins |
| Residual (ss register-array) | max_slew 2857, max_cap 7 — noted, not gating |

Five hard sign-off checks (setup/hold/DRC/LVS/antenna) all zero; the ss-corner
max-slew/max-cap residuals are the known fp32 register-array item (as `vecu_softmax`).

### Sky130A (OpenLane/LibreLane 3.0.5) — in progress
`pdk/sky130/openlane/rmsnorm/config.json`. Loosened to FP_CORE_UTIL 28 /
PL_TARGET_DENSITY_PCT 38 + GRT_ALLOW_CONGESTION (the fp32+rsqrt-LUT datapath congests
at the tighter Sky130 tracks). Clock 105 ns start (one-fp32-op path, like
`vecu_softmax`).

## Scope / still open

- `1/D` is folded as an fp32 constant for the default `D=16` (`0x3D800000`).
  Changing `D` requires regenerating `INV_D32` (and the golden's `f16_from_real(1/D)`
  rounding must match). `eps = 2^-16` is fixed (exactly representable).
- The gain `g` is streamed in alongside `x` here (self-contained/testable); in the
  full chip it comes from a weight ROM/SRAM lane.
