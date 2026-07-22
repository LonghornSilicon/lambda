# `mate_pv` — synthesizable INT8 P·V MAC tile (RTL)

**Status:** RTL complete + pipelined, bit-exact to the reference, Yosys-clean (643 FFs, no latches), GDSII-signed-off on Sky130A at 71 MHz.
**Home:** `rtl/mate_pv.sv` (+ `rtl/tb/tb_mate_pv.sv`, `rtl/tb/gen_mate_pv_vectors.py`).
**One line:** the token-reduction vector-MAC core of the MatE matrix engine, in
hand-written synthesizable Verilog for the Sky130 flow — the piece of the P·V datapath
that was previously **HLS-only**.

## Why this exists

The attention output stage `o[n] = Σ_t A[t]·V̂[t][n]` (P·V) was the one block with no
synthesizable RTL — the real MatE (`architecture/src/blocks/mate`) is a SystemC/Stratus
HLS project targeting Cadence N16FFC, so the cross-block cosim could not exercise the
actual accumulation between the KVE (rotated V̂) and the `wht_inverse_out` unspin. This
block fills that gap: a real INT8 MAC with the correct accumulator width, in the same
OpenLane/Sky130 flow as the other blocks, so the end-to-end cosim runs through a genuine
P·V accumulation rather than copying V̂ straight across.

## What it computes (bit-exact contract)

Signed **INT8 × INT8 → INT32**, no saturation — bit-exact to the ACU MAC-array reference
`sw/reference_model/mac_array_ref.{hpp,py}` `matmul_int8` for M=1 (one attention row):

```
o[n] = Σ_t  A[t] · V[t][n]        A,V ∈ int8   →   o ∈ int32
```

This is the ACU's **INT8 tile** (`precision_controller.d_fp16 == 0`). The FP16 tile is
tolerance-only in the reference (`rel_err < 5e-3`, see `MAC_ARRAY_DESIGN.md`) and is not
part of this integer datapath; only INT8 carries a bit-exact contract, so that is what the
RTL implements.

## Why INT32 (not INT24)

The P·V tile reduces over the **token** dimension, so the accumulator width scales with
**context length**, not hidden dim. A maximally-flat causal row of length L drives every
code to ±127, so `|acc| ≤ 127·127·L → 14+⌈log₂L⌉ bits`: INT24 overflows past ~520 tokens,
INT32 covers ~133k. (The hidden-dim reductions — W4A8 GEMM, Q·Kᵀ — fit INT24; only the
token-reduction P·V accumulator needs INT32.) See `analysis/pv_accumulator_width.py` and
the `arch.yml` accumulator rationale. The testbench's flat-attention corner (K=520, all
±127, max|acc| = 8,387,080) is the empirical proof this overflows INT24 territory and needs
the INT32 register.

## Interface (house streaming style)

One token per clock, `s_valid=1`; scalar A-code on `a_data`, the N-wide packed V-row on
`v_data`; `s_last=1` on the final token. `c_valid` pulses when the row's Σ is ready, with
the N int32 results on `c_data`; accumulators auto-reset for the next row. **Pipelined:**
the per-lane product is registered (`prod_reg`) between the multiply and the accumulate, so
the result is emitted **2 cycles after `s_last`** (the value is identical — same Σ, just
delayed — so still bit-exact; downstream uses the `c_valid` handshake). Throughput 1
token/cycle. `N` is the head-dim lane count (default 8; the cosim uses the full head dim).

## Verification

- **Bit-exact:** `make sim_mate` — `mate_pv` vs `matmul_int8` on 7 rows incl. the K=520
  flat corner: **7/7, 0 errors**. Golden computed in pure Python (integer `Σ A·V`, provably
  identical to the reference's `int32` matmul) so it runs on the bare venv (no numpy).
- **Synthesis:** `yosys synth -flatten` — **643 FFs** (8 acc×32 + 8 out×32 + 8 prod_reg×16
  + v1 + last1 + c_valid), no latches. This is the `expected-ff-count` for the CI synth gate.

## GDSII — clean at 71 MHz (Sky130A)

`openlane/mate_pv/` reaches GDSII with **all six sign-off checks zero** (setup / hold /
DRC / LVS / antenna / Max-Cap), committed under `openlane/mate_pv/results/`. Clock 14 ns
(**71 MHz** — nearly 2× the un-pipelined 40 MHz, thanks to `prod_reg`); the physical run
synthesizes `N=4` lanes (proxy, like TIU's `N_SLOTS=4`). The openlane README documents each
sign-off metric's derivation and rough 16 nm (TSMC N16) estimates with disclaimers.

## Cross-block cosim — true end-to-end

Vendored into the `chip/verif` cross-block cosim (`make -C chip/verif cosim`) as the real P·V
accumulation between the KVE's rotated V̂ and `wht_inverse_out`: int32 bit-exact vs
`matmul_int8`, and the full KVE → P·V → inverse chain reconstructs the reference
attention output to ~0.26 %.

## Still open

- Nothing blocking. Further fmax would come from a wider pipeline (register the add too)
  or a real N16 PDK; the Sky130 proxy is signed off clean at 71 MHz.
