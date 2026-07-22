# `mate_qkt` — synthesizable Q·Kᵀ decode-scoring engine (RTL)

**Status:** RTL complete + pipelined, bit-exact to the reference's FP16 Q·Kᵀ semantics, Yosys-clean (627 FFs N=8, no latches). GF180MCU hardening is Phase 4 (chipathon shuttle PDK); this phase is functional RTL + cosim integration.
**Home:** `rtl/mate_qkt.sv` (+ `rtl/tb/tb_mate_qkt.sv`, `rtl/tb/gen_mate_qkt_vectors.py`).
**One line:** the decode-time score generator of the MatE matrix engine — computes one query's attention logits against the cached keys in hand-written synthesizable Verilog, replacing the reference stand-in in the cross-block cosim.

## Why this exists

The cross-block cosim (`architecture/rtl/tb/tb_chip_cosim.sv`) fed the precision
controller a **hard-coded** score row — the Q·Kᵀ scores were a reference stand-in,
not RTL. This block makes those scores real: a genuine Q·Kᵀ reduction feeds the
precision gate, so the decode path `Q·Kᵀ → gate → P·V` runs through actual score
hardware end to end.

## What it computes (bit-exact contract)

For decode there is ONE query token, scored against L cached keys:

```
score[l] = round_fp16( Σ_d  Q[d] · K[l][d] )      Q ∈ int8, K ∈ fp16 → score ∈ fp16
```

Realises the ACU MAC-array reference `sw/reference_model/mac_array_ref.{hpp,py}`
`matmul_fp16`, invoked as `matmul_fp16(Q, K.T)` in `integration_example.py`:

- **INT8 Q → fp16, exact.** Q arrives per-tensor/-tile symmetric-quantized
  (|q| ≤ 127); any integer with |q| ≤ 2048 is exactly representable in binary16,
  so the promotion never loses a bit.
- **K is fp16, per-channel-dequantized by the KVE key path**
  (`cq_dequant_f16(code, scale) = round_fp16(code · fp16 per-channel scale)`).
  MatE consumes it directly as fp16.
- **fp16 × fp16 → fp32, exact**, then the head-dim (D) reduction is accumulated in
  **fp32 with round-to-nearest-even at every add**, and the score is **rounded to
  fp16 exactly once** on emit (RTNE, overflow → inf, underflow → subnormal/0).

The fp16 datapath (`fp16_mul` / `fp32_add` / `fp32_to_fp16`) is **byte-identical**
to `mate_pv_fp16` — the same signed-off IEEE-754 arithmetic, reused for the Q·Kᵀ
reduction. The only new logic is `int8_to_fp16` at the query input.

### Accumulation order

Sequential in head-dim channel order — the faithful streaming-MAC realisation of
the reference's "fp32 accumulate, round-to-fp16-once" semantics. This RTL is
bit-exact to a sequential-fp32 golden, which is itself bit-exact to numpy's
sequential float32 over 1500+ random rows (D ≤ 520). numpy's BLAS `@` sums in a
different pairwise order and so agrees to a few ULP — within the FP16 path's
documented `rel_err < 5e-3` (NOT used as the bit-exact golden here).

## Interface (house streaming style — same as `mate_pv` / `mate_pv_fp16`)

One **head-dim channel** per clock, `s_valid=1`: the query's scalar INT8 code
`Q[d]` on `a_data`, and the d-th channel of every key (the N-wide packed fp16
vector `K[0..N-1][d]`) on `k_data`; `s_last=1` on the final channel (d = D-1).
`c_valid` pulses when the L scores are ready, with the N fp16 scores on `c_data`;
accumulators auto-reset for the next query. **Pipelined:** the per-key
int8→fp16→fp32 product is registered (`prod_reg`) between multiply and accumulate,
so scores emerge **2 cycles after `s_last`** (value identical, just delayed).
`N = L` is the number of cached keys scored in parallel (lanes); the reduction
runs over the streamed head-dim D (dynamic, via `s_last`).

## Verification

- **Bit-exact:** `make sim_mate_qkt` — `mate_qkt` vs the sequential-fp32 golden
  across **three key-counts, 12 rows each, 0 errors**: **L=1** (single key), **L=8**
  (rich head-dim corners incl. the D=520 long reduction), **L=520** (long-context /
  wide-lane). Corner cases: D=1, long D reduction, zero keys, zero query, peaked
  key (one score dominates), subnormal fp16 keys, signed cancellation. Golden is
  pure Python (bit-manipulation fp16/fp32, verified vs numpy sequential float32) so
  the TB needs no numpy.
- **Synthesis:** Yosys — **627 FFs (N=8)** (8×32 fp32 acc + 8×32 fp32 prod_reg +
  8×16 fp16 c_data + 3 control, minus a few constant-optimised bits), **no
  latches** (`t:$dlatch`-free assertion). Identical FF count to `mate_pv_fp16`.

## Cross-block cosim — real Q·Kᵀ scores

Vendored into the `architecture` rtl-branch cosim (`make` in `architecture/rtl`)
as **BLOCK 1's** score source: the precision controller is now gated on scores
**computed by `mate_qkt`** from a query Q and the KVE-dequantized keys, replacing
the hard-coded stand-in. The score row is checked against the sequential-fp32
golden within `rel_err < 5e-3`; the whole cosim stays green (`ALL BLOCKS PASS`).

## Physical

GF180MCU is the chipathon shuttle PDK, so full hardening (place-and-route,
sign-off) is Phase 4 on GF180 — not Sky130. This phase ships functional RTL with a
Yosys smoke-synth (latch-free, ~627 FFs) only.

## Still open

- fmax is bounded by the same single-cycle fp32-accumulate recurrence as
  `mate_pv_fp16` (the fp32 adder is the critical path); Phase 4 GF180 P&R will set
  the real clock.
