# `vecu_softmax` — synthesizable VecU decode online-softmax slice (RTL)

**Status:** RTL complete + **micro-sequenced (rebalanced timing)**, bit-exact to the LUT online-softmax golden, Yosys-clean (1062 FFs at N=16, no latches). **Signed off on Sky130A** (this micro-sequenced revision, commit `2c458aa`): 105 ns / 9.5 MHz, all 9 IPVT corners clean (DRC/LVS/antenna/max-cap 0; residual ss slew only) — `pdk/sky130/openlane/vecu_softmax/README.md`. Also **GF180-hardened** (multi-cycle, 260 ns, ss +60.9 ns; `chip/pdk/gf180/docs/gf180_gls_report.md` §1).
**Home:** `rtl/vecu_softmax.sv` (+ `rtl/tb/tb_vecu_softmax.sv`, `rtl/tb/gen_vecu_softmax_vectors.py`, golden `sw/reference_model/vecu_softmax_ref.py`).
**One line:** the decode-time softmax of the VecU vector unit — turns a row of Q·Kᵀ scores into attention weights in hand-written synthesizable Verilog, replacing the last reference stand-in in the cross-block cosim.

## Why this exists

The cross-block cosim fed the P·V tile attention **weights** taken from the
reference. This block makes them real RTL: the decode attention pass
`Q·Kᵀ → softmax → P·V` now runs entirely on hardware (mate_qkt → vecu_softmax →
mate_pv/mate_pv_fp16), so the cosim's only inputs are Q/K/V.

## What it computes (the hardware algorithm — not `np.softmax`)

For one decode row of L fp16 scores it emits the L fp16 attention weights
`p_j = exp(s_j - m) / Σ_k exp(s_k - m)`, modelling the actual VecU microcode
(architecture/dataflow_walkthrough.md Stage 9; paper/lambda.tex §VecU):

- **64-entry exp LUT, x ∈ [-16, 0], linear interpolation.** `E[i] =
  round_fp16(exp(-0.25·i))`, i = 0..63 (`E[0] = 1.0` anchors the running max
  exactly); the last interval interpolates toward `exp(-16)`; `x < -16` clamps 0.
  The index is a fixed-point `floor(-x·4096)` (top 6 bits → entry, low 10 →
  fraction); the interpolation runs in fp32.
- **Online recurrence with the `exp(m_old-m_new)` rescale** (Milakov &
  Gimelshein 2018; the FlashAttention running-max/running-sum core): per row keep
  `m` (running max, fp16) and `ℓ` (running sum-of-exp, **fp32 accumulator**); on
  each score `m_new = max(m, s); ℓ = ℓ·exp(m-m_new) + exp(s-m_new)`.
- **Emit** `p_j = exp(s_j - m_final) · (1/ℓ_final)` — reciprocal in fp16, product
  rounded to fp16.

`ℓ` and the exp/interp run in fp32 (like the MatE accumulator) so the block's
error is the LUT approximation, not fp16 sum drift. The fp16/fp32 primitives
(`fp16↔fp32`, `fp32_add`, `fp32_mul`, `fp32_to_fp16`) are the same IEEE datapath
as `mate_pv_fp16`/`mate_qkt`; `fp16_div` (the `1/ℓ` reciprocal) is the one new op.

## Two comparison bars (the LUT-accuracy question, kept separate)

| bar | what it proves | result |
|---|---|---|
| **(a) RTL bit-exact to the LUT-golden** | the RTL matches its spec exactly | `make sim_vecu_softmax`: 14 committed rows + 400-row random stress, **0 mismatches** |
| **(b) LUT-golden vs exact fp64 softmax** | the LUT is accurate enough | **max rel-err ≈ 2.0 %** to the peak weight (worst case: long, wide, near-uniform rows). Run `python sw/reference_model/vecu_softmax_ref.py`. |

The ≈2 % is inherent to a 64-entry linear-interp exp LUT over the full [-16, 0]
range (worst near-uniform rows, where the interpolation errors do not cancel in
the softmax ratio); the dominant term is the ~0.8 % linear-interp error of `exp`.
This is the number the cosim's P·V-vs-reference tolerance is set from.

## Interface (house streaming style)

- **LOAD** — one fp16 score per clock on `s_data`, `s_valid=1`, `s_last=1` on the
  final score. Scores are buffered (depth `N`).
- **EMIT** — after the compute micro-sequence finishes the block streams the L
  weights: `w_valid` pulses (one weight every ~8 cycles) with the fp16 weight on
  `w_data`, `w_last` on the final one. The consumer just waits on the `w_valid`
  handshake, so the (data-independent) latency is transparent — the cosim/TB waits
  already tolerate it.
- `busy` is high from the first score until the last weight. `N` is the max row
  length (buffer depth); a single N-elaborated block handles any L ≤ N.

## Timing structure — micro-sequenced (closes GF180 ss with normal resizing)

Two revisions closed / then rebalanced the GF180 slow corner:

1. **Un-pipelined** ran the whole `fp16 → exp-LUT → fp32 → fp16` chain in one cycle
   (~366 ns post-route at `ss_125C_4v50`, missing setup by −26.5 ns).
2. **3-stage pipeline** (superseded) split it into 3 stages and closed ss (+19.2 ns),
   but the split was **uneven** — the longest stage held a full `fp32 mul + add`
   (or two `fp32_sub`s), ~263 ns at ss, forcing a 300 ns clock **plus an aggressive
   resize that ~2×'d the cell count (55k → 101k, 1.49 mm²)**.
3. **Micro-sequenced (this revision)** rebalances so every register-to-register path
   holds **at most one fp32 op** — evenly ~½ the 3-stage's longest stage:

Scores are buffered on load (1/cycle), then the recurrence and the emit run as
micro-sequences executing one fp32 op per cycle, registering the intermediate each
cycle:

| phase | per-item cycle sequence (one fp32 op each) |
|---|---|
| COMPUTE (per score) | read · `max`+`sub` · index/LUT · `diff sub` · `interp mul` · `interp add` · `rescale mul` · `accumulate add` |
| EMIT (per weight)   | read · `sub` · index/LUT · `diff sub` · `interp mul` · `interp add` · `·(1/ℓ) mul` · `round-to-fp16` |

Because the op sequence and rounding are **unchanged**, the result is **identical**
(bit-exact to the same golden). The recurrence is single-issue (no overlap hazards)
and the intermediate registers are reused across scores/weights. Added latency
(~8 cycles/score compute + ~8 cycles/weight emit) is transparent via the `w_valid`
handshake — decode is latency-tolerant.

**What this buys (and what it does NOT — area was not reclaimed):** the longest path is
now **one fp32 op** instead of two, so ss closes at a faster clock with **normal,
non-aggressive resizing**. The predicted area reclaim did **not** materialize: the GF180
re-harden **measured** the multi-cycle datapath at **111,253 cells / 1.64 mm²**, ~10 %
*larger* than the 3-stage's 101,236 cells / 1.49 mm² (the FSM + reused-intermediate
registers + score buffer offset the fewer parallel fp32 units). The 1.49 mm² was largely
**inherent, not resize bloat** (`chip/pdk/gf180/docs/gf180_gls_report.md` §1). The real win
is timing robustness at normal effort + a tighter clock, not area. Sequential cost:
**1062 FFs at N=16** (256 score buffer + fp32 ℓ / m / 1/ℓ + one reused set of
COMPUTE + EMIT intermediates + pointers/steps) — FFs are small; the combinational
resize-avoidance dominates the net area. The one-fp32-op path is the floor without
pipelining *inside* the adder/multiplier (a deeper, higher-FF change reserved for a
later round if ss needs an even faster clock).

## Verification

- **Bit-exact (bar a):** `make sim_vecu_softmax` — **14 rows, 0 errors** across
  corners: L=1, L=520 long context, peaked, near-uniform, all-negative,
  monotone-increasing (forces the rescale every step), subnormal scores, and a
  far-below-max score that clamps to 0. Plus a 400-row random stress, 0 errors.
  Bit-exactness is unchanged by the timing rebalance. Golden is pure Python
  (bit-manipulation fp16/fp32) so the TB needs no numpy.
- **Synthesis:** Yosys — **1062 FFs (N=16)** (256 score buffer + fp32 ℓ / m / 1/ℓ +
  one reused set of COMPUTE + EMIT intermediates + pointers/steps), **no latches**
  (`t:$dlatch`-free assertion; function locals default-initialised so no transient
  latch-inference either).

## Cross-block cosim — closes Q·Kᵀ → softmax → P·V

Vendored into the `architecture` rtl-branch cosim: the attention weights feeding
the P·V tile are now **computed by `vecu_softmax`** from the `mate_qkt` scores,
replacing the reference-supplied weights. The weights are checked against exact
softmax within the measured LUT tolerance, and the P·V output against the
reference attention within a tolerance set from the LUT error. Cosim stays green
(`ALL BLOCKS PASS`).

## Scope / still open

- This is the **decode-softmax slice only**. VecU's other microcode ops (RoPE,
  RMSNorm, SiLU, residual) are separate and pending — RoPE/RMSNorm are needed for
  the chip-top raw-Q/K path (the cosim's loaded Qwen tiles are already RoPE'd), so
  they are not part of this slice.
- fp_max is set by the single fp32-op-per-cycle micro-sequence. The real P&R clocks are
  now set: **Sky130A 105 ns / 9.5 MHz** (signed off, `pdk/sky130/openlane/vecu_softmax/`) and
  **GF180 260 ns** (ss +60.9 ns, `chip/pdk/gf180/docs/gf180_gls_report.md` §1).
