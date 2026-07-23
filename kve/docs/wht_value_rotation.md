# CQ-3-rot — Walsh-Hadamard-rotated INT3 values

**Status:** integrated — the reference codec, accuracy, this spec, **and the RTL**
(`wht_unit`, `cq_wht_value`, `wht_inverse_out`, `cq_value_path_wht`) with its bit-exact proof
all live in this block (`kve/rtl/`) on `main`. There is no separate `rtl` branch (monorepo
migration 2026-07-22). **Idea and result:** Abhiram Bandi + Chaithu Talasila.
**One line:** rotate each per-token value row by a fixed Walsh-Hadamard transform before
quantizing, so values drop from 4 to a **flat, uniform 3 bits** (no calibration) — keys
untouched.

## Why 3-bit works only with the rotation

Naive INT3 values collapse for the same reason naive INT4 does: one high-magnitude channel
in a token's value vector hijacks the per-token amax scale, starving the other D−1 dims.
Spinning the row by a Walsh-Hadamard transform (WHT) spreads that channel's energy across
all D dims, so no single channel sets the scale; the quantizer then sees a well-conditioned,
roughly-flat row. On read the WHT is undone (it is self-inverse). Measured on HellaSwag,
n=1000: CQ-3-rot is within **~0.005** of FP16 — Qwen2-1.5B −0.004, 0.5B −0.007,
Llama-3.2-1B +0.004 — while naive INT3 craters like naive INT4.

**Keys are untouched.** The rotation only ever mixes a token's own D dims. Keys are quantized
*per channel over a token group* (the GQA-critical axis); rotating them would scramble that
axis — exactly the move that sank TurboQuant+. So the rotation lives only on the per-token
value path, where it composes cleanly. (The post-mortem: TurboQuant's *value* rotation was
always fine; only its *key* rotation was the problem. We keep the innocent half.)

We evaluated adopting more of TurboQuant (its Lloyd-Max codebook, L2 normalization, dense
random-orthogonal rotation) and it bought **zero or negative** accuracy for real hardware
cost — the fixed Walsh-Hadamard + our existing amax/uniform INT3 quant is the whole win.
See `analysis/WHT_INTEGRATION_FINDINGS.md`.

## The codec (bit-exact definition)

Power-of-two head dim D (64 on 0.5B/Llama, 128 on 1.5B). fp16 throughout, matching the base
ChannelQuant codec's arithmetic.

- **Compress (per token):** `r = FWHT_raw(v)` — raw fp16 butterfly, add/sub only, **no
  normalization** (the per-token amax absorbs the √D magnitude, so the INT3 codes are
  identical to the orthonormal transform). Then the usual per-token amax → fp16 scale →
  INT3 quant. Pack 8 codes → 3 bytes (`pack_int3`, exactly **3.000 bits/value**).
- **Decompress:** dequant each code to fp16 → `FWHT_raw` (inverse) → **×(1/D)**, an exact
  2⁻ᵏ shift → fp32. No √D, no re-rounding.

Reference: `sw/reference_model/channelquant_ref.hpp` (`compress/decompress_values_wht3`,
header-only) and the numpy mirror `analysis/wht_value_pyref.py`. The standalone ChannelQuant
research repo carries the same codec (`reference/channelquant_ref.py`,
`compress/decompress_values_wht`) and is bit-exact to this one.

## Hardware — Path B (sum in rotated space, unspin once)

Because the WHT is linear and the attention output is `o = Σ_t A[t]·V̂[t]`:

1. **KVE write:** rotate the value row (`wht_unit`, forward) then quant/pack INT3. Store
   rotated 3-bit codes + the fp16 per-token scale.
2. **KVE read:** dequant to the **rotated** V̂ (fp16) — *no* inverse WHT in the KVE.
3. **MatE P·V:** accumulate `Σ_t A[t]·V̂rot[t]` in rotated space (INT8 or FP16 tile, the ACU
   gate unchanged — rotated values are flatter, so the INT8 path is if anything *more*
   accurate).
4. **MatE output:** apply the inverse WHT **once** on the D-vector result (`wht_inverse_out`
   = `wht_unit` + ×1/D), in fp on the accumulator output (**not** inside the INT32 P·V
   integer accumulator — the orthonormal inverse can grow an element by up to √D, so convert
   to fp first). This is O(D·log D) per (query, head), not per cached token.

RTL: `wht_unit.sv` (structural fp16 add/sub butterfly, self-inverse), `wht_inverse_out.sv`
(the MatE output stage), and `cq_wht_value.sv` (the full per-token codec — the bit-exact
validation module; Path B splits its forward half into the KVE and its inverse half into
MatE, identical by linearity).

## Verification

- **Reference parity:** C++ `channelquant_ref.hpp` == numpy mirror, **348,160 / 348,160**
  elements bit-exact across 44 real-Qwen slices (D=64 + D=128).
- **RTL bit-exact:** `make sim_wht` — `cq_wht_value` vs the reference V̂ on real Qwen:
  **D=128 245,760/245,760, D=64 102,400/102,400** (= 348,160/348,160). The `wht_unit`
  butterfly is separately checked vs numpy (10,240/10,240 fp16 words).
- **Accuracy:** `analysis/wht_ref_accuracy.py` (the exact reference codec, in-model): −0.004
  / −0.007 / +0.004 as above.

## Compression

Values 4 → 3 bits takes the codec from ~4.2 to **~3.3 bits/value** (~3.8× → **~4.8×** KV
compression); stacked with the Token Importance Unit's ~4× eviction, **~15× → ~19×**
effective KV footprint. Side-channel: one fp16 per-token scale, unchanged.

## Interaction with the TIU (retired lever)

Flat uniform 3-bit values **retire the TIU `tier_keep` value-demotion** (CQ-8-vs-CQ-4 per
token) — there is no per-token value bit-width to select anymore. The TIU keeps its
keep/evict role; the value-precision role of `tier_keep` is dropped (see the
`token-importance-unit` tier-handshake doc).

## Synthesis tier — `wht_unit_syn` (done)

`fp16_addsub_syn.sv` is a synthesizable IEEE-754 half add/subtract (no `real` math —
align / add / normalize / round-half-even, subnormals handled), **bit-exact to the
behavioral oracle** `cq_fp_pkg::fp16_add`/`fp16_sub` over a 375,403-pair random sweep +
directed edge cases. `wht_unit_syn.sv` wires those cores into the same butterfly and is
**bit-identical to the behavioral `wht_unit`** on real Qwen rows (D=128 10,240/10,240,
D=64 5,120/5,120). `make sim_wht_syn`. This is the tape-out-ready form of the butterfly
(cf. `cq_units_syn.sv` for the base codec's quant/dequant cores).

## Path B (done)

`cq_value_path_wht.sv` is the KVE side of Path B: forward-WHT the row, per-token
amax + INT3 quant, and on read emit the **rotated** fp16 reconstruction per channel — the
inverse WHT + x(1/D) runs ONCE on the P·V output (`wht_inverse_out`), not per token.
Chaining `cq_value_path_wht -> wht_inverse_out` is **bit-exact to the full reference V̂** on
real Qwen (`make sim_wht_pathb`: D=128 10,240/10,240, D=64 5,120/5,120). So the
store-rotated / unspin-once dataflow is proven equivalent to the full codec.

## Still open

- Fold Path B into the pipelined streaming schedule of the top FSM (a throughput
  optimization; the arithmetic and dataflow are proven).
- Chip-hub docs (the monorepo-root `arch.yml` codec-of-record + MatE inverse-WHT stage, org
  profile README compression numbers) — update in the monorepo on `main`.
