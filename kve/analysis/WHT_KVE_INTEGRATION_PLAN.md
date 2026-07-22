# WHT value rotation → ChannelQuant / KVE integration plan

**Status:** the value codec, docs, and RTL all live in this block. Remaining Phase-3 RTL (Path B streaming, synth adder) tracked in `rtl/`.
**Idea/result:** Abhiram Bandi + Chaithu Talasila. **Adopt the Walsh-Hadamard rotation
only** (A/B verdict, `WHT_INTEGRATION_FINDINGS.md`): rotate each per-token VALUE row, keep
ChannelQuant's existing amax + uniform quant at **3 bits**. Keys unchanged (CQ-4+). No
codebook, no L2-norm, no Haar, no QJL.

## The whole change in one sentence

Insert a fp16 Walsh-Hadamard butterfly in front of the **value** quantizer (and its
inverse on read), drop the value tier from 4-bit to 3-bit. Keys, APA gate, TIU eviction,
and the accumulators are untouched by the codec itself.

## Layer-by-layer changes

### 1. Reference codec (the authority) — `sw/reference_model/channelquant_ref.{hpp,cpp}` + `analysis/channelquant_hw.py`
- Add `fwht_f16(vec, D)`: orthonormal fp16 Walsh-Hadamard (log₂D butterfly stages of
  add/sub + one 1/√D scale). Symmetric ⇒ self-inverse ⇒ same unit forward/inverse.
- Value **encode**: `rotate → amax → scale → quant(3b)` (quant core already
  bit-parameterized: `qmax=(1<<(bits-1))-1`, so 3-bit = qmax 3). Value **decode**:
  `dequant → inverse-rotate`.
- Keep the double-precision mirror discipline (this is what gave us 696k-element
  bit-exactness). The fp16 WHT rounding must match RTL exactly.
- **3-bit packing:** need `pack_int3` (8 codes → 3 bytes) for the real memory win; a
  4-bit slot would waste the 4th bit and defeat the purpose. This is new packing code.

### 2. RTL — `kv-cache-engine/rtl/`
- **New `wht_unit.sv`:** fp16 Walsh-Hadamard over D (param 64/128), add/sub tree + 1/√D,
  pipelined log₂D stages. Same module for forward and inverse.
- **`cq_value_path.sv`:** insert `wht_unit` on the compress input. On decompress: today it
  emits one reconstructed channel per beat (combinational, `dec_idx`); the inverse-WHT
  mixes all D channels, so the read side must **gather all D, then inverse-WHT** — the main
  RTL change (see Path A/B decision below).
- **3-bit tier:** add `bits==3` to the pack/quant path + `pack_int3`.
- Regenerate real-Qwen bit-exact vectors for the rotated 3-bit value path; add a
  `sim_qwen` variant proving RTL == codec on real data (same bar as the 696k proof).

### 3. Cross-block
- **TIU (`token-importance-unit`):** flat 3-bit values **obsolete** the `tier_keep`
  CQ-8/CQ-4 value selector (a signed-off feature). Retire its value-precision role (keep
  only keep/evict), or repurpose — DECISION, not automatic. Touches `docs/tier_handshake.md`
  + the tier RTL semantics.
- **APA/MatE:** unchanged for Path A; for Path B (recommended) MatE gains one inverse-WHT
  on the P·V output per (query, head) — do it in fp on the accumulator output, not inside
  the INT32 integer accumulator.

### 4. Docs / spec / numbers
- Codec-of-record: values become **WHT-rotated per-token INT3** (was INT4). Recompute
  ~4.2 → ~3.3 b/val, ~3.8× → ~4.8× KV compression, ~15× → ~19× stacked. Update KVE README,
  `architecture/arch.yml`, org README, paper. Credit Abhiram + Chaithu.

## Five decisions — LOCKED (Chaithu + Abhiram, 2026-07-20)

1. **Path B.** Sum in rotated space; one inverse-WHT on the A·V output per (query, head),
   done in fp on the accumulator output (NOT inside the INT32 integer accumulator). MatE
   gains an output inverse-WHT stage; KVE just emits rotated dequant.
2. **True 3-bit packing** — `pack_int3`, 8 codes → 3 bytes. No 4-bit slot.
3. **Randomized Hadamard** — H·diag(±1) with one fixed shared sign vector (a sign ROM);
   free worst-case incoherence insurance. rotate = WHT(x·s); unrotate = WHT(y)·s (s²=1).
4. **Retire the TIU `tier_keep` value-demotion** — flat rotated-3bit obsoletes it; drop the
   value-precision role (keep only keep/evict). Update `docs/tier_handshake.md` + tier RTL.
5. **fp16-WHT + Llama gate** — RUNNING now (`wht_fp16_check.py`): randomized Hadamard +
   fp16 butterfly + 3-bit on Qwen2-0.5B/1.5B + Llama-3.2-1B. Coding is gated on this passing.

## Sequencing (all on this branch; merge to main only after review)

- **Phase 0 — validate:** fp16-WHT accuracy check + Llama-3.2-1B cross-family. Gate.
- **Phase 1 — reference:** C++ + Python codec, WHT value path + 3-bit tier + `pack_int3`,
  3-way parity.
- **Phase 2 — RTL:** `wht_unit.sv` + `cq_value_path` integration + real-Qwen bit-exact vectors.
- **Phase 3 — cross-block:** TIU tier decision, Path A/B datapath.
- **Phase 4 — docs/spec/paper + compression numbers.**
- **Merge:** only after Chaithu + Abhiram sign-off.
