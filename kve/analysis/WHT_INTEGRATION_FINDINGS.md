# WHT value rotation — what to integrate into ChannelQuant

**Status:** integrated (reference, analysis, and RTL all live in this block).
**Idea/result:** Abhiram Bandi + Chaithu Talasila.
**One line:** Integrate the **Walsh-Hadamard rotation only**. TurboQuant's codebook,
L2-normalization, and dense Haar rotation add hardware for **zero or negative** accuracy
benefit — the cheap part is the whole win.

## The A/B (HellaSwag acc_norm, n=1000, keys held CQ-4+, values 3-bit unless noted)

| value codec | Qwen2-1.5B (D=128) | Qwen2-0.5B (D=64) |
|---|---|---|
| fp16 | 0.590 | 0.489 |
| val4 uniform (current 4-bit) | −0.005 | −0.002 |
| val3 naive (no rotation) | −0.010 | −0.022 |
| **WHT / amax / uniform (ours)** | **−0.003** | **0.000** |
| WHT / L2 / uniform | −0.301 | −0.076 |
| WHT / L2 / lloyd (Hadamard + TurboQuant codebook) | −0.008 | −0.004 |
| Haar / L2 / lloyd (full TurboQuant, dense R) | −0.004 | −0.004 |
| WHT / amax / uniform, 4-bit | −0.009 | −0.013 |

## Reading it — five decisive facts

1. **WHT + amax + uniform (the simple version) wins.** −0.003 / 0.000, matching or beating
   every TurboQuant variant, and beating today's 4-bit (−0.005 / −0.002) at 3 bits — so
   **rotated 3-bit replaces 4-bit, lossless, −25% value memory.**
2. **TurboQuant's Lloyd-Max codebook buys nothing here.** WHT/L2/lloyd (−0.008 / −0.004) is
   *worse than or equal to* WHT/amax/uniform. Even full Haar+codebook (−0.004) doesn't beat
   the simple version. No LUT, no re-fit — the codebook direction is a dead end for this data.
3. **Haar ≈ Hadamard.** Dense random-orthogonal (−0.004) vs fixed Walsh-Hadamard (−0.003) is
   noise. The expensive O(D²)+ROM rotation is NOT worth it — the HW-cheap Hadamard is as good.
4. **Normalization and quantizer are coupled.** amax pairs with uniform; L2 pairs with the
   codebook. Mixing (L2 + uniform) is catastrophic (−0.30 / −0.076) because unit-sphere
   coordinates cluster near 0 and uniform ticks waste their range. amax + uniform is the
   coherent, best pairing — and it's exactly ChannelQuant's *existing* per-token quantizer.
5. **3-bit is the sweet spot.** Rotated 4-bit is no better (worse, in noise) than rotated
   3-bit. No reason to spend the 4th bit.

## Integration decision

**Adopt:** a Walsh-Hadamard rotation on the per-token VALUE row, then ChannelQuant's
existing amax + uniform INT quant at **3 bits**. Keys unchanged (CQ-4+).

**Do NOT adopt from TurboQuant:** the Lloyd-Max codebook (no benefit, needs a LUT), the
L2-norm (no benefit, needs a sqrt unit), the dense Haar rotation (no benefit, needs a D×D
ROM + O(D²) matmul), the QJL sketch (only for rotated *keys*, which we don't do). The
minimal change is the cheapest change: a WHT butterfly (add/sub only) in front of the
value quant, and one inverse-WHT on the read side.

## Still open before this could go to master

- **fp16-WHT check** — this A/B rotated in fp32; confirm the hardware fp16 butterfly
  (7 stages at D=128) doesn't erode the result, and mirror it bit-exact RTL↔reference.
- **Retire the TIU value-demotion tier** — flat rotated-3bit obsoletes `tier_keep`'s
  CQ-8/CQ-4 value selector (a signed-off feature); conscious decision, not automatic.
- **Path B datapath** — sum in rotated space, one inverse-WHT on the output per decode step
  (do it in fp, not the INT32 accumulator).
- **Cross-family** — confirm on Llama-3.2-1B (D=64) like the base codec.
