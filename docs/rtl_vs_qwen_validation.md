# RTL validated against a real Qwen2 model

**Status:** Done — the ChannelQuant RTL reconstructs real Qwen2 **key AND value** tensors
**bit-for-bit** (≈700k elements across both model sizes) identical to the codec used for
the accuracy numbers, and that codec loses **≤0.02 acc_norm** vs FP16 from Qwen2-0.5B up
to Qwen2-7B.
**Date:** 2026-07-19.

## Why this exists

Two gaps sat between "the ChannelQuant RTL is verified" and "the accuracy numbers we
quote are the accuracy of the silicon":

1. The RTL parity tests drove **synthetic** golden vectors, not real model tensors.
2. The HellaSwag numbers were measured with a torch codec using **fp32 scales**. The
   silicon rounds every scale to **fp16** and uses round-half-to-even. So the quoted
   accuracy was for an *approximation* of the hardware, not the hardware.

This closes both: one fp16-exact software codec (`analysis/channelquant_hw.py`,
mirroring `channelquant_ref` in double precision) is (a) proven bit-identical to the RTL
on a grid of real Qwen2 slices — **both the value and the key path** — and (b) run inside
Qwen2 to measure its true accuracy at three model scales.

## Part 1 — the RTL == the codec, on real Qwen data (bit-exact)

`channelquant_hw.py --mode dump-multi` runs Qwen2, takes a **grid of real
(layer, head) slices**, and writes each as fp16 inputs alongside the fp16-exact codec's
reconstruction. Two testbenches replay them through the actual datapath RTL and compare
element-by-element:

| Testbench | RTL block | Real-Qwen coverage | Result |
|---|---|---|---|
| `tb_qwen_multi.sv` (`make sim_qwen`) | `cq_value_path` | 44 slices, D=64 **and** D=128 | **348,160 / 348,160 bit-exact** |
| `tb_qwen_key.sv` | `cq_key_path` | 44 slices, D=64 **and** D=128 | **348,160 / 348,160 bit-exact** |

Per-token INT4 **values** and per-channel grouped-INT4 + FP16-outlier-lane **keys** both
reconstruct exactly, across 5–6 layers × 4 heads × two head-dims. On real model tensors,
the fp16-exact codec **is** the silicon.

### Two bugs this caught in the *software* codec (the RTL was right both times)

1. **Signed zero.** `torch.round(-0.3)` returns −0.0, but the RTL code is a signed integer
   (0 has no sign). Fixed by casting the code to `int` before dequant.
2. **fp16 re-rounding of K̂.** The value path returns fp32 `code·scale`, but the key path
   returned the reconstruction re-cast to the input dtype (fp16), silently rounding away
   the sub-fp16 mantissa bits the RTL keeps. This showed up as ~28% of key elements off by
   ≤1 ULP; returning fp32 (like the hardware) closed it to 0.

## Part 2 — that silicon-faithful codec's Qwen accuracy, 0.5B → 7B

`channelquant_hw.py --mode accuracy` runs the same codec inside a Qwen2 forward and
measures HellaSwag acc_norm (tier CQ-4+), comparing FP16, the old fp32-scale
approximation, and the fp16-exact (= RTL) path:

| model | FP16 | approx (fp32 scales) | **hw (fp16-exact = RTL)** |
|---|---|---|---|
| Qwen2-0.5B (n=1000) | 0.489 | 0.474 (−0.015) | **0.480 (−0.009)** |
| Qwen2-1.5B (n=1000) | 0.590 | 0.587 (−0.003) | **0.581 (−0.009)** |
| Qwen2-7B (n=500) | 0.696 | 0.676 (−0.020) | **0.676 (−0.020)** |

**The RTL-faithful codec costs ≤0.02 acc_norm** across a 14× model-size range — and at
0.5B is actually *better* than the fp32-scale approximation (fp16 scales round more
consistently). At 7B, n=500 puts −0.020 within ~1σ of the measurement-noise floor. The
compression numbers we ship are the silicon's, not an optimistic proxy.

**Cross-family check — it isn't Qwen-specific.** The same fp16-exact codec on
**Llama-3.2-1B** (different family: tokenizer, RoPE, tied embeddings) costs **−0.007**
acc_norm (n=1000, CQ-4+) — in line with Qwen2. ChannelQuant generalizes across model
families, not just the one it was tuned on.

For the long-context view — where the KV cache is actually large — see the
`token-importance-unit` finding `long-context-holds.md`: the full-stack perplexity penalty
is flat (~7% on 1.5B) from 256 to 4096 tokens.

## Reproduce

```sh
# regenerate the real-Qwen test-vector grid (needs a GPU + the Qwen2 weights)
python analysis/channelquant_hw.py --mode dump-multi --model Qwen/Qwen2-0.5B --outdir ../rtl/tb/testvectors/qwen/g05b
python analysis/channelquant_hw.py --mode dump-multi --model Qwen/Qwen2-1.5B --outdir ../rtl/tb/testvectors/qwen/g15b
# bit-exact RTL checks (value + key)
make -C rtl sim_qwen                                                 # single-slice value path
iverilog -g2012 -DQD=64  -Irtl -o v64  rtl/tb/tb_qwen_multi.sv rtl/cq_value_path.sv rtl/amax_unit.sv rtl/cq_units.sv rtl/cq_units_syn.sv && (cd rtl && vvp ../v64  +TVDIR=tb/testvectors/qwen/g05b/multi)
iverilog -g2012        -Irtl -o k   rtl/tb/tb_qwen_key.sv   rtl/cq_key_path.sv rtl/residual_buffer.sv rtl/scale_bank.sv rtl/amax_unit.sv rtl/cq_units.sv rtl/cq_units_syn.sv && (cd rtl && vvp ../k    +TVDIR=tb/testvectors/qwen/g15b/multi)
# silicon-faithful accuracy
python analysis/channelquant_hw.py --mode accuracy --model Qwen/Qwen2-7B --tier cq4+ --n 500
```
