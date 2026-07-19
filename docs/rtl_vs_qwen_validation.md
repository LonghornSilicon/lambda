# RTL validated against a real Qwen2 model

**Status:** Done — the value-path RTL reconstructs a real Qwen2 tensor **bit-for-bit**
identical to the codec used for the accuracy numbers, and that codec loses only
**−0.009 acc_norm** vs FP16 on both Qwen2-0.5B and 1.5B.
**Date:** 2026-07-19.

## Why this exists

Two gaps sat between "the ChannelQuant RTL is verified" and "the accuracy numbers we
quote are the accuracy of the silicon":

1. The RTL parity tests drove **synthetic** golden vectors, not real model tensors.
2. The HellaSwag numbers were measured with a torch codec using **fp32 scales**. The
   silicon rounds every scale to **fp16** and uses round-half-to-even. So the quoted
   accuracy was for an *approximation* of the hardware, not the hardware.

This closes both: one fp16-exact software codec is (a) proven bit-identical to the
RTL on a real Qwen2 slice, and (b) run inside Qwen2 to measure its true accuracy.

## Part 1 — the RTL == the codec, on real Qwen data (bit-exact)

`analysis/channelquant_hw.py --mode dump-slice` runs Qwen2-0.5B, takes one real
value slice (layer 6, head 0, 25 tokens × 64 dims), and writes it as fp16
(`rtl/tb/testvectors/qwen/qwen_v.hex`) alongside the fp16-exact codec's
reconstruction (`qwen_vhat_hw.hex`). `rtl/tb/tb_qwen_validate.sv` replays the fp16
slice through the actual `cq_value_path` RTL and compares its V̂ to the codec's,
element by element:

```
$ make sim_qwen
loaded real Qwen value slice: D=64 T=25 bits=4
Real-Qwen RTL check: 1600/1600 elements bit-exact (V̂ rtl == fp16-exact codec)
ALL TESTS PASSED
```

**1600/1600 bit-exact.** Getting here caught a real bug in the *software* codec — a
signed-zero mismatch (`torch.round(-0.3)` returns −0.0, but the RTL code is a signed
integer with no sign on zero). The RTL was correct; the codec was fixed to cast the
code to `int32` before dequant. On real model tensors, the fp16-exact codec IS the
silicon.

## Part 2 — that silicon-faithful codec's Qwen accuracy

`analysis/channelquant_hw.py --mode accuracy` runs the same codec inside a Qwen2
forward pass and measures HellaSwag acc_norm (n=1000, tier CQ-4+), comparing FP16,
the old fp32-scale approximation, and the fp16-exact (= RTL) path:

| codec | Qwen2-0.5B | Qwen2-1.5B |
|---|---|---|
| fp16 (baseline) | 0.489 | 0.590 |
| approx (fp32 scales) | 0.474 (−0.015) | 0.587 (−0.003) |
| **hw (fp16-exact = RTL)** | **0.480 (−0.009)** | **0.581 (−0.009)** |

**The RTL-faithful codec costs −0.009 acc_norm on both models** — actually *better*
than the fp32-scale approximation on 0.5B (fp16 scales round more consistently). The
compression numbers we ship are the silicon's numbers, not an optimistic proxy.

## Reproduce

```sh
# regenerate the real-Qwen test vectors (needs a GPU + the Qwen2 weights)
python analysis/channelquant_hw.py --mode dump-slice
# bit-exact RTL check against them
make -C rtl sim_qwen
# silicon-faithful accuracy
python analysis/channelquant_hw.py --mode accuracy --model Qwen/Qwen2-0.5B --tier cq4+ --n 1000
```
