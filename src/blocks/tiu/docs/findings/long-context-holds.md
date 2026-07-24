# Does the full stack hold when the KV cache is actually big?

**Status:** Tested — **yes.** The full-stack perplexity penalty is **flat-to-shrinking**
as context grows from 256 to 4096 tokens; it does not blow up when the cache gets large.
**Date:** 2026-07-19.

## The objection this answers

Our headline accuracy numbers are HellaSwag acc_norm, and HellaSwag prompts are only
~50–100 tokens. But the whole point of the accelerator is the **KV cache** — a cost that
only bites at long context. At a 25% TIU budget on a 60-token prompt, the cache never
gets big enough to stress eviction or compression. A fair critic says: *you're testing a
long-context feature on short-context data.* So measure the thing directly — **perplexity
vs context length** — where a 25% budget at 4096 tokens is genuinely evicting ~3072 of
them and ChannelQuant is compressing a real, large cache.

## Setup

`analysis/long_context_eval.py` runs the exact all-3-blocks attention from
`full_stack_integration.py` (TIU evict at 25% budget, recent-ratio 0.5 + KVCE CQ-4+ +
APA INT8 S·V) over WikiText-2 test, and computes per-token perplexity for full-precision
vs the full stack at each context length (8 non-overlapping windows each).

## Result — perplexity ratio (full stack / FP16) by context length

**Qwen2-1.5B, 25% KV budget:**

| ctx | FP16 ppl | full-stack ppl | penalty |
|----:|---------:|---------------:|--------:|
| 256 | 14.576 | 15.750 | +8.1% |
| 512 | 13.126 | 13.994 | +6.6% |
| 1024 | 12.277 | 13.249 | +7.9% |
| 2048 | 8.420 | 9.018 | +7.1% |
| 4096 | 8.607 | 9.230 | **+7.2%** |

**Qwen2-0.5B, 25% KV budget:**

| ctx | FP16 ppl | full-stack ppl | penalty |
|----:|---------:|---------------:|--------:|
| 256 | 22.424 | 25.686 | +14.6% |
| 512 | 18.795 | 20.639 | +9.8% |
| 1024 | 17.797 | 19.737 | +10.9% |
| 2048 | 12.070 | 13.434 | +11.3% |
| 4096 | 12.135 | 13.296 | **+9.6%** |

## Reading it

1. **The penalty does not grow with context.** On 1.5B it is essentially **flat at ~7%**
   from 256 to 4096 tokens; on 0.5B it is highest at the *shortest* context (+14.6% @256)
   and lower at 4096 (+9.6%). The failure mode a critic fears — eviction throwing away
   tokens a long prompt still needs, penalty compounding as the cache grows — **does not
   happen** in this range. The full stack is as good (relatively) at 4k as at 256.
2. **Bigger model, smaller penalty.** 1.5B (~7%) tolerates the same aggressive stack far
   better than 0.5B (~10%); the richer representation absorbs 4-bit KV + 75% eviction.
   This is the right direction for scaling up.
3. **The budget knob barely matters at long context.** Backing 1.5B off to a 35% budget
   moves 4096 from +7.2% to +6.8% — a fraction of a point. 25% is already near the knee at
   long context, so the aggressive operating point costs almost nothing extra vs a safer one.

## Cross-family: same behavior on Llama-3.2-1B

Re-ran the sweep on **Llama-3.2-1B** (a different family — tokenizer, RoPE, tied
embeddings) with the identical full stack:

| ctx | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| full/FP16 ppl | +6.5% | +8.0% | +7.5% | +9.2% | +9.2% |

Same picture: a flat ~6–9% penalty that does not compound out to 4096 tokens. The
robustness is a property of the stack, not of the Qwen architecture. (Llama also takes
the stacked blocks *better* on HellaSwag — ALL-3 is −0.017 vs Qwen2's ~−0.03; see
`all-three-blocks-integration.md`.)

## Caveat

Perplexity is more sensitive than multiple-choice acc_norm, so +7% ppl is the *stress*
number, not a contradiction of the ~3% acc_norm result — different metric, and this one
is applied at every token including the hardest. The takeaway is the **trend**: it is flat.
WikiText-2 windows up to 4096 tokens; needle-in-a-haystack retrieval at longer contexts is
the natural next probe.

Reproduce:
```sh
python analysis/long_context_eval.py --model Qwen/Qwen2-1.5B --ctxs 256,512,1024,2048,4096 --windows 8 --frac 0.25
```
