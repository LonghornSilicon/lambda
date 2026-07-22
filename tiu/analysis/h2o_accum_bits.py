#!/usr/bin/env python3
"""Token Importance Unit (block 3) — accumulator bit-width (SCORE_WIDTH) study.

The RTL stores each token's accumulated attention mass in a fixed-point accumulator
of SCORE_WIDTH bits. Post-softmax masses A[q,j] in [0,1] are added into token j's
register step by step; the sum can reach up to the context length (Σ_j acc = i+1, so
any single acc[i,j] ∈ [0, i+1] ≤ Tk). The top-(C-L) heavy-hitter selection then reads
those registers. Question: how few bits can the accumulator be before the quantized
scores reorder the top-k and cost accuracy?

Model: an unsigned fixed-point accumulator spanning [0, FULLSCALE] with FULLSCALE = Tk
(the running context length — the minimal correct sizing, since acc never exceeds it).
B bits give 2^B-1 uniform levels; we round-to-nearest and saturate the accumulated mass
BEFORE top-k. Sweep B ∈ {16,12,10,8,6,4} at the gold config (25% budget, recent_ratio
0.5, per-head), plus an unquantized (fp32) reference. HellaSwag acc_norm, n=500.

Deliverable: smallest B with negligible acc loss -> recommended SCORE_WIDTH. Same
transformers 5.x AttentionInterface / fp32-scores / self-built-causal conventions as
h2o_analysis.py. Save h2o_accum_bits.json.
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

# bits=0 means "no quantization" (fp32 reference accumulator)
CFG = {"enabled": False, "frac": 0.25, "recent_ratio": 0.5, "bits": 0}


def quantize_acc(acc, bits, fullscale):
    """Uniform round-to-nearest, saturating, unsigned fixed-point over [0, fullscale]."""
    levels = (1 << bits) - 1
    step = fullscale / levels
    q = torch.round(acc / step).clamp_(0, levels)
    return q * step


def h2o_attention(module, query, key, value, attention_mask,
                  scaling=None, dropout=0.0, **kwargs):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])

    B, H, Tq, Tk = query.shape[0], query.shape[1], query.shape[-2], key.shape[-2]
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    causal = j <= i
    scores = scores.masked_fill(~causal, float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)

    if CFG["enabled"] and CFG["frac"] < 1.0:
        C = max(1, round(CFG["frac"] * Tk))
        L = max(1, round(CFG["recent_ratio"] * C))
        Hn = max(C - L, 0)
        acc = A.cumsum(dim=2)                                    # accumulated mass
        if CFG["bits"] > 0:                                      # fixed-point accumulator
            acc = quantize_acc(acc, CFG["bits"], float(Tk))
        recent = (i - j) < L
        eligible = causal & ~recent
        if Hn > 0:
            acc_e = torch.where(eligible, acc, torch.full_like(acc, float("-inf")))
            k = min(Hn, Tk)
            top = acc_e.topk(k, dim=-1)
            heavy = torch.zeros_like(A, dtype=torch.bool)
            heavy.scatter_(-1, top.indices, top.values > float("-inf"))
        else:
            heavy = torch.zeros_like(A, dtype=torch.bool)
        keep = recent | heavy
        over = (i + 1) > C
        keep = torch.where(over, keep, causal)
        Am = A * keep
        Am = Am / Am.sum(-1, keepdim=True).clamp(min=1e-9)
        A = Am

    A = A.to(query.dtype)
    A = F.dropout(A, p=dropout, training=module.training)
    out = torch.matmul(A, value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("h2o_bits", h2o_attention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--bits", default="16,12,10,8,6,4")
    ap.add_argument("--out", default="h2o_accum_bits.json")
    args = ap.parse_args()
    CFG["frac"], CFG["recent_ratio"] = args.frac, args.recent_ratio

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="h2o_bits").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    def run(enabled, bits):
        CFG["enabled"], CFG["bits"] = enabled, bits
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=args.n,
                                      bootstrap_iters=0)
        return out["results"]["hellaswag"]["acc_norm,none"]

    fp16_full = run(False, 0)                                   # no eviction at all
    gold_fp32 = run(True, 0)                                    # gold config, fp32 accumulator
    print(f"fp16 full-cache={fp16_full:.4f}  gold(fp32 acc)={gold_fp32:.4f}")

    bits_list = [int(b) for b in args.bits.split(",")]
    results = {}
    for b in bits_list:
        acc = run(True, b)
        results[str(b)] = {
            "bits": b,
            "acc_norm": round(acc, 4),
            "delta_vs_fp32_gold": round(acc - gold_fp32, 4),
            "delta_vs_fp16_full": round(acc - fp16_full, 4),
        }
        print(f"  bits={b:2d}  acc_norm={acc:.4f}  Δgold={acc-gold_fp32:+.4f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": args.n, "frac": args.frac,
                   "recent_ratio": args.recent_ratio,
                   "fp16_full_cache": round(fp16_full, 4),
                   "gold_fp32_accumulator": round(gold_fp32, 4),
                   "fullscale": "Tk (context length)",
                   "results": results}, f, indent=2)
    print("\n=== accumulator bit-width vs HellaSwag acc_norm (gold 25% budget) ===")
    print(f"  fp32 accumulator reference = {gold_fp32:.4f}")
    for name, r in results.items():
        print(f"  SCORE_WIDTH={name:>2}b  acc={r['acc_norm']:.4f}  Δvs-fp32={r['delta_vs_fp32_gold']:+.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
