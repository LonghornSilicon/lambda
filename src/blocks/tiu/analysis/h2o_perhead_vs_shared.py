#!/usr/bin/env python3
"""Token Importance Unit (block 3) — per-head vs shared-budget ablation.

analysis/h2o_analysis.py selects heavy hitters PER HEAD: each (layer, head) keeps
its own top-(C-L) tokens by accumulated mass. That means the eviction decision (and
the KV rows physically dropped) differs head-to-head within a layer — the hardware
must track a keep-set per head.

A SHARED budget is cheaper silicon: rank tokens by accumulated attention mass SUMMED
over all heads in the layer, and keep ONE per-layer keep-set that every head shares.
The KV cache then drops whole (layer, token) rows, so the importance accumulator and
top-k are 1/H the width and there is a single eviction index per layer.

Question: does the cheaper shared budget cost HellaSwag accuracy vs per-head?

Same transformers 5.x AttentionInterface / fp32-scores / self-built-causal-mask
conventions as h2o_analysis.py. Save h2o_perhead_vs_shared.json.
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

# mode: "perhead" (top-k per head) or "shared" (top-k on head-summed mass, one set/layer)
CFG = {"enabled": False, "frac": 1.0, "recent_ratio": 0.5, "mode": "perhead"}


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
        recent = (i - j) < L                                    # [Tq,Tk], future-inclusive
        eligible = causal & ~recent

        if CFG["mode"] == "shared":
            # one keep-set per (batch, query-step) shared across all heads:
            # rank by attention mass SUMMED over heads.
            acc = A.cumsum(dim=2).sum(dim=1, keepdim=True)      # [B,1,Tq,Tk]
        else:
            acc = A.cumsum(dim=2)                               # [B,H,Tq,Tk]

        if Hn > 0:
            acc_e = torch.where(eligible, acc, torch.full_like(acc, float("-inf")))
            k = min(Hn, Tk)
            top = acc_e.topk(k, dim=-1)
            heavy = torch.zeros_like(acc, dtype=torch.bool)
            heavy.scatter_(-1, top.indices, top.values > float("-inf"))
        else:
            heavy = torch.zeros_like(acc, dtype=torch.bool)
        keep = recent | heavy                                   # [B,1,..] or [B,H,..]
        over = (i + 1) > C
        keep = torch.where(over, keep, causal)                  # broadcasts over H if shared
        Am = A * keep                                           # future A=0, so mask is safe
        Am = Am / Am.sum(-1, keepdim=True).clamp(min=1e-9)
        A = Am

    A = A.to(query.dtype)
    A = F.dropout(A, p=dropout, training=module.training)
    out = torch.matmul(A, value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("h2o_phvs", h2o_attention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--fracs", default="0.5,0.35,0.25,0.15")
    ap.add_argument("--out", default="h2o_perhead_vs_shared.json")
    args = ap.parse_args()
    CFG["recent_ratio"] = args.recent_ratio

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="h2o_phvs").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    def run(enabled, frac, mode):
        CFG["enabled"], CFG["frac"], CFG["mode"] = enabled, frac, mode
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=args.n,
                                      bootstrap_iters=0)
        return out["results"]["hellaswag"]["acc_norm,none"]

    fracs = [float(x) for x in args.fracs.split(",")]
    base = run(False, 1.0, "perhead")
    print(f"fp16 full-cache baseline acc_norm={base:.4f}")

    results = {}
    for fr in fracs:
        ph = run(True, fr, "perhead")
        sh = run(True, fr, "shared")
        results[f"{fr:.2f}"] = {
            "frac": fr,
            "perhead_acc": round(ph, 4),
            "shared_acc": round(sh, 4),
            "gap_shared_minus_perhead": round(sh - ph, 4),
            "perhead_delta_vs_full": round(ph - base, 4),
            "shared_delta_vs_full": round(sh - base, 4),
        }
        print(f"  budget={fr:.2f}  perhead={ph:.4f}  shared={sh:.4f}  gap={sh-ph:+.4f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": args.n, "recent_ratio": args.recent_ratio,
                   "fp16_baseline": round(base, 4), "results": results}, f, indent=2)
    print("\n=== per-head vs shared budget (HellaSwag acc_norm) ===")
    for name, r in results.items():
        print(f"  budget={name}  perhead={r['perhead_acc']:.4f}  "
              f"shared={r['shared_acc']:.4f}  gap={r['gap_shared_minus_perhead']:+.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
