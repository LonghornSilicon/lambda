#!/usr/bin/env python3
"""Token Importance Unit (LonghornSilicon block 3) — long-context knee study.

HellaSwag sequences are short, so H2O eviction barely triggers (kept ~0.97 at a
25% budget) and the accuracy knee is soft. This script exercises the policy on
genuinely long sequences: WikiText-2 test concatenated into ~1024-token windows,
measuring perplexity under H2O eviction vs KV budget. On long context the cache
budget actually bites every window, so the knee shows up much more sharply than
on HellaSwag.

Same H2O policy and transformers 5.x AttentionInterface as analysis/h2o_analysis.py:
  - scores in fp32 (fp16 QK^T over D overflows to NaN),
  - build the causal mask ourselves (transformers 5.x passes attention_mask=None),
  - recent=(i-j)<L includes future positions, so AND with causal.

Deliverable: perplexity vs KV-budget fraction. Save h2o_longctx_wikitext.json.
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from datasets import load_dataset

CFG = {"enabled": False, "frac": 1.0, "recent_ratio": 0.5}
STATS = {"kept": 0, "total": 0}


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
        acc = A.cumsum(dim=2)
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
        # recent=(i-j)<L is True for all future j too; AND with causal so the
        # kept-fraction stat counts only real (causal) retained positions.
        STATS["kept"] += int((keep & causal).sum().item())
        STATS["total"] += int(causal.sum().item()) * B * H
        Am = A * keep
        Am = Am / Am.sum(-1, keepdim=True).clamp(min=1e-9)
        A = Am

    A = A.to(query.dtype)
    A = F.dropout(A, p=dropout, training=module.training)
    out = torch.matmul(A, value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("h2o", h2o_attention)


@torch.no_grad()
def perplexity(model, input_ids, window):
    """Fixed-window (non-overlapping) perplexity: full NLL per window."""
    nll_sum, n_tok = 0.0, 0
    n = input_ids.shape[1]
    for s in range(0, n - 1, window):
        ids = input_ids[:, s:s + window]
        if ids.shape[1] < 2:
            break
        out = model(ids)
        logits = out.logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="sum")
        nll_sum += loss.item()
        n_tok += tgt.numel()
    return math.exp(nll_sum / n_tok), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--fracs", default="1.0,0.5,0.35,0.25,0.15,0.10,0.05")
    ap.add_argument("--max_tokens", type=int, default=131072,
                    help="cap total tokens evaluated (keeps runtime bounded)")
    ap.add_argument("--out", default="h2o_longctx_wikitext.json")
    args = ap.parse_args()
    CFG["recent_ratio"] = args.recent_ratio

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="h2o").cuda().eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] > args.max_tokens:
        ids = ids[:, :args.max_tokens]
    ids = ids.cuda()
    print(f"eval tokens: {ids.shape[1]}  window: {args.window}")

    fracs = [float(x) for x in args.fracs.split(",")]
    results = {}
    for fr in fracs:
        CFG["enabled"], CFG["frac"] = True, fr
        STATS["kept"] = STATS["total"] = 0
        ppl, n_tok = perplexity(model, ids, args.window)
        kept = STATS["kept"] / STATS["total"] if STATS["total"] else 1.0
        results[f"{fr:.2f}"] = {"frac": fr, "ppl": round(ppl, 4),
                                "kept_frac": round(kept, 4)}
        print(f"  budget={fr:.2f}  ppl={ppl:.3f}  kept={kept:.3f}")

    base = results[f"{fracs[0]:.2f}"]["ppl"]
    for r in results.values():
        r["ppl_ratio_vs_full"] = round(r["ppl"] / base, 4)
        r["delta_ppl_vs_full"] = round(r["ppl"] - base, 4)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "window": args.window, "n_tokens": n_tok,
                   "recent_ratio": args.recent_ratio, "results": results}, f, indent=2)
    print("\n=== H2O WikiText-2 perplexity vs KV budget (long context) ===")
    for name, r in results.items():
        print(f"  budget={name}  ppl={r['ppl']:.3f}  x{r['ppl_ratio_vs_full']:.3f}  kept={r['kept_frac']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
