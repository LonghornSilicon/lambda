#!/usr/bin/env python3
"""Does the full stack hold accuracy when the KV cache is actually BIG?

The critic's objection to our HellaSwag numbers: HellaSwag prompts are ~50-100 tokens,
so TIU eviction (25% budget) + ChannelQuant never stress a large cache — the exact
regime the accelerator exists for. This measures **perplexity vs context length** on a
long document (WikiText-103), full stack (TIU evict 25% + KVCE cq4+ + APA INT8) vs FP16
full cache. If the gap stays flat as context grows from 256 -> 4096 tokens, the ~3%
result is a real long-context number, not a short-prompt artifact.

Reuses the exact all-3-blocks attention from full_stack_integration.py (same CFG).
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_stack_integration as fsi   # registers "full_stack"; exposes CFG


def set_stack(mode):
    fsi.CFG.update({"tiu": False, "graded": False, "kvce": "off", "apa": False, "tier_mode": None})
    if mode == "full":
        fsi.CFG.update({"tiu": True, "kvce": "cq4+", "apa": True})   # 25% budget (frac) set in CFG


def window_ppl(model, ids, ctx, stride_windows):
    """Mean per-token NLL over up to `stride_windows` non-overlapping windows of length ctx."""
    T = ids.shape[1]
    nlls, ntok = [], 0
    starts = list(range(0, T - ctx, ctx))[:stride_windows]
    for s in starts:
        w = ids[:, s:s + ctx]
        with torch.no_grad():
            logits = model(w).logits[:, :-1, :].float()
            tgt = w[:, 1:]
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                  reduction="sum")
        nlls.append(nll.item()); ntok += tgt.numel()
    return math.exp(sum(nlls) / ntok), len(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--ctxs", default="256,512,1024,2048,4096")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--out", default="long_context_result.json")
    a = ap.parse_args()
    fsi.CFG["frac"] = a.frac; fsi.CFG["recent_ratio"] = a.recent_ratio
    ctxs = [int(x) for x in a.ctxs.split(",")]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="full_stack").cuda().eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    print(f"tokenized stream: {ids.shape[1]} tokens")

    R = {}
    for ctx in ctxs:
        row = {}
        for mode in ["fp16", "full"]:
            set_stack(mode)
            try:
                ppl, nw = window_ppl(model, ids, ctx, a.windows)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"  ctx={ctx} {mode}: OOM, skipped"); continue
            row[mode] = ppl
            print(f"  ctx={ctx:5d} {mode:5s} ppl={ppl:8.3f}  ({nw} windows)")
        if "fp16" in row and "full" in row:
            row["ppl_ratio"] = round(row["full"] / row["fp16"], 4)
            row["ppl_delta_pct"] = round(100 * (row["full"] / row["fp16"] - 1), 2)
            print(f"  ctx={ctx:5d}  full/fp16 = {row['ppl_ratio']:.4f}  (+{row['ppl_delta_pct']:.2f}% ppl)")
        R[str(ctx)] = row

    with open(a.out, "w") as f:
        json.dump({"model": a.model, "frac": a.frac, "recent_ratio": a.recent_ratio,
                   "windows": a.windows, "results": R}, f, indent=2)
    print("\n=== full stack (TIU 25% + cq4+ + APA) perplexity vs FP16, by context length ===")
    for ctx, row in R.items():
        if "ppl_ratio" in row:
            print(f"  ctx={ctx:>5}  fp16={row['fp16']:.3f}  full={row['full']:.3f}  "
                  f"ratio={row['ppl_ratio']:.4f}  (+{row['ppl_delta_pct']:.2f}%)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
