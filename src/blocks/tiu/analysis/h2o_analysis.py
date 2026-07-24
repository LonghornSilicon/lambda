#!/usr/bin/env python3
"""Token Importance Unit (LonghornSilicon block 3) — algorithm analysis phase.

H2O accumulated-mass policy study on real Qwen2 traces. This validates the
retention algorithm BEFORE any RTL (per docs/new_block_blueprint.md).

Policy (Zhang et al. 2023, "H2O: Heavy-Hitter Oracle"):
  - Per (layer, head), accumulate each key token's post-softmax attention mass:
        acc[i, j] = sum_{q<=i} A[q, j]           (attention received up to step i)
  - Maintain a fixed KV cache budget C tokens. When the sequence exceeds C, keep
        * a recent local window of L tokens, plus
        * the top (C-L) "heavy hitter" tokens by accumulated mass,
    and evict the rest (their K/V no longer attended to).

The shelved ACU sparsity study proved this is the right signal: post-softmax
attention mass predicts token importance (r~0.99); pre-softmax proxies do not.

Deliverable: HellaSwag acc_norm vs KV-budget fraction, i.e. how small can the
cache get before accuracy drops — the curve that sizes the block.
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

# recent-window share of the budget; heavy hitters get the rest
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
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)      # query pos (offset 0 in eval)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    causal = j <= i
    scores = scores.masked_fill(~causal, float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)            # [B,H,Tq,Tk]

    if CFG["enabled"] and CFG["frac"] < 1.0:
        C = max(1, round(CFG["frac"] * Tk))                      # cache budget (tokens)
        L = max(1, round(CFG["recent_ratio"] * C))              # recent window
        Hn = max(C - L, 0)                                       # heavy-hitter slots
        acc = A.cumsum(dim=2)                                    # acc[.,.,i,j]
        recent = (i - j) < L                                    # keep last L (incl. current)
        eligible = causal & ~recent                             # heavy-hitter candidates
        if Hn > 0:
            acc_e = torch.where(eligible, acc, torch.full_like(acc, float("-inf")))
            k = min(Hn, Tk)
            top = acc_e.topk(k, dim=-1)                          # [B,H,Tq,k]
            heavy = torch.zeros_like(A, dtype=torch.bool)
            heavy.scatter_(-1, top.indices, top.values > float("-inf"))
        else:
            heavy = torch.zeros_like(A, dtype=torch.bool)
        keep = recent | heavy                                   # retained set
        over = (i + 1) > C                                      # only evict once over budget
        keep = torch.where(over, keep, causal)                  # under budget => keep all
        STATS["kept"] += int(keep.sum().item())
        STATS["total"] += int(causal.sum().item()) * B * H   # causal is [Tq,Tk]; keep is [B,H,Tq,Tk]
        Am = A * keep
        Am = Am / Am.sum(-1, keepdim=True).clamp(min=1e-9)      # renormalize over retained
        A = Am

    A = A.to(query.dtype)
    A = F.dropout(A, p=dropout, training=module.training)
    out = torch.matmul(A, value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("h2o", h2o_attention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--fracs", default="1.0,0.75,0.5,0.35,0.25,0.15,0.1")
    ap.add_argument("--out", default="h2o_result.json")
    args = ap.parse_args()
    CFG["recent_ratio"] = args.recent_ratio

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="h2o").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    fracs = [float(x) for x in args.fracs.split(",")]
    results = {}
    for fr in fracs:
        CFG["enabled"], CFG["frac"] = True, fr
        STATS["kept"] = STATS["total"] = 0
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=args.n,
                                      bootstrap_iters=0)
        acc = out["results"]["hellaswag"]["acc_norm,none"]
        kept = STATS["kept"] / STATS["total"] if STATS["total"] else 1.0
        results[f"{fr:.2f}"] = {"frac": fr, "acc_norm": acc, "kept_frac": round(kept, 4)}
        print(f"  budget={fr:.2f}  acc_norm={acc:.4f}  kept={kept:.3f}")

    base = results[f"{fracs[0]:.2f}"]["acc_norm"]
    for r in results.values():
        r["delta_vs_full"] = round(r["acc_norm"] - base, 4)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": args.n,
                   "recent_ratio": args.recent_ratio, "results": results}, f, indent=2)
    print("\n=== H2O accuracy vs KV budget (Δ vs full cache) ===")
    for name, r in results.items():
        print(f"  budget={name}  acc={r['acc_norm']:.4f}  Δ={r['delta_vs_full']:+.4f}  kept={r['kept_frac']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
