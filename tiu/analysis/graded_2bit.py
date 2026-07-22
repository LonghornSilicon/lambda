#!/usr/bin/env python3
"""Does a sub-4-bit VALUE tier make graded demotion memory-neutral?

Isolates the KVCE value-precision ladder (keys held at uniform CQ-4+ per-channel;
no eviction, no APA) and compares, on Qwen2 HellaSwag:

  fp16            values FP16                                   (16.0 b/val, ceiling)
  uniform4        values all INT4                              ( 4.0 b/val, today's floor)
  uniform2        values all INT2                              ( 2.0 b/val, "does it crater?")
  graded_promote  top10 FP16 / 15 CQ8 / 25 CQ4 / 50 CQ4        ( 5.8 b/val, current design — a memory TAX)
  graded_neutral  top10 FP16 / 30 CQ4 / bottom60 CQ2           ( 4.0 b/val, memory-NEUTRAL, 2-bit floor)

Token importance = accumulated post-softmax attention mass (H2O). The reference
codec is bit-parameterized, so INT2 is a one-line change; only the SILICON lacks a
2-bit pack tier. Prints the measured avg value-bits alongside acc so memory
neutrality is visible, not asserted.
"""
import argparse, json, math
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
CFG = {"mode": "fp16", "G": 128, "k_out": 2}
STATS = {"vbits_sum": 0.0, "vtok": 0}

# ladders: (frac_rank_threshold, value_bits); 16 == fp16 identity
LADDERS = {
    "fp16":           [(1.01, 16)],
    "uniform4":       [(1.01, 4)],
    "uniform2":       [(1.01, 2)],
    "graded_promote": [(0.10, 16), (0.25, 8), (0.50, 4), (1.01, 4)],
    "graded_neutral": [(0.10, 16), (0.40, 4), (1.01, 2)],
}


def _q_per_token(x, bits):
    if bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    return torch.round(x / scale).clamp(qmin, qmax) * scale


def _q_keys_cq4plus(k, G, k_out):
    B, H, T, D = k.shape
    kf = k.float(); out = torch.empty_like(kf)
    out_idx = kf.abs().amax(2).topk(k_out, -1).indices
    om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device); om.scatter_(-1, out_idx, True)
    for a in range(0, T, G):
        b = min(a + G, T); grp = kf[:, :, a:b, :]
        s = torch.clamp(grp.abs().amax(2, keepdim=True) / 7, min=EPS)
        out[:, :, a:b, :] = torch.round(grp / s).clamp(-8, 7) * s
    out = torch.where(om.unsqueeze(2).expand(B, H, T, D), k.to(torch.float16).float(), out)
    return out.to(k.dtype)


def _bits_for_rank(frac_rank, ladder):
    """frac_rank [B,H,Tk] in [0,1] (0 = most important) -> per-token value bits."""
    bits = torch.full_like(frac_rank, ladder[-1][1])
    for thr, b in reversed(ladder):
        bits = torch.where(frac_rank < thr, torch.tensor(float(b), device=frac_rank.device), bits)
    return bits


def _graded_values(value, frac_rank, ladder):
    vf = value.float(); out = vf.clone()
    bitmap = _bits_for_rank(frac_rank, ladder)                # [B,H,Tk]
    for b in sorted({bb for _, bb in ladder}):
        if b >= 16:
            continue
        m = (bitmap.round() == b)
        if m.any():
            out = torch.where(m.unsqueeze(-1), _q_per_token(vf, b), out)
    # accounting
    STATS["vbits_sum"] += float(bitmap.sum().item())
    STATS["vtok"] += int(bitmap.numel())
    return out.to(value.dtype)


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], key.shape[-2]
    ladder = LADDERS[CFG["mode"]]
    # keys: uniform CQ-4+ for every non-fp16 mode (isolates the value question)
    if CFG["mode"] != "fp16":
        key = _q_keys_cq4plus(key, CFG["G"], CFG["k_out"])
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    causal = j <= i
    scores = scores.masked_fill(~causal, float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)
    # values: graded per-token by importance rank (accumulated mass over the seq)
    if CFG["mode"] != "fp16":
        final = A.cumsum(2)[:, :, -1, :]                      # [B,H,Tk] total received mass
        finm = torch.where(causal[-1].unsqueeze(0).unsqueeze(0).expand_as(final),
                           final, torch.full_like(final, -1.0))
        order = finm.argsort(-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(Tk, device=A.device).expand_as(order))
        frac_rank = rank.float() / max(Tk - 1, 1)
        value = _graded_values(value, frac_rank, ladder)
    out = torch.matmul(A.to(query.dtype), value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("graded2", attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="graded_2bit_result.json")
    a = ap.parse_args()
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="graded2").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    R = {}
    for mode in ["fp16", "uniform4", "uniform2", "graded_promote", "graded_neutral"]:
        CFG["mode"] = mode
        STATS["vbits_sum"] = 0.0; STATS["vtok"] = 0
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = out["results"]["hellaswag"]["acc_norm,none"]
        vbits = (STATS["vbits_sum"] / STATS["vtok"]) if STATS["vtok"] else 16.0
        R[mode] = {"acc_norm": acc, "avg_value_bits": round(vbits, 3)}
        print(f"  {mode:16s} acc={acc:.4f}  avg_value_bits={vbits:.3f}")
    base = R["fp16"]["acc_norm"]
    for m, r in R.items():
        r["delta_vs_fp16"] = round(r["acc_norm"] - base, 4)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "n": a.n, "results": R}, f, indent=2)
    print("\n=== value ladder: memory vs accuracy (keys uniform CQ-4+) ===")
    for m, r in R.items():
        print(f"  {m:16s} {r['avg_value_bits']:5.2f} b/val   acc={r['acc_norm']:.4f}  Δ={r['delta_vs_fp16']:+.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
