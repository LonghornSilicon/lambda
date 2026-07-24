#!/usr/bin/env python3
"""Does a SMARTER 2-bit VALUE tier make graded demotion memory-neutral?

Extends `analysis/graded_2bit.py` (naive per-token INT2 craters; a memory-neutral
naive-2bit ladder is a wash on 0.5B and -0.065 on 1.5B). The open door was a
KIVI/KVQuant-style 2-bit codec:

  * grouped INT2 : one INT2 scale per GROUP of G2 channels along head-dim D
                   (per-group amax -> scale -> clamp(round,-2,1)). Extra bits:
                   2 + 16/G2 per value for the fp16 group scales (accounted for).
  * residual win : keep the most-recent R (fraction of seq) tokens at CQ-4
                   regardless of importance (KIVI keeps a recent window hot).

Keys held at uniform CQ-4+ per-channel throughout (isolate the value question);
no eviction, no APA. Same transformers-5.x AttentionInterface plumbing and the
same gotchas as graded_2bit.py: self-built causal mask, fp32 QK^T (fp16 -> NaN at
D=128), bit-parameterized reference quantizer.

Question: does grouped-INT2 (+residual window) let a memory-neutral graded ladder
(~4.0 b/val: top FP16 + recent CQ4 + boring grouped-INT2) MATCH OR BEAT uniform
CQ-4 on BOTH Qwen2-0.5B and 1.5B -- especially 1.5B where naive-2bit failed hardest?
"""
import argparse, json, math
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
CFG = {"mode": "fp16", "G": 128, "k_out": 2, "g2": 32, "top_frac": 0.10, "resid_frac": 0.10}
STATS = {"vbits_sum": 0.0, "vtok": 0}

# per-token value scheme codes
FP16, CQ4, CQ8, G2 = 0, 1, 2, 3


def _q_per_token(x, bits):
    """Per-token symmetric quant (one scale over all D dims), bit-parameterized."""
    if bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    return torch.round(x / scale).clamp(qmin, qmax) * scale


def _q_grouped(x, bits, g2):
    """KIVI-style grouped quant: one scale per group of g2 channels along D.
    INT2 -> qmax=1,qmin=-2 -> levels {-2,-1,0,1}."""
    *lead, D = x.shape
    xr = x.reshape(*lead, D // g2, g2)
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    amax = xr.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    q = torch.round(xr / scale).clamp(qmin, qmax) * scale
    return q.reshape(*lead, D)


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


def _bits_per_code(g2):
    return {FP16: 16.0, CQ4: 4.0, CQ8: 8.0, G2: 2.0 + 16.0 / g2}


def _assign_codes(frac_rank, Tk):
    """Per-token value scheme [B,H,Tk] for the current config.
    frac_rank in [0,1], 0 = most important (by accumulated post-softmax mass)."""
    mode = CFG["mode"]; dev = frac_rank.device
    B, H, _ = frac_rank.shape
    if mode == "uniform4":
        return torch.full_like(frac_rank, CQ4, dtype=torch.long)
    if mode == "grouped2_uniform":
        return torch.full_like(frac_rank, G2, dtype=torch.long)
    # residual window: most-recent R fraction (largest position index) -> CQ4
    code = torch.full_like(frac_rank, G2, dtype=torch.long)
    Rc = int(round(CFG["resid_frac"] * Tk))
    if Rc > 0:
        j = torch.arange(Tk, device=dev)
        recent = (j >= Tk - Rc).view(1, 1, Tk).expand(B, H, Tk)
        code = torch.where(recent, torch.full_like(code, CQ4), code)
    if mode == "graded_grouped2_neutral":
        top = frac_rank < CFG["top_frac"]                    # importance promotion wins
        code = torch.where(top, torch.full_like(code, FP16), code)
    return code


def _graded_values(value, code):
    vf = value.float(); out = vf.clone(); g2 = CFG["g2"]
    m = code == CQ4
    if m.any(): out = torch.where(m.unsqueeze(-1), _q_per_token(vf, 4), out)
    m = code == CQ8
    if m.any(): out = torch.where(m.unsqueeze(-1), _q_per_token(vf, 8), out)
    m = code == G2
    if m.any(): out = torch.where(m.unsqueeze(-1), _q_grouped(vf, 2, g2), out)
    # accounting (fp16 tokens left as-is)
    b4c = _bits_per_code(g2)
    bits = torch.zeros_like(code, dtype=torch.float32)
    for c, b in b4c.items():
        bits = torch.where(code == c, torch.tensor(b, device=code.device), bits)
    STATS["vbits_sum"] += float(bits.sum().item())
    STATS["vtok"] += int(bits.numel())
    return out.to(value.dtype)


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], key.shape[-2]
    if CFG["mode"] != "fp16":
        key = _q_keys_cq4plus(key, CFG["G"], CFG["k_out"])
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    causal = j <= i
    scores = scores.masked_fill(~causal, float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)
    if CFG["mode"] != "fp16":
        final = A.cumsum(2)[:, :, -1, :]                      # [B,H,Tk] total received mass
        finm = torch.where(causal[-1].unsqueeze(0).unsqueeze(0).expand_as(final),
                           final, torch.full_like(final, -1.0))
        order = finm.argsort(-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(Tk, device=A.device).expand_as(order))
        frac_rank = rank.float() / max(Tk - 1, 1)
        code = _assign_codes(frac_rank, Tk)
        value = _graded_values(value, code)
    out = torch.matmul(A.to(query.dtype), value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("graded2", attn)


CONFIGS = [
    {"name": "fp16",                    "mode": "fp16"},
    {"name": "uniform4",                "mode": "uniform4"},
    {"name": "grouped2_uniform_g32",    "mode": "grouped2_uniform", "g2": 32},
    {"name": "grouped2_uniform_g16",    "mode": "grouped2_uniform", "g2": 16},
    {"name": "grouped2_resid_g32_r10",  "mode": "grouped2_resid", "g2": 32, "resid_frac": 0.10},
    {"name": "grouped2_resid_g32_r20",  "mode": "grouped2_resid", "g2": 32, "resid_frac": 0.20},
    {"name": "grouped2_resid_g16_r20",  "mode": "grouped2_resid", "g2": 16, "resid_frac": 0.20},
    {"name": "graded_grouped2_neutral_g32_t10_r10", "mode": "graded_grouped2_neutral", "g2": 32, "top_frac": 0.10, "resid_frac": 0.10},
    {"name": "graded_grouped2_neutral_g32_t10_r15", "mode": "graded_grouped2_neutral", "g2": 32, "top_frac": 0.10, "resid_frac": 0.15},
    {"name": "graded_grouped2_neutral_g16_t07_r10", "mode": "graded_grouped2_neutral", "g2": 16, "top_frac": 0.07, "resid_frac": 0.10},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="graded_grouped2_result.json")
    a = ap.parse_args()
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="graded2").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    R = {}
    for cfg in CONFIGS:
        CFG.update({"g2": 32, "top_frac": 0.10, "resid_frac": 0.10})   # reset defaults
        CFG.update(cfg)
        STATS["vbits_sum"] = 0.0; STATS["vtok"] = 0
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = out["results"]["hellaswag"]["acc_norm,none"]
        vbits = (STATS["vbits_sum"] / STATS["vtok"]) if STATS["vtok"] else 16.0
        R[cfg["name"]] = {"acc_norm": acc, "avg_value_bits": round(vbits, 3),
                          "params": {k: v for k, v in cfg.items() if k not in ("name", "mode")},
                          "mode": cfg["mode"]}
        print(f"  {cfg['name']:34s} acc={acc:.4f}  avg_value_bits={vbits:.3f}")
    base = R["fp16"]["acc_norm"]
    for m, r in R.items():
        r["delta_vs_fp16"] = round(r["acc_norm"] - base, 4)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "n": a.n, "results": R}, f, indent=2)
    print("\n=== grouped-INT2 value ladder: memory vs accuracy (keys uniform CQ-4+) ===")
    for m, r in R.items():
        print(f"  {m:34s} {r['avg_value_bits']:5.2f} b/val   acc={r['acc_norm']:.4f}  Δ={r['delta_vs_fp16']:+.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
