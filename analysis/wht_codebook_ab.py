#!/usr/bin/env python3
"""A/B: how much of TurboQuant's value machinery should ChannelQuant actually adopt?

Idea/result: Abhiram Bandi + Chaithu Talasila (WHT-rotated 3-bit values). This script
separates the THREE independent factors so we integrate only what earns its keep:

  rotation      : none | wht (fixed Walsh-Hadamard, HW-cheap) | haar (dense random
                  orthogonal — what TurboQuant actually uses; O(D^2), needs a DxD ROM)
  normalization : amax (L-inf, our per-token scale) | l2 (unit-sphere, TurboQuant's)
  codebook      : uniform ticks | lloyd (TurboQuant's Beta-matched Lloyd-Max centroids)

Keys are held at ChannelQuant CQ-4+ (per-channel INT4 + FP16 outlier lane) in every row —
the rotation only ever touches per-token VALUES. HellaSwag acc_norm, higher = better.
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channelquant_hw import q_keys_per_channel_hw

CBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turboquant_codebooks")
_CB = {}
def codebook(d, bits):
    key = (d, bits)
    if key not in _CB:
        j = json.load(open(f"{CBDIR}/codebook_d{d}_b{bits}.json"))
        _CB[key] = (torch.tensor(j["bounds"][1:-1], dtype=torch.float32),   # interior bounds
                    torch.tensor(j["centroids"], dtype=torch.float32))
    return _CB[key]

_R = {}
def haar(d, device):
    if d not in _R:
        g = np.random.default_rng(0).standard_normal((d, d))
        q, r = np.linalg.qr(g)
        q = q * np.sign(np.diag(r))                     # Haar-uniform on O(d)
        _R[d] = torch.tensor(q, dtype=torch.float32)
    return _R[d].to(device)


def fwht(x):
    *lead, D = x.shape
    y = x.reshape(-1, D).clone(); h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h)
        u, v = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((u + v, u - v), dim=2).reshape(-1, D); h *= 2
    return (y / math.sqrt(D)).reshape(*lead, D)


def rotate(v, rot):
    if rot == "wht":  return fwht(v)
    if rot == "haar": return v @ haar(v.shape[-1], v.device).t()
    return v
def unrotate(r, rot):
    if rot == "wht":  return fwht(r)
    if rot == "haar": return r @ haar(r.shape[-1], r.device)
    return r


def quant_values(v, rot, norm, cb, bits):
    """v: [...,D] fp32 -> dequantized fp32 (rotation undone). Keys handled elsewhere."""
    D = v.shape[-1]
    r = rotate(v.float(), rot)
    if norm == "l2":
        s = r.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    else:                                                # amax
        s = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    u = r / s
    if cb == "lloyd":
        bnd, cen = codebook(D, bits); bnd = bnd.to(v.device); cen = cen.to(v.device)
        idx = torch.searchsorted(bnd, u.contiguous(), right=True)
        u_hat = cen[idx]
    else:                                                # uniform midrise, 2^(b-1)-1
        qmax = (1 << (bits - 1)) - 1
        u_hat = torch.round(u * qmax).clamp(-qmax, qmax) / qmax
    return unrotate(u_hat * s, rot)


CFG = {"mode": "fp16", "rot": "none", "norm": "amax", "cb": "uniform", "bits": 3, "G": 128, "k_out": 2}


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        key = q_keys_per_channel_hw(key, CFG["G"], CFG["k_out"])
        value = quant_values(value, CFG["rot"], CFG["norm"], CFG["cb"], CFG["bits"]).to(value.dtype)
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]; j = torch.arange(Tk, device=s.device)[None, :]
    A = F.softmax(s.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


AttentionInterface.register("ab", attn)


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="wht_codebook_ab.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="ab").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    # (label, mode, rot, norm, cb, bits)
    grid = [
        ("fp16",                       "fp16", "none", "amax", "uniform", 3),
        ("val4 uniform (current)",     "q",    "none", "amax", "uniform", 4),
        ("val3 uniform (naive)",       "q",    "none", "amax", "uniform", 3),
        ("WHT / amax / uniform  3b",   "q",    "wht",  "amax", "uniform", 3),   # our reproduction
        ("WHT / L2   / uniform  3b",   "q",    "wht",  "l2",   "uniform", 3),   # +L2 norm
        ("WHT / L2   / lloyd    3b",   "q",    "wht",  "l2",   "lloyd",   3),   # +codebook (HW candidate)
        ("Haar/ L2   / lloyd    3b",   "q",    "haar", "l2",   "lloyd",   3),   # full TurboQuant (dense R)
        ("WHT / amax / uniform  4b",   "q",    "wht",  "amax", "uniform", 4),   # rotated 4b (headroom?)
    ]
    R = {}
    for label, mode, rot, norm, cb, bits in grid:
        CFG.update({"mode": mode, "rot": rot, "norm": norm, "cb": cb, "bits": bits})
        torch.manual_seed(0)
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = o["results"]["hellaswag"]["acc_norm,none"]
        R[label] = acc
        print(f"  {label:26s} acc={acc:.4f}")
    base = R["fp16"]
    print(f"\n=== value-codec A/B, keys CQ-4+, {a.model} n={a.n} (Δ vs fp16) ===")
    for k, v in R.items():
        print(f"  {k:26s} {v:.4f}  Δ={v - base:+.4f}")
    json.dump({"model": a.model, "n": a.n,
               "results": {k: {"acc_norm": v, "delta_vs_fp16": round(v - base, 4)} for k, v in R.items()}},
              open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
