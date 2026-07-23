#!/usr/bin/env python3
"""KV-cache Value quantization at a MATCHED ~4-bit budget: OUTLIER-accommodation
vs NORMAL-VALUE-accommodation on REAL Qwen2 (HellaSwag acc_norm).

Three schemes are applied to the Value tensor in the actual attention KV path
(registered custom attn op, so `value` here is the real [B, H, T, D] cache that
feeds the A·V matmul). Keys are left FP16 so the ONLY variable is the V codec.
All three cost 4 bits/value + one fp16 per-token scale (16/D bits) => matched.

  (A) BASELINE  — plain uniform per-token INT4 (signed [-8,7]), scale = amax/7.
                  No outlier handling; the single largest value sets the scale so
                  every normal value is quantized coarsely.  == reference fq_per_token.
  (B) OliVe OVP — OUTLIER accommodation. Per-token scale = 3σ/7 (3σ rule, spec §5),
                  so values within ±3σ use the fine INT4 grid [-7,7] (codeword
                  1000 removed). Values beyond 3σ are OUTLIERS: encoded in abfloat
                  E2M1 (bias=2, magnitudes {8,12,16,24,32,48,64,96}, spec §2) and
                  their adjacent PAIR NEIGHBOUR (the "victim") is pruned to 0 so the
                  escape code fits in 4 bits (spec §3/§4). Both-outlier pairs keep
                  the larger, prune the smaller. Net = 4 bits/value.
  (C) NORMAL    — the INVERSE. Same fine 3σ/7 scale and same [-7,7] normal grid as
                  (B), so normal values get IDENTICAL full fidelity — but outliers
                  are simply CLIPPED to ±7 (== OliVe's Outlier-Suppression baseline).
                  No abfloat, no victim spent on outliers. Net = 4 bits/value.

So B and C share everything except the outlier mechanism, isolating exactly the
"spend the budget on outliers vs on normals" question; A is the naive control.

  HF_HOME=<cache> <venv>/bin/python analysis/olive_outlier_vs_normal.py \
      --model Qwen/Qwen2-0.5B --n 500
Outputs analysis/olive_outlier_vs_normal_<tag>.json.
Spec: OliVe (Guo et al., ISCA'23, arXiv:2304.07493), OVP §1-§6.
"""
import argparse, json, math, os, sys, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
TNORM = 7.0                       # max normal INT4 magnitude (spec §1: 1000 removed)
ABMAGS = [8, 12, 16, 24, 32, 48, 64, 96]   # abfloat E2M1 bias=2 magnitudes (spec §2)


def _scale_3sigma(v):
    """Per-token scale = 3σ/7 (spec §5): 3σ maps to the top normal code (7).

    σ is a ROBUST estimate (MAD·1.4826) so a single large outlier does not inflate
    the scale and coarsen the normal grid — faithful to 'the 3σ rule of the (normal)
    near-Gaussian distribution'. This keeps the normal codebook genuinely fine (the
    whole premise of both outlier- and normal-accommodation) and cleanly separates
    outliers (>3σ) from normals."""
    v = v.float()
    med = v.median(dim=-1, keepdim=True).values
    mad = (v - med).abs().median(dim=-1, keepdim=True).values
    sigma = 1.4826 * mad
    return (3.0 * sigma / TNORM).clamp_min(EPS).to(torch.float16).float()


def _scale_amax(v, qmax):
    amax = v.float().abs().amax(dim=-1, keepdim=True)
    return (amax / qmax).clamp_min(EPS).to(torch.float16).float()


def q_baseline_int4(value):
    """(A) plain per-token INT4, signed [-8,7], amax scale. No outlier handling."""
    dt = value.dtype
    v = value.float()
    s = _scale_amax(v, 7.0)                       # qmax for int4 = 7
    code = torch.round(v / s).clamp(-8, 7)
    return (code * s).to(dt)


def _abfloat(x, mags):
    """Nearest abfloat magnitude (spec §2), sign preserved. x already in scale units."""
    ax = x.abs().unsqueeze(-1)
    idx = (ax - mags).abs().argmin(dim=-1)
    return torch.sign(x) * mags[idx]


def q_olive_ovp(value):
    """(B) OliVe Outlier-Victim-Pair: preserve outliers in abfloat, prune the victim."""
    dt = value.dtype
    v = value.float()
    s = _scale_3sigma(v)
    x = v / s
    *lead, D = x.shape
    mags = torch.tensor(ABMAGS, dtype=torch.float32, device=x.device)
    xp = x.reshape(*lead, D // 2, 2)
    a, b = xp[..., 0], xp[..., 1]
    absa, absb = a.abs(), b.abs()
    outa, outb = absa > TNORM, absb > TNORM
    any_out = outa | outb
    afa, afb = _abfloat(a, mags), _abfloat(b, mags)
    i4a = a.round().clamp(-TNORM, TNORM)          # normal grid [-7,7]
    i4b = b.round().clamp(-TNORM, TNORM)
    keep0 = outa & (~outb | (absa >= absb))       # slot0 wins ties
    keep1 = outb & (~outa | (absb > absa))
    z = torch.zeros_like(a)
    ra = torch.where(any_out, torch.where(keep0, afa, z), i4a)
    rb = torch.where(any_out, torch.where(keep1, afb, z), i4b)
    r = torch.stack([ra, rb], dim=-1).reshape(*lead, D)
    return (r * s).to(dt)


def q_normal_clip(value):
    """(C) Normal-value accommodation: fine 3σ grid for normals, CLIP outliers to ±7."""
    dt = value.dtype
    v = value.float()
    s = _scale_3sigma(v)
    code = torch.round(v / s).clamp(-TNORM, TNORM)
    return (code * s).to(dt)


QUANT = {"A": q_baseline_int4, "B": q_olive_ovp, "C": q_normal_clip}
CFG = {"mode": "fp16"}


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1)
        value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        value = QUANT[CFG["mode"]](value)         # quantize the real V cache
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]
    j = torch.arange(Tk, device=s.device)[None, :]
    A = F.softmax(s.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


AttentionInterface.register("oliveref", attn)


def outlier_stats(model, tok):
    """Diagnostic: fraction of V values beyond 3σ (the outliers B preserves / C clips)."""
    import numpy as np
    samp = tok("The quick brown fox jumps over the lazy dog. " * 8,
               return_tensors="pt")["input_ids"][:, :128].to(model.device)
    fr = {}

    def hook(mod, inp, out):
        v = out.float()
        sig = v.std(dim=-1, keepdim=True)
        fr.setdefault("f", []).append(float((v.abs() > 3 * sig).float().mean()))
    hs = [m.register_forward_hook(hook) for n, m in model.named_modules() if n.endswith(".v_proj")]
    with torch.no_grad():
        model(samp, use_cache=False)
    for h in hs:
        h.remove()
    return round(float(np.mean(fr["f"])), 5) if fr.get("f") else None


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.tag or a.model.split("/")[-1]
    outp = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"olive_outlier_vs_normal_{tag}.json")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, attn_implementation="oliveref").cuda().eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    eff_bits = round(4 + 16.0 / D, 4)
    ostat = outlier_stats(model, tok)
    print(f"[setup] {a.model} D={D} eff_bits/value={eff_bits} "
          f"mean_frac_V_beyond_3sigma={ostat}", flush=True)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    labels = {"fp16": "FP16 baseline", "A": "(A) plain uniform INT4 per-token",
              "B": "(B) OliVe OVP  [OUTLIER accommodation]",
              "C": "(C) normal-value accommodation [clip outliers]"}
    R = {}
    for mode in ["fp16", "A", "B", "C"]:
        CFG["mode"] = mode
        torch.manual_seed(0)
        t0 = time.time()
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = float(o["results"]["hellaswag"]["acc_norm,none"])
        R[mode] = acc
        print(f"  {labels[mode]:48s} acc_norm={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    base = R["fp16"]
    out = {"model": a.model, "task": "hellaswag", "n": a.n, "head_dim": D,
           "eff_bits_per_value": eff_bits, "mean_frac_V_beyond_3sigma": ostat,
           "spec": "OliVe OVP (Guo et al., ISCA'23, arXiv:2304.07493)",
           "keys": "fp16 (V-only codec, isolates the outlier mechanism)",
           "fp16_acc_norm": round(base, 4),
           "configs": {}}
    for mode in ["A", "B", "C"]:
        out["configs"][mode] = {"label": labels[mode], "acc_norm": round(R[mode], 4),
                                "delta_vs_fp16": round(R[mode] - base, 4),
                                "eff_bits_per_value": eff_bits}
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\n[deltas vs fp16 {base:.4f}]  A={R['A']-base:+.4f}  "
          f"B={R['B']-base:+.4f}  C={R['C']-base:+.4f}", flush=True)
    print(f"[done] {outp}", flush=True)


if __name__ == "__main__":
    main()
