#!/usr/bin/env python3
"""Walsh–Hadamard-rotated VALUE quant inside ChannelQuant — silicon-faithful reproduction.

Idea and original result: **Abhiram Bandi and Chaithu Talasila.** Spin each value ROW by a
WHT before per-token quant so no single hot channel sets the per-token scale, then unspin on
read (WHT is self-inverse). Keys stay exactly as they are (per-channel INT4 + FP16 outlier
lane) — the rotation only touches per-token values, so it can't smear the per-channel key
structure that killed TurboQuant+ (the post-mortem: TurboQuant's value rotation was innocent,
only the key rotation was guilty). This module reproduces their result in the hw-faithful
codec (fp16 scales, double-precision mirror, bit-exact to RTL).

This reruns it in the hw-faithful codec (fp16 scales, double-precision mirror) to confirm
the trend on the same path that is bit-exact to the RTL. Keys held at CQ-4+ throughout.
Grid: fp16 | val4 naive | val3 naive | val3 rotated | (+ val2 rotated to locate the floor).
"""
import argparse, math, json, os, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channelquant_hw import q_per_token_hw, q_keys_per_channel_hw   # hw-faithful cores


def fwht(x):
    """Orthonormal fast Walsh–Hadamard transform over the last dim (D = 2^k).
    Orthonormal ⇒ symmetric & orthogonal ⇒ self-inverse: fwht(fwht(x)) == x."""
    *lead, D = x.shape
    assert (D & (D - 1)) == 0, f"WHT needs power-of-2 D, got {D}"
    y = x.reshape(-1, D).clone()
    h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h)
        u = y[:, :, 0, :]; v = y[:, :, 1, :]
        y = torch.stack((u + v, u - v), dim=2).reshape(-1, D)
        h *= 2
    return (y / math.sqrt(D)).reshape(*lead, D)


def q_val(v, bits, rotate):
    vf = v.float()
    if rotate:
        vf = fwht(vf)                       # spin rows
    qv = q_per_token_hw(vf, bits)           # per-token INT quant (hw-faithful)
    if rotate:
        qv = fwht(qv)                        # unspin (self-inverse)
    return qv.to(v.dtype)


CFG = {"mode": "fp16", "vbits": 4, "rotate": False, "G": 128, "k_out": 2}


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        key = q_keys_per_channel_hw(key, CFG["G"], CFG["k_out"])          # CQ-4+ keys, unchanged
        value = q_val(value, CFG["vbits"], CFG["rotate"]).to(value.dtype)  # rotated (or not) values
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]; j = torch.arange(Tk, device=s.device)[None, :]
    s = s.masked_fill(j > i, float("-inf"))
    A = F.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


AttentionInterface.register("wht", attn)


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="wht_value_result.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="wht").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    grid = [("fp16", dict(mode="fp16")),
            ("val4 naive (current)", dict(mode="q", vbits=4, rotate=False)),
            ("val3 naive", dict(mode="q", vbits=3, rotate=False)),
            ("val3 ROTATED", dict(mode="q", vbits=3, rotate=True)),
            ("val2 ROTATED", dict(mode="q", vbits=2, rotate=True))]
    R = {}
    for name, cfg in grid:
        CFG.update({"mode": "fp16", "vbits": 4, "rotate": False}); CFG.update(cfg)
        torch.manual_seed(0)
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = o["results"]["hellaswag"]["acc_norm,none"]
        R[name] = acc
        print(f"  {name:22s} acc_norm={acc:.4f}")
    base = R["fp16"]
    print(f"\n=== WHT-rotated values, keys CQ-4+, {a.model} n={a.n} (Δ vs fp16) ===")
    for name, acc in R.items():
        print(f"  {name:22s} {acc:.4f}  Δ={acc - base:+.4f}")
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "n": a.n,
                   "results": {k: {"acc_norm": v, "delta_vs_fp16": round(v - base, 4)} for k, v in R.items()}}, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
