#!/usr/bin/env python3
"""Accuracy of the EXACT Phase-1 reference codec (raw-forward WHT + 2^-k inverse, fixed,
INT3), to confirm it lands at the gate's fixed-WHT number. Torch mirror of
channelquant_ref.hpp compress/decompress_values_wht3 (bit-exact to it by construction).
Keys held CQ-4+. WHT value rotation: Abhiram Bandi + Chaithu Talasila."""
import argparse, json, math, os, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channelquant_hw import q_keys_per_channel_hw
EPS = 2.0 ** -14


def fwht_raw(x):                      # raw fp16 butterfly, NO scaling (amax absorbs sqrt(D))
    *lead, D = x.shape
    y = x.reshape(-1, D).to(torch.float16); h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h)
        u, v = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((u + v, u - v), dim=2).reshape(-1, D).to(torch.float16)
        h *= 2
    return y.reshape(*lead, D)


def q_wht3(v):
    D = v.shape[-1]
    r = fwht_raw(v.to(torch.float16))                                  # forward
    amax = r.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax.double() / 3, min=EPS).to(torch.float16)
    code = torch.round(r.double() / scale.double()).clamp(-4, 3)       # srint, INT3
    rhat = (code * scale.double()).to(torch.float16)                   # dequant -> fp16
    y = fwht_raw(rhat)                                                 # inverse
    return (y.double() * (1.0 / D)).to(torch.float32)                 # x2^-k -> fp32


CFG = {"mode": "fp16", "G": 128, "k_out": 2}


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        key = q_keys_per_channel_hw(key, CFG["G"], CFG["k_out"])
        value = q_wht3(value).to(value.dtype)
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]; j = torch.arange(Tk, device=s.device)[None, :]
    A = F.softmax(s.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


AttentionInterface.register("whtref", attn)


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="wht_ref_accuracy.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="whtref").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    R = {}
    for name, mode in [("fp16", "fp16"), ("WHT-INT3 ref (fixed)", "q")]:
        CFG["mode"] = mode
        torch.manual_seed(0)
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        R[name] = o["results"]["hellaswag"]["acc_norm,none"]
        print(f"  {name:22s} acc={R[name]:.4f}")
    base = R["fp16"]
    print(f"  Δ(WHT-INT3 ref vs fp16) = {R['WHT-INT3 ref (fixed)'] - base:+.4f}")
    json.dump({"model": a.model, "n": a.n, "fp16": base,
               "wht_int3_ref": R["WHT-INT3 ref (fixed)"],
               "delta": round(R["WHT-INT3 ref (fixed)"] - base, 4)}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
