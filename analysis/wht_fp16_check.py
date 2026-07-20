#!/usr/bin/env python3
"""Phase-0 gate: does the HARDWARE-REALISTIC value rotation still hold?

Locks in the design decisions (Abhiram Bandi + Chaithu Talasila):
  - randomized Hadamard  (H·diag(±1), one fixed shared sign vector — free worst-case
    incoherence insurance vs a fixed WHT)
  - fp16 butterfly       (the silicon does the WHT in fp16, not fp32 — verify the
    log2(D) stages of fp16 add/sub don't erode the result)
  - amax + uniform 3-bit values, keys CQ-4+ (A/B verdict: no codebook, no L2)

Compares fp32-WHT (ideal) vs fp16-WHT (silicon) and randomized vs fixed, on
Qwen2-0.5B / Qwen2-1.5B / Llama-3.2-1B. If fp16 randomized ≈ fp32 ≈ fp16-baseline,
the rotation is safe to take to RTL. HellaSwag acc_norm, higher = better.
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channelquant_hw import q_per_token_hw, q_keys_per_channel_hw

_SIGN = {}
def sign(D, device):
    seed = CFG.get("seed", 1234)
    key = (D, seed)
    if key not in _SIGN:
        v = np.random.default_rng(seed).integers(0, 2, size=D) * 2 - 1   # ±1 draw for this seed
        _SIGN[key] = torch.tensor(v, dtype=torch.float32)
    return _SIGN[key].to(device)


def fwht_dt(x, dt):
    """Orthonormal fast Walsh-Hadamard with the butterfly carried in dtype `dt`
    (torch.float16 mirrors the silicon; torch.float32 is the ideal reference)."""
    *lead, D = x.shape
    y = x.reshape(-1, D).to(dt); h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h)
        u, v = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((u + v, u - v), dim=2).reshape(-1, D).to(dt)   # add/sub in dt
        h *= 2
    return (y.to(dt) / math.sqrt(D)).to(dt).reshape(*lead, D)


def quant_value(v, bits, dt, randomized):
    D = v.shape[-1]; x = v.float()
    s = sign(D, v.device) if randomized else None
    if randomized: x = x * s
    r = fwht_dt(x, dt).float()          # rotate (butterfly in dt)
    q = q_per_token_hw(r, bits)         # amax + uniform, hw-faithful
    y = fwht_dt(q, dt).float()          # unrotate (self-inverse). (Path B does this once on
    if randomized: y = y * s            #  the A·V output; per-value here is identical by linearity.)
    return y


CFG = {"mode": "fp16", "bits": 3, "dt": torch.float16, "rand": True, "seed": 1234, "G": 128, "k_out": 2}


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        key = q_keys_per_channel_hw(key, CFG["G"], CFG["k_out"])
        if CFG["mode"] == "rot":
            value = quant_value(value, CFG["bits"], CFG["dt"], CFG["rand"]).to(value.dtype)
        else:                                # plain (no rotation) baseline
            value = q_per_token_hw(value.float(), CFG["bits"]).to(value.dtype)
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]; j = torch.arange(Tk, device=s.device)[None, :]
    A = F.softmax(s.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


AttentionInterface.register("f16chk", attn)


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="wht_fp16_check.json")
    ap.add_argument("--seed_sweep", action="store_true",
                    help="fixed WHT vs several randomized sign-vector draws (fp16, 3b)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="f16chk").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    f16, f32 = torch.float16, torch.float32
    if a.seed_sweep:
        grid = [("fp16", dict(mode="fp16")),
                ("fixed-WHT fp16 / 3b", dict(mode="rot", bits=3, dt=f16, rand=False))]
        grid += [(f"rand seed={s} fp16 / 3b", dict(mode="rot", bits=3, dt=f16, rand=True, seed=s))
                 for s in (0, 1, 7, 42, 1234)]
    else:
        grid = [
            ("fp16",                       dict(mode="fp16")),
            ("val4 plain (current)",       dict(mode="plain", bits=4)),
            ("rand-WHT fp32 / 3b (ideal)", dict(mode="rot", bits=3, dt=f32, rand=True)),
            ("rand-WHT fp16 / 3b (SILICON)",dict(mode="rot", bits=3, dt=f16, rand=True)),
            ("fixed-WHT fp16 / 3b",        dict(mode="rot", bits=3, dt=f16, rand=False)),
        ]
    R = {}
    for label, cfg in grid:
        CFG.update({"mode": "fp16", "bits": 3, "dt": f16, "rand": True, "seed": 1234}); CFG.update(cfg)
        torch.manual_seed(0)
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = o["results"]["hellaswag"]["acc_norm,none"]
        R[label] = acc
        print(f"  {label:30s} acc={acc:.4f}")
    base = R["fp16"]
    print(f"\n=== fp16-WHT + randomized Hadamard gate, keys CQ-4+, {a.model} n={a.n} ===")
    for k, v in R.items():
        print(f"  {k:30s} {v:.4f}  Δ={v - base:+.4f}")
    json.dump({"model": a.model, "n": a.n,
               "results": {k: {"acc_norm": v, "delta_vs_fp16": round(v - base, 4)} for k, v in R.items()}},
              open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
