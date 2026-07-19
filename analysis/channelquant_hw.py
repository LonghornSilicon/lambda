#!/usr/bin/env python3
"""Hardware-faithful ChannelQuant codec + Qwen validation.

The HellaSwag accuracy we quote used a torch codec with fp32 scales. The SILICON
(cq_value_path / cq_key_path RTL, bit-exact with channelquant_ref.hpp) rounds every
scale to fp16 and uses round-half-to-even. This module is the fp16-EXACT codec —
amax in fp16, scale in fp16, round-half-even — mirroring the RTL arithmetic
(channelquant_ref.hpp: real_to_f16 == numpy .astype(float16); srint == round-half-even).

Two uses:
  --dump-slice : run Qwen2, take one real (layer,head) K/V slice, and write
                 qwen_v.hex (fp16 in), qwen_vhat_hw.hex (fp32 V̂ from THIS codec).
                 The RTL testbench (rtl/tb/tb_qwen_validate.sv) replays qwen_v.hex
                 through cq_value_path and checks its V̂ == qwen_vhat_hw.hex bit-for-bit
                 -> proves the RTL codec == this codec on REAL Qwen data.
  --accuracy   : run THIS codec inside a Qwen2 forward and measure HellaSwag, vs FP16
                 and vs the fp32-scale approximation -> the silicon-faithful accuracy.
"""
import argparse, json, math, os
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14


# ---- fp16-exact per-token quant (mirrors cq_units: amax f16, scale f16, rint) ----
def q_per_token_hw(x: torch.Tensor, bits: int) -> torch.Tensor:
    """x: [..., D] fp16. Dequantized fp32, mirroring channelquant_ref (== RTL) exactly:
    every intermediate in DOUBLE (f16_to_real is a 64-bit real), fp16 scale, round-half-even.
    Matching fp64 (not fp32) is what makes it bit-exact on round-half boundaries."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    xf16 = x.to(torch.float16)
    amax = xf16.abs().amax(dim=-1, keepdim=True)                       # winner's fp16 magnitude
    scale = torch.clamp(amax.double() / qmax, min=EPS).to(torch.float16)  # scale_from_amax (double/qmax -> fp16)
    sd = scale.double()                                                # f16_to_real(scale)
    # code = clamp(round_half_even(x/s), qmin, qmax); .to(int64) kills signed -0.0 (RTL code is a
    # signed integer; round(-0.3)=0, not -0), so a zero code dequants to +0.0 like the hardware.
    code = torch.round(xf16.double() / sd).clamp(qmin, qmax).to(torch.int64).double()
    return (code * sd).to(torch.float32)                               # real_to_f32(code * s)


def q_per_token_approx(x: torch.Tensor, bits: int) -> torch.Tensor:
    """The fp32-scale approximation we originally quoted, for comparison."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1, keepdim=True) / qmax, min=EPS)
    return torch.round(xf / scale).clamp(qmin, qmax) * scale


def q_keys_per_channel_hw(k: torch.Tensor, G: int, k_out: int) -> torch.Tensor:
    """Per-channel grouped INT4 keys + k FP16 outlier channels (fp16-exact)."""
    B, H, T, D = k.shape
    kf16 = k.to(torch.float16)
    out = torch.empty(B, H, T, D, dtype=torch.float32, device=k.device)
    out_idx = kf16.abs().amax(2).topk(k_out, -1).indices if k_out > 0 else None
    om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
    if out_idx is not None:
        om.scatter_(-1, out_idx, True)
    for a in range(0, T, G):
        b = min(a + G, T)
        grp = kf16[:, :, a:b, :]
        amax = grp.abs().amax(2, keepdim=True)                         # per-channel fp16 magnitude
        scale = torch.clamp(amax.double() / 7, min=EPS).to(torch.float16)  # double/qmax -> fp16 (== RTL)
        sd = scale.double()
        code = torch.round(grp.double() / sd).clamp(-8, 7).to(torch.int64).double()
        out[:, :, a:b, :] = (code * sd).to(torch.float32)
    out = torch.where(om.unsqueeze(2).expand(B, H, T, D), kf16.float(), out)
    return out                       # fp32 K̂ (code*scale), like the RTL — NOT re-rounded to fp16


CFG = {"mode": "hw", "tier": "cq4", "G": 128, "k_out": 2}


def kvce_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if CFG["mode"] != "fp16":
        qfn = q_per_token_hw if CFG["mode"] == "hw" else q_per_token_approx
        vbits = 8 if CFG["tier"] == "cq8" else 4
        value = qfn(value.float(), vbits).to(value.dtype)
        if CFG["tier"] == "cq8":
            key = qfn(key.float(), 8).to(key.dtype)
        else:  # cq4 / cq4+ : per-channel grouped keys (+outliers for cq4+)
            if CFG["mode"] == "hw":
                key = q_keys_per_channel_hw(key, CFG["G"], CFG["k_out"] if CFG["tier"] == "cq4+" else 0)
            else:
                key = _approx_keys(key, CFG["G"], CFG["k_out"] if CFG["tier"] == "cq4+" else 0)
    Tq, Tk = query.shape[-2], key.shape[-2]
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device).unsqueeze(-1); j = torch.arange(Tk, device=s.device).unsqueeze(0)
    s = s.masked_fill(j > i, float("-inf"))
    A = F.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(A, value).transpose(1, 2).contiguous(), A


def _approx_keys(k, G, k_out):
    B, H, T, D = k.shape; kf = k.float(); out = torch.empty_like(kf)
    if k_out > 0:
        om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
        om.scatter_(-1, kf.abs().amax(2).topk(k_out, -1).indices, True)
    else:
        om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
    for a in range(0, T, G):
        b = min(a + G, T); grp = kf[:, :, a:b, :]
        sc = torch.clamp(grp.abs().amax(2, keepdim=True) / 7, min=EPS)
        out[:, :, a:b, :] = torch.round(grp / sc).clamp(-8, 7) * sc
    out = torch.where(om.unsqueeze(2).expand(B, H, T, D), k.to(torch.float16).float(), out)
    return out.to(k.dtype)


AttentionInterface.register("kvce_hw", kvce_attention)

CAP = {}
def cap_hook(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(n_rep, 1) if n_rep > 1 else key
    v = value.repeat_interleave(n_rep, 1) if n_rep > 1 else value
    if scaling is None: scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], k.shape[-2]
    sc = torch.matmul(query.float(), k.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=sc.device).unsqueeze(-1); j = torch.arange(Tk, device=sc.device).unsqueeze(0)
    A = F.softmax(sc.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32)
    CAP[module.layer_idx] = (k.detach().cpu(), v.detach().cpu())
    return torch.matmul(A.to(query.dtype), v).transpose(1, 2).contiguous(), A


def f16hex(x): return format(int(np.float16(x).view(np.uint16)), "04x")
def f32hex(x): return format(int(np.float32(x).view(np.uint32)), "08x")


def dump_slice(model_id, layer, head, T, outdir):
    AttentionInterface.register("cap", cap_hook)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 attn_implementation="cap").cuda().eval()
    ids = tok("The KV cache engine compresses key and value tensors with ChannelQuant "
              "before the attention score matmul inside the accelerator datapath.",
              return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)
    K, V = CAP[layer]                                   # [1,H,T,D]
    D = V.shape[-1]; T = min(T, V.shape[-2])
    Vsl = V[0, head, :T, :].to(torch.float16).numpy()   # [T,D] fp16
    # value path is CQ-8-agnostic; validate INT4 per-token values (tier CQ-4)
    Vhat = q_per_token_hw(torch.from_numpy(Vsl.astype(np.float32)).unsqueeze(0).unsqueeze(0), 4)[0, 0].numpy()
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/qwen_v.hex", "w") as f:
        f.write(f"{D} {T} 4\n")
        for t in range(T):
            f.write(" ".join(f16hex(Vsl[t, d]) for d in range(D)) + "\n")
    with open(f"{outdir}/qwen_vhat_hw.hex", "w") as f:
        for t in range(T):
            f.write(" ".join(f32hex(Vhat[t, d]) for d in range(D)) + "\n")
    print(f"wrote {outdir}/qwen_v.hex + qwen_vhat_hw.hex : D={D} T={T} (layer {layer} head {head})")


LONG_PROMPT = (
    "The KV cache engine compresses key and value tensors with ChannelQuant before the "
    "attention score matmul inside the accelerator datapath. Per-channel INT4 keys grouped "
    "over a token window, per-token INT4 values, and a small full-precision outlier lane "
    "keep the reconstruction near lossless while cutting memory almost four times. The token "
    "importance unit then evicts the tokens that never receive attention, and the precision "
    "controller routes the score-times-value multiply to INT8 whenever the tile is well "
    "conditioned, which is nearly always on real language modeling workloads.")


def dump_multi(model_id, layers, heads, T, outdir):
    """Dump a GRID of real Qwen (layer,head) slices — value AND key path — with a manifest,
    so the RTL testbenches can replay many real tensors, not one. Proves bit-exactness across
    the whole model and both head-dims (D=64 on 0.5B, D=128 on 1.5B)."""
    AttentionInterface.register("cap", cap_hook)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 attn_implementation="cap").cuda().eval()
    ids = tok(LONG_PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)                                          # one forward populates CAP[all layers]
    mdir = os.path.join(outdir, "multi"); os.makedirs(mdir, exist_ok=True)
    man = []
    for layer in layers:
        if layer not in CAP:
            continue
        K, V = CAP[layer]                                   # [1,H,T,D]
        D = V.shape[-1]; Tn = min(T, V.shape[-2])
        for head in heads:
            if head >= V.shape[1]:
                continue
            i = len(man)
            # ---- value path (per-token INT4) ----
            Vsl = V[0, head, :Tn, :].to(torch.float16).numpy()
            Vhat = q_per_token_hw(torch.from_numpy(Vsl.astype(np.float32)).unsqueeze(0).unsqueeze(0), 4)[0, 0].numpy()
            with open(f"{mdir}/val_{i}.hex", "w") as f:
                f.write(f"{D} {Tn} 4\n")
                for t in range(Tn):
                    f.write(" ".join(f16hex(Vsl[t, d]) for d in range(D)) + "\n")
            with open(f"{mdir}/valhat_{i}.hex", "w") as f:
                for t in range(Tn):
                    f.write(" ".join(f32hex(Vhat[t, d]) for d in range(D)) + "\n")
            # ---- key path (per-channel grouped INT4 + k=2 fp16 outlier lane) ----
            Ksl = K[0, head, :Tn, :].to(torch.float16)                       # [T,D]
            kt = Ksl.unsqueeze(0).unsqueeze(0)                               # [1,1,T,D]
            Khat = q_keys_per_channel_hw(kt, G=128, k_out=2)[0, 0].float().numpy()
            out_idx = kt.abs().amax(2).topk(2, -1).indices[0, 0].tolist()    # 2 outlier channels
            mask = [1 if d in out_idx else 0 for d in range(D)]
            with open(f"{mdir}/key_{i}.hex", "w") as f:
                f.write(f"{D} {Tn}\n")
                for t in range(Tn):
                    f.write(" ".join(f16hex(Ksl[t, d].item()) for d in range(D)) + "\n")
            with open(f"{mdir}/keymask_{i}.hex", "w") as f:
                f.write("\n".join(f"{m:02x}" for m in mask) + "\n")
            with open(f"{mdir}/keyhat_{i}.hex", "w") as f:
                for t in range(Tn):
                    f.write(" ".join(f32hex(Khat[t, d]) for d in range(D)) + "\n")
            man.append((i, D, Tn, layer, head))
    with open(f"{mdir}/manifest.txt", "w") as f:
        f.write(f"{len(man)}\n")
        for (i, D, Tn, layer, head) in man:
            f.write(f"{i} {D} {Tn} {layer} {head}\n")
    tot = sum(D * Tn for (_, D, Tn, _, _) in man)
    print(f"wrote {len(man)} slices to {mdir}/ (manifest.txt): {tot} value elems + {tot} key elems, "
          f"D={man[0][1] if man else '?'} layers={layers} heads={heads}")


def accuracy(model_id, n, tier, out):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 attn_implementation="kvce_hw").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
    R = {}
    grid = [("fp16", "fp16", tier), ("approx (fp32 scales)", "approx", tier),
            ("hw (fp16-exact = RTL)", "hw", tier)]
    for name, mode, ti in grid:
        CFG["mode"], CFG["tier"] = mode, ti
        torch.manual_seed(0)
        o = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=n, bootstrap_iters=0)
        acc = o["results"]["hellaswag"]["acc_norm,none"]
        R[name] = acc
        print(f"  {name:24s} acc_norm={acc:.4f}")
    base = R["fp16"]
    print("\n=== silicon-faithful KVCE accuracy (tier %s, n=%d) ===" % (tier, n))
    for name, acc in R.items():
        print(f"  {name:24s} {acc:.4f}  Δ={acc-base:+.4f}")
    with open(out, "w") as f:
        json.dump({"model": model_id, "n": n, "tier": tier,
                   "results": {k: {"acc_norm": v, "delta_vs_fp16": round(v - base, 4)} for k, v in R.items()}}, f, indent=2)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--mode", choices=["dump-slice", "dump-multi", "accuracy"], required=True)
    ap.add_argument("--layers", default="2,6,10,14,18")
    ap.add_argument("--heads", default="0,3,7,11")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tier", default="cq4")
    ap.add_argument("--layer", type=int, default=6); ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--outdir", default="../rtl/tb/testvectors/qwen")
    ap.add_argument("--out", default="channelquant_hw_result.json")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.mode == "dump-slice":
        dump_slice(a.model, a.layer, a.head, a.T, os.path.join(here, a.outdir))
    elif a.mode == "dump-multi":
        layers = [int(x) for x in a.layers.split(",")]
        heads = [int(x) for x in a.heads.split(",")]
        dump_multi(a.model, layers, heads, a.T, os.path.join(here, a.outdir))
    else:
        accuracy(a.model, a.n, a.tier, os.path.join(here, a.out))


if __name__ == "__main__":
    main()
