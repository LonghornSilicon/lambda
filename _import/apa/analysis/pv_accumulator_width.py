#!/usr/bin/env python3
"""P·V integer-accumulator width re-derivation for the MatE array (Qwen2 dims).

arch.yml sizes the K-axis accumulator at INT24, derived from the FFN/weight GEMM
(INT8×INT4 over hidden dim ≤4096) on the retired 3B target. But the INT8 P·V tile
reduces over the *token* dimension, not the hidden dim — so its accumulator width
scales with CONTEXT LENGTH, and the flat-attention corner is far worse than a peaked
one. This measures both:

  * THEORETICAL worst case: P quantized per-query-row to INT8 ([0,127]), V per-token to
    INT8 ([-127,127]); acc[q,d] = Σ_t P_int[q,t]·V_int[t,d]. A maximally flat causal row
    of length L makes every P_int≈127, so |acc| ≤ 127·127·L → needs 14+ceil(log2 L) bits.
  * EMPIRICAL: the actual max |acc| over every (layer,head,query,dim) on REAL Qwen2
    attention at ctx = 1k/2k/4k — real softmax is peaky, so this is the number that
    actually occurs, vs the bound the hardware must not overflow.

The integer accumulator value is scale-independent (it is the sum of the INT8 CODES),
so this is exactly the width the silicon adder/register must carry.
"""
import argparse, math, os, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from datasets import load_dataset

CAP = {}


def cap_hook(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(n_rep, 1) if n_rep > 1 else key
    v = value.repeat_interleave(n_rep, 1) if n_rep > 1 else value
    CAP[module.layer_idx] = (query.detach(), k.detach(), v.detach())
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], k.shape[-2]
    s = torch.matmul(query.float(), k.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device)[:, None]; j = torch.arange(Tk, device=s.device)[None, :]
    A = F.softmax(s.masked_fill(j > i, float("-inf")), -1, dtype=torch.float32)
    return torch.matmul(A.to(query.dtype), v).transpose(1, 2).contiguous(), A


def qint8_lastdim(x):
    """Per-row symmetric INT8 codes over the last dim (the ChannelQuant/APA convention)."""
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    return torch.round(x / (amax / 127.0)).clamp(-127, 127)


def bits_signed(v):
    return (math.floor(math.log2(v)) + 2) if v >= 1 else 1   # +1 for magnitude ceil, +1 for sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--ctxs", default="1024,2048,4096")
    ap.add_argument("--layers", default="2,10,18,26")
    ap.add_argument("--out", default="pv_accumulator_width.json")
    a = ap.parse_args()
    AttentionInterface.register("cap", cap_hook)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="cap").cuda().eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    layers = [int(x) for x in a.layers.split(",")]
    R = {}
    for ctx in [int(x) for x in a.ctxs.split(",")]:
        CAP.clear()
        with torch.no_grad():
            model(ids[:, :ctx])
        gmax = 0.0; arg = None
        for L in layers:
            if L not in CAP:
                continue
            q, k, v = CAP[L]                                  # [1,H,T,D]
            H, T, D = q.shape[1], q.shape[2], q.shape[3]
            scaling = 1.0 / math.sqrt(D)
            i = torch.arange(T, device=q.device)[:, None]; j = torch.arange(T, device=q.device)[None, :]
            causal = (j <= i)
            for h in range(H):
                s = torch.matmul(q[0, h].float(), k[0, h].float().t()) * scaling
                P = F.softmax(s.masked_fill(~causal, float("-inf")), -1, dtype=torch.float32)  # [T,T]
                Pi = qint8_lastdim(P).double()               # per-query-row INT8 codes
                Vi = qint8_lastdim(v[0, h].float()).double() # per-token INT8 codes  [T,D]
                acc = Pi @ Vi                                # [T,D] exact integer sums in fp64
                m = acc.abs().max().item()
                if m > gmax:
                    gmax = m; arg = (L, h)
        # theoretical flat-attention worst case for a causal row of length ctx
        theo = 127.0 * 127.0 * ctx
        R[str(ctx)] = {
            "empirical_max_abs_acc": gmax,
            "empirical_bits_signed": bits_signed(gmax),
            "empirical_argmax_layer_head": list(arg) if arg else None,
            "theoretical_flat_max_abs": theo,
            "theoretical_bits_signed": bits_signed(theo),
            "int24_capacity": 2 ** 23,
            "empirical_fits_int24": gmax < 2 ** 23,
            "theoretical_fits_int24": theo < 2 ** 23,
        }
        print(f"ctx={ctx:5d}: empirical max|acc|={gmax:,.0f} ({bits_signed(gmax)}b signed, "
              f"fits INT24={gmax < 2**23}) | flat worst-case={theo:,.0f} "
              f"({bits_signed(theo)}b, fits INT24={theo < 2**23})")
    import json
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "layers": layers, "results": R}, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
