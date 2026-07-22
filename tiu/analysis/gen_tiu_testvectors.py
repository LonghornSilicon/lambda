#!/usr/bin/env python3
"""Generate replay test vectors for token_importance_unit from REAL Qwen2 attention.

Runs Qwen2-0.5B on a fixed prompt, hooks one (layer, head), and streams its causal
post-softmax attention through an H2O cache of N_SLOTS slots. Emits the exact
LOAD / ACC / EVICT op stream the RTL consumes, with the expected eviction victim on
every EVICT — computed by a Python reference that mirrors the RTL's argmin semantics
(seed slot 0, strict-less scan 0..N-1, saturating WEIGHT_WIDTH-bit accumulators).

Output: rtl/tb/testvectors/tiu_trace.hex, lines of  "<op> <slot> <weight> <exp>":
  op 0 = ACC   (slot, weight)
  op 1 = LOAD  (slot)
  op 2 = EVICT (exp = expected victim slot)
"""
import argparse, os, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

CAP = {}          # captured attention per layer: {layer_idx: A[H,Tq,Tk]}

def tiu_cap_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    n_rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
    v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], k.shape[-2]
    s = torch.matmul(query.float(), k.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=s.device).unsqueeze(-1)
    j = torch.arange(Tk, device=s.device).unsqueeze(0)
    s = s.masked_fill(j > i, float("-inf"))
    A = F.softmax(s, dim=-1, dtype=torch.float32)
    CAP[module.layer_idx] = A[0].detach().cpu()        # capture every layer; [H,Tq,Tk]
    return torch.matmul(A.to(query.dtype), v).transpose(1, 2).contiguous(), A


def sat_add(s, w, smax):
    return min(s + w, smax)


def argmin_victim(score, valid, n):
    """Mirror the RTL: seed with slot 0, strict-less scan 0..n-1."""
    m_score = score[0] if valid[0] else (1 << 30)
    m_idx = 0
    for k in range(n):
        if valid[k] and score[k] < m_score:
            m_score = score[k]; m_idx = k
    return m_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--slots", type=int, default=8)         # N_SLOTS
    ap.add_argument("--score_width", type=int, default=8)   # SCORE_WIDTH
    ap.add_argument("--weight_width", type=int, default=8)  # WEIGHT_WIDTH
    ap.add_argument("--prompt", default=(
        "The token importance unit tracks how much attention each cached key "
        "receives so the accelerator can evict the least useful tokens first, "
        "keeping the key value cache within a fixed on chip budget as the context "
        "grows longer and longer during autoregressive decoding."))
    ap.add_argument("--out", default="../rtl/tb/testvectors/tiu_trace.hex")
    args = ap.parse_args()

    AttentionInterface.register("tiu_cap", tiu_cap_attn)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="tiu_cap").cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)

    A = CAP[args.layer][args.head]                 # [Tq, Tk]
    T = A.shape[0]
    smax = (1 << args.score_width) - 1
    wmax = (1 << args.weight_width) - 1
    C = args.slots

    score = [0] * C
    valid = [0] * C
    slot_tok = [-1] * C          # which token occupies each slot
    ops = []                     # (op, slot, weight, exp)
    n_evict = 0

    for t in range(T):
        # place token t: use a free slot, else evict the argmin victim then reuse it
        free = next((k for k in range(C) if not valid[k]), None)
        if free is None:
            victim = argmin_victim(score, valid, C)
            ops.append((2, 0, 0, victim))           # EVICT (expected victim)
            valid[victim] = 0
            n_evict += 1
            free = victim
        ops.append((1, free, 0, 0))                 # LOAD token t into `free`
        score[free] = 0; valid[free] = 1; slot_tok[free] = t
        # query t attends to every currently-cached token -> accumulate its mass
        for k in range(C):
            if valid[k]:
                w = int(round(float(A[t, slot_tok[k]]) * wmax))
                if w > wmax: w = wmax
                if w > 0:
                    ops.append((0, k, w, 0))        # ACC
                    score[k] = sat_add(score[k], w, smax)

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        f.write(f"// TIU replay trace: model={args.model} layer={args.layer} head={args.head} "
                f"T={T} slots={C} score_w={args.score_width} weight_w={args.weight_width}\n")
        f.write(f"// ops={len(ops)} evictions={n_evict}\n")
        for op, s, w, e in ops:
            f.write(f"{op:x} {s:x} {w:02x} {e:x}\n")
    print(f"wrote {outp}: {len(ops)} ops, {n_evict} evictions over T={T} tokens, "
          f"layer {args.layer} head {args.head}")


if __name__ == "__main__":
    main()
