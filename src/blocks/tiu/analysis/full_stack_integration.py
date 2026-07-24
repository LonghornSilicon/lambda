#!/usr/bin/env python3
"""All-three-blocks integration test on Qwen2.

Composes the full LonghornSilicon attention datapath in one custom attention:

  block 3  TIU   — H2O keep/evict (and optional graded demotion) to a KV budget
  block 2  KVCE  — ChannelQuant per-channel INT4 keys / per-token INT4 values (+outliers)
                   applied to the RETAINED K/V (per-token tier in graded mode)
  block 1  ACU   — precision controller routes the retained S.V through INT8 or FP16

Pipeline order mirrors the chip: KVCE decompresses K/V -> attention scores ->
TIU observes the scores and rules keep/demote/evict -> ACU routes the S.V MAC.

Grid isolates each block and then stacks all three, measured against the FP16
full-cache baseline (HellaSwag acc_norm).
"""
import argparse, json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
CFG = {"tiu": False, "frac": 0.25, "recent_ratio": 0.5, "graded": False,
       "kvce": "off", "apa": False, "G": 128, "k_out": 2}
STATS = {"int8": 0, "fp16": 0, "kept": 0, "total": 0}


# ---------- KVCE: ChannelQuant (torch port of channelquant_ref.cpp) ----------
def _q_per_token(x, bits):
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    return torch.round(x / scale).clamp(qmin, qmax) * scale


def _fwht_raw(x):
    """Raw fixed Walsh-Hadamard over the last dim (fp16, add/sub only), matching the KVE
    codec (channelquant_ref.hpp fwht_raw_f16 / wht_unit.sv). D must be a power of two."""
    *lead, D = x.shape
    y = x.reshape(-1, D).to(torch.float16); h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h)
        u, v = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((u + v, u - v), dim=2).reshape(-1, D).to(torch.float16)
        h *= 2
    return y.reshape(*lead, D)


def _q_wht3(v):
    """CQ-3-rot VALUE codec (Abhiram Bandi + Chaithu Talasila): rotate the row, per-token
    amax + INT3, dequant to fp16, inverse rotate, x(1/D). Bit-exact to the RTL."""
    D = v.shape[-1]
    r = _fwht_raw(v.to(torch.float16))
    amax = r.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax.double() / 3, min=EPS).to(torch.float16)
    code = torch.round(r.double() / scale.double()).clamp(-4, 3)
    rhat = (code * scale.double()).to(torch.float16)
    return (_fwht_raw(rhat).double() * (1.0 / D)).to(torch.float32)


def _q_keys_per_channel(k, bits, G, k_out):
    B, H, T, D = k.shape
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    kf = k.float(); out = torch.empty_like(kf)
    if k_out > 0:
        out_idx = kf.abs().amax(dim=2).topk(k_out, dim=-1).indices        # [B,H,k_out]
        om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
        om.scatter_(-1, out_idx, True)
    else:
        om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
    for a in range(0, T, G):
        b = min(a + G, T)
        grp = kf[:, :, a:b, :]
        scale = torch.clamp(grp.abs().amax(dim=2, keepdim=True) / qmax, min=EPS)
        out[:, :, a:b, :] = torch.round(grp / scale).clamp(qmin, qmax) * scale
    kf16 = k.to(torch.float16).float()
    out = torch.where(om.unsqueeze(2).expand(B, H, T, D), kf16, out)
    return out.to(k.dtype)


def _kvce(key, value, tier):
    if tier == "off":
        return key, value
    if tier == "cq8":
        return _q_per_token(key.float(), 8).to(key.dtype), _q_per_token(value.float(), 8).to(value.dtype)
    if tier == "cq3rot":                                   # CQ-4+ keys + WHT-rotated INT3 values
        return _q_keys_per_channel(key, 4, CFG["G"], CFG["k_out"]), _q_wht3(value).to(value.dtype)
    k_out = CFG["k_out"] if tier == "cq4+" else 0
    return _q_keys_per_channel(key, 4, CFG["G"], k_out), _q_per_token(value.float(), 4).to(value.dtype)


def _kvce_graded_values(value, tier_id):
    """Per-token mixed-precision VALUES only. tier_id: [B,H,T] in {0:fp16,1:cq8,2:cq4,3:cq4+}.
    Keys are NOT graded per-token: ChannelQuant compresses keys PER-CHANNEL over a token
    group, so per-token key demotion degenerates to the per-token-key path that collapses
    GQA accuracy (the failure ChannelQuant exists to avoid). Keys stay uniform per-channel."""
    vf = value.float()
    v_out = vf.clone()                       # fp16-tier values stay identity
    for tid, bits in [(1, 8), (2, 4), (3, 4)]:
        m = (tier_id == tid)
        if m.any():
            vq = _q_per_token(vf, bits)
            v_out = torch.where(m.unsqueeze(-1), vq, v_out)
    return v_out.to(value.dtype)


# ---------- ACU: APA precision-controller routing + INT8 S.V ----------
def _int8_sv(P, V):
    Pf, Vf = P.float(), V.float()
    ps = Pf.abs().amax(-1, keepdim=True).clamp(min=1e-9) / 127.0
    Pq = torch.round(Pf / ps).clamp(-128, 127)
    vs = Vf.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-9) / 127.0
    Vq = torch.round(Vf / vs).clamp(-128, 127)
    return (torch.matmul(Pq, Vq) * ps * vs).to(P.dtype)


def _apa_sv(P, V, scores):
    valid = torch.isfinite(scores)
    s = torch.where(valid, scores, torch.zeros_like(scores))
    smax = s.abs().amax(-1, keepdim=True).clamp(min=1e-9)
    sq = torch.round(s / smax * 127.0).clamp(-128, 127).abs()
    N = valid.sum(-1).clamp(min=1)
    fp16_row = (sq.amax(-1) * N) > (sq.sum(-1) * 10)
    STATS["fp16"] += int(fp16_row.sum().item())
    STATS["int8"] += int(fp16_row.numel() - fp16_row.sum().item())
    o16 = torch.matmul(P, V)
    o8 = _int8_sv(P, V)
    return torch.where(fp16_row.unsqueeze(-1), o16, o8)


# ---------- TIU: H2O importance -> keep mask / graded tier ----------
def _tiu(A, causal, i, j):
    """Returns (keep_mask [B,H,Tq,Tk] bool, tier_id [B,H,Tk] or None)."""
    B, H, Tq, Tk = A.shape
    C = max(1, round(CFG["frac"] * Tk))
    L = max(1, round(CFG["recent_ratio"] * C))
    Hn = max(C - L, 0)
    acc = A.cumsum(dim=2)                                     # accumulated mass
    recent = (i - j) < L
    eligible = causal & ~recent
    if Hn > 0:
        acc_e = torch.where(eligible, acc, torch.full_like(acc, float("-inf")))
        top = acc_e.topk(min(Hn, Tk), dim=-1)
        heavy = torch.zeros_like(A, dtype=torch.bool)
        heavy.scatter_(-1, top.indices, top.values > float("-inf"))
    else:
        heavy = torch.zeros_like(A, dtype=torch.bool)
    keep = (recent | heavy) & causal        # recent includes future (i-j<0<L); clip to causal
    over = (i + 1) > C
    keep = torch.where(over, keep, causal)

    tier_id = None
    if CFG["graded"]:
        final = acc[:, :, -1, :]                             # [B,H,Tk] mass over whole seq
        vmask = causal[-1].unsqueeze(0).unsqueeze(0).expand_as(final)
        if CFG.get("tier_mode") == "threshold":
            # 2-tier keep/demote by an accumulated-mass threshold — EXACTLY the RTL
            # tier_keep semantics (score >= threshold). Threshold = per-(B,H) mean of
            # valid tokens' accumulated mass (a calibratable percentile in silicon).
            denom = vmask.sum(-1, keepdim=True).clamp(min=1)
            thr = (torch.where(vmask, final, torch.zeros_like(final)).sum(-1, keepdim=True) / denom)
            keep_hh = final >= thr                            # heavy hitter?
            tier_id = torch.where(keep_hh, torch.tensor(1, device=A.device),   # keep -> CQ-8
                                  torch.tensor(2, device=A.device))            # demote -> CQ-4
        else:
            # rank-fraction map (design-space generalization)
            finm = torch.where(vmask, final, torch.full_like(final, -1.0))
            order = finm.argsort(dim=-1, descending=True)
            rank = torch.empty_like(order); rank.scatter_(-1, order,
                  torch.arange(Tk, device=A.device).expand_as(order))
            frac_rank = rank.float() / max(Tk - 1, 1)
            tier_id = torch.full((B, H, Tk), 3, dtype=torch.long, device=A.device)
            tier_id = torch.where(frac_rank < 0.50, torch.tensor(2, device=A.device), tier_id)
            tier_id = torch.where(frac_rank < 0.25, torch.tensor(1, device=A.device), tier_id)
            tier_id = torch.where(frac_rank < 0.10, torch.tensor(0, device=A.device), tier_id)
    return keep, tier_id


def full_attention(module, query, key, value, attention_mask,
                   scaling=None, dropout=0.0, **kwargs):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], key.shape[-2]
    i = torch.arange(Tq, device=query.device).unsqueeze(-1)
    j = torch.arange(Tk, device=query.device).unsqueeze(0)
    causal = j <= i

    # --- first pass for TIU importance / tier (on cq4-quantized K,V if KVCE on) ---
    tier_id = None
    if CFG["tiu"]:
        k0, v0 = _kvce(key, value, "cq4" if CFG["kvce"] != "off" else "off")
        s0 = (torch.matmul(query.float(), k0.float().transpose(-1, -2)) * scaling)
        s0 = s0.masked_fill(~causal, float("-inf"))
        A0 = F.softmax(s0, dim=-1, dtype=torch.float32)
        keep, tier_id = _tiu(A0, causal, i, j)
        STATS["kept"] += int(keep.sum().item()); STATS["total"] += int(causal.sum().item()) * query.shape[0] * query.shape[1]
    else:
        keep = causal

    # --- KVCE on retained K,V ---
    if CFG["graded"] and tier_id is not None:
        key, _ = _kvce(key, value, "cq4+")                 # keys: uniform per-channel
        value = _kvce_graded_values(value, tier_id)         # values: per-token mixed precision
    else:
        key, value = _kvce(key, value, CFG["kvce"])

    # --- scores + causal + TIU eviction mask ---
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~keep, float("-inf"))
    P = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    P = F.dropout(P, p=dropout, training=module.training)

    # --- ACU: route S.V ---
    if CFG["apa"]:
        attn = _apa_sv(P, value, scores)
    else:
        attn = torch.matmul(P, value)
    return attn.transpose(1, 2).contiguous(), P


AttentionInterface.register("full_stack", full_attention)


def run_cfg(lm, name, n, **kw):
    import lm_eval
    CFG.update({"tiu": False, "graded": False, "kvce": "off", "apa": False, "tier_mode": None})
    CFG.update(kw)
    for k2 in STATS: STATS[k2] = 0
    torch.manual_seed(0)
    out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=n, bootstrap_iters=0)
    acc = out["results"]["hellaswag"]["acc_norm,none"]
    it = STATS["int8"] + STATS["fp16"]
    kept = STATS["kept"] / STATS["total"] if STATS["total"] else None
    int8 = STATS["int8"] / it if it else None
    print(f"  {name:16s} acc={acc:.4f}  kept={kept}  int8={int8}")
    return {"acc_norm": acc, "kept": kept, "int8": int8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    ap.add_argument("--out", default="full_stack_result.json")
    args = ap.parse_args()
    CFG["frac"] = args.frac; CFG["recent_ratio"] = args.recent_ratio

    from lm_eval.models.huggingface import HFLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="full_stack").cuda().eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    R = {}
    R["fp16"]            = run_cfg(lm, "fp16", args.n)
    R["tiu"]             = run_cfg(lm, "tiu(evict)", args.n, tiu=True)
    R["kvce"]            = run_cfg(lm, "kvce(cq4+)", args.n, kvce="cq4+")
    R["apa"]             = run_cfg(lm, "apa", args.n, apa=True)
    R["tiu+kvce"]        = run_cfg(lm, "tiu+kvce", args.n, tiu=True, kvce="cq4+")
    R["ALL3"]            = run_cfg(lm, "ALL3", args.n, tiu=True, kvce="cq4+", apa=True)
    R["kvce(cq3rot)"]    = run_cfg(lm, "kvce(cq3rot)", args.n, kvce="cq3rot")
    R["ALL3(cq3rot)"]    = run_cfg(lm, "ALL3(cq3rot)", args.n, tiu=True, kvce="cq3rot", apa=True)
    R["ALL3+graded"]     = run_cfg(lm, "ALL3+graded", args.n, tiu=True, kvce="cq4+", apa=True, graded=True)
    # RTL tier-handshake semantics: 2-tier keep->CQ8 / demote->CQ4 by mass threshold, + APA
    R["ALL3+tier(hs)"]   = run_cfg(lm, "ALL3+tier(hs)", args.n, tiu=True, kvce="cq4+",
                                   apa=True, graded=True, tier_mode="threshold")

    base = R["fp16"]["acc_norm"]
    for r in R.values(): r["delta"] = round(r["acc_norm"] - base, 4)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": args.n, "frac": args.frac,
                   "recent_ratio": args.recent_ratio, "results": R}, f, indent=2)
    print(f"\n=== ALL-3-BLOCKS  (budget={args.frac}, recent_ratio={args.recent_ratio}, Δ vs fp16) ===")
    for name, r in R.items():
        print(f"  {name:16s} acc={r['acc_norm']:.4f}  Δ={r['delta']:+.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
