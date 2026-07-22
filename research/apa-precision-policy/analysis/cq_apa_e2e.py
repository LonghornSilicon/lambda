#!/usr/bin/env python3
"""ChannelQuant (KVCE, block 2) + APA precision controller (ACU, block 1) end-to-end
verification on Qwen2 via HellaSwag.

Reconstructed after the scratchpad wipe. The ChannelQuant codec is a torch port of
the authoritative C++ reference (kv-cache-engine/sw/reference_model/channelquant_ref.cpp):
  - values / CQ-8 keys : per-token symmetric quant over D dims
  - CQ-4/CQ-4+ keys    : per-channel scale over a token group of G, INT4,
                         with k=2 top-|.| outlier channels held FP16 (CQ-4+)
  - scale = max(amax/qmax, EPS=2^-14); round-half-to-even; clamp [qmin,qmax]

The APA half routes each attention tile's S.V through INT8 or FP16 using the
synthesized precision-controller rule (precision_controller_ref.py):
  max(|s|)*N > 10*sum(|s|)  ->  FP16 ,  else INT8   (s = int8-quantized scores)

Grid: tier in {off(fp16), cq4, cq4+} x apa in {off, on}.
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers import AttentionInterface

EPS = 2.0 ** -14

# ---- global config + stats (read inside the attention hook) -----------------
CFG = {"tier": "off", "apa": False, "G": 128, "k_out": 2}
STATS = {"int8_tiles": 0, "fp16_tiles": 0}


# ---- ChannelQuant codec (torch, mirrors channelquant_ref.cpp) ---------------
def _q_per_token(x, bits):
    """Per-token symmetric quant over the last dim. x: [..., D]."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    code = torch.round(x / scale).clamp(qmin, qmax)   # torch.round = round-half-to-even
    return code * scale


def _q_keys_per_channel(k, bits, G, k_out):
    """Per-channel INT4 keys over token groups of size G, with k_out FP16 outlier
    channels. k: [B, H, T, D]. Returns dequantized k_hat, same shape/dtype."""
    B, H, T, D = k.shape
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    kf = k.float()
    out = torch.empty_like(kf)

    # outlier channels: top-k by max-|.| across all tokens, per (B,H)
    if k_out > 0:
        chan_mag = kf.abs().amax(dim=2)                       # [B,H,D]
        out_idx = chan_mag.topk(k_out, dim=-1).indices        # [B,H,k_out]
        outlier_mask = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
        outlier_mask.scatter_(-1, out_idx, True)
    else:
        outlier_mask = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)

    # per-channel grouped INT4 over keep channels
    for a in range(0, T, G):
        b = min(a + G, T)
        grp = kf[:, :, a:b, :]                                # [B,H,g,D]
        amax = grp.abs().amax(dim=2, keepdim=True)            # [B,H,1,D] per-channel
        scale = torch.clamp(amax / qmax, min=EPS)
        code = torch.round(grp / scale).clamp(qmin, qmax)
        out[:, :, a:b, :] = code * scale

    # outlier channels: identity FP16 (widen through fp16 rounding)
    kf16 = k.to(torch.float16).float()
    om = outlier_mask.unsqueeze(2).expand(B, H, T, D)
    out = torch.where(om, kf16, out)
    return out.to(k.dtype)


def apply_channelquant(key, value, tier):
    """key/value: [B, H_kv, T, D]. Returns quant-dequant K_hat, V_hat."""
    if tier == "off":
        return key, value
    if tier == "cq8":
        return _q_per_token(key.float(), 8).to(key.dtype), _q_per_token(value.float(), 8).to(value.dtype)
    # cq4 / cq4+ : keys per-channel INT4 (+outliers for cq4+), values per-token INT4
    k_out = CFG["k_out"] if tier == "cq4+" else 0
    k_hat = _q_keys_per_channel(key, 4, CFG["G"], k_out)
    v_hat = _q_per_token(value.float(), 4).to(value.dtype)
    return k_hat, v_hat


# ---- APA precision controller routing + INT8 S.V ----------------------------
def _int8_sv(P, V):
    """Simulate INT8 P.V: per-row int8 P, per-token int8 V, INT8 matmul, dequant.
    P: [B,H,Tq,Tk]  V: [B,H,Tk,D]  -> [B,H,Tq,D]."""
    Pf, Vf = P.float(), V.float()
    ps = Pf.abs().amax(-1, keepdim=True).clamp(min=1e-9) / 127.0    # per-row  [B,H,Tq,1]
    Pq = torch.round(Pf / ps).clamp(-128, 127)
    vs = Vf.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-9) / 127.0  # per-tensor [B,H,1,1]
    Vq = torch.round(Vf / vs).clamp(-128, 127)
    acc = torch.matmul(Pq, Vq)      # INT32-domain accumulate (done in fp32 to avoid fp16 overflow)
    return (acc * ps * vs).to(P.dtype)                             # ps per-row, vs scalar


def _route_and_sv(P, V, scores):
    """Per query-row APA routing on int8-quantized scores, then INT8 or FP16 S.V.
    `scores` still carries -inf at masked positions. Returns attn_output, updates STATS."""
    Bs, H, Tq, Tk = scores.shape
    valid = torch.isfinite(scores)                                  # mask-correct: real keys only
    s = torch.where(valid, scores, torch.zeros_like(scores))
    # int8-quantize the valid scores per row for the controller decision
    smax = s.abs().amax(-1, keepdim=True).clamp(min=1e-9)
    sq = torch.round(s / smax * 127.0).clamp(-128, 127).abs()       # int8 magnitudes
    N = valid.sum(-1).clamp(min=1)                                  # valid keys per row [B,H,Tq]
    lhs = sq.amax(-1) * N                                           # max * N
    rhs = sq.sum(-1) * 10                                           # sum * 10
    fp16_row = lhs > rhs                                            # True => FP16

    STATS["fp16_tiles"] += int(fp16_row.sum().item())
    STATS["int8_tiles"] += int((~fp16_row).numel() - fp16_row.sum().item())

    out_fp16 = torch.matmul(P, V)
    out_int8 = _int8_sv(P, V).to(out_fp16.dtype)
    sel = fp16_row.unsqueeze(-1)                                     # [B,H,Tq,1]
    return torch.where(sel, out_fp16, out_int8)


# ---- custom attention function ----------------------------------------------
def cq_apa_attention(module, query, key, value, attention_mask,
                     scaling=None, dropout=0.0, **kwargs):
    # KVCE half: quantize K,V (per kv-head, before GQA repeat)
    key, value = apply_channelquant(key, value, CFG["tier"])

    # GQA repeat
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    # scores in fp32: fp16 QK^T over D=128 (Qwen2-1.5B) overflows to NaN
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling  # [B,H,Tq,Tk]
    Tq, Tk = query.shape[-2], key.shape[-2]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :Tk]
    else:
        # transformers 5.x passes None for custom interfaces (expects is_causal);
        # build the additive causal mask ourselves (query pos offset = Tk - Tq).
        qpos = torch.arange(Tk - Tq, Tk, device=scores.device).unsqueeze(-1)
        kpos = torch.arange(Tk, device=scores.device).unsqueeze(0)
        causal = (kpos <= qpos)                                     # [Tq,Tk] True=keep
        scores = scores.masked_fill(~causal, float("-inf"))
    P = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    P = F.dropout(P, p=dropout, training=module.training)

    if CFG["apa"]:
        attn = _route_and_sv(P, value, scores)   # router handles the -inf mask internally
    else:
        attn = torch.matmul(P, value)

    return attn.transpose(1, 2).contiguous(), P


AttentionInterface.register("cq_apa", cq_apa_attention)


def build_model(model_id, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="cq_apa").cuda().eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="cq_apa_result.json")
    ap.add_argument("--G", type=int, default=128)
    args = ap.parse_args()
    CFG["G"] = args.G

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    dtype = torch.float16
    model, tok = build_model(args.model, dtype)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    grid = [("fp16", "off", False), ("cq4", "cq4", False), ("cq4+", "cq4+", False),
            ("apa", "off", True), ("cq4+apa", "cq4", True), ("cq4+ +apa", "cq4+", True)]

    results = {}
    for name, tier, apa in grid:
        CFG["tier"], CFG["apa"] = tier, apa
        STATS["int8_tiles"] = STATS["fp16_tiles"] = 0
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=args.n,
                                      bootstrap_iters=0)
        acc = out["results"]["hellaswag"]["acc_norm,none"]
        tot = STATS["int8_tiles"] + STATS["fp16_tiles"]
        frac = STATS["int8_tiles"] / tot if tot else None
        results[name] = {"acc_norm": acc, "tier": tier, "apa": apa, "int8_frac": frac}
        print(f"[{name:12s}] acc_norm={acc:.4f}  int8_frac={frac}")

    fp16 = results["fp16"]["acc_norm"]
    for name, r in results.items():
        r["delta_vs_fp16"] = round(r["acc_norm"] - fp16, 4)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": args.n, "results": results}, f, indent=2)
    print("\n=== SUMMARY (Δ vs fp16) ===")
    for name, r in results.items():
        print(f"  {name:12s} acc={r['acc_norm']:.4f}  Δ={r['delta_vs_fp16']:+.4f}  int8={r['int8_frac']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
