"""Phi-2 (2.7B) validation — pre-softmax ratio gate, /dev/shm streaming.

Different architecture family from Qwen2: Microsoft Phi, full MHA (not GQA),
head_dim=80, 32 layers. Same methodology as validate_real_llm_v2.py for comparison.
Routes HF_HOME to /dev/shm so weights never touch disk; deletes cache after load.
"""

import sys, os, math, json, shutil
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Route HF cache to RAM
HF_CACHE_DIR = "/dev/shm/hf_phi2"
os.environ["HF_HOME"] = HF_CACHE_DIR

FIGURES_DIR      = os.path.join(os.path.dirname(__file__), "figures")
MODEL_ID         = "microsoft/phi-2"
BLOCK_SIZE       = 64
RATIO_THRESHOLD  = 10.0

LONG_PROMPTS = [
    ("prose", " ".join([
        "The development of artificial intelligence has proceeded through several distinct phases.",
        "Early systems relied on explicit rule-based programming where human experts encoded",
        "domain knowledge directly into software. These expert systems achieved impressive results",
        "in narrow domains but struggled to generalize. The shift toward machine learning",
        "represented a fundamental change in approach allowing systems to learn patterns from data",
        "rather than following hand-crafted rules. Deep learning further accelerated this trend",
        "by enabling models to learn hierarchical representations automatically from raw inputs.",
        "The attention mechanism introduced in the transformer architecture allowed models to",
        "selectively focus on relevant parts of the input when producing each output token.",
        "This led to dramatic improvements in language understanding and generation tasks.",
        "Large language models trained on vast corpora of text demonstrated emergent capabilities",
        "that were not explicitly programmed and were not anticipated from smaller models.",
        "The scaling laws observed in these models suggested that performance would continue",
        "to improve with more data more compute and more parameters following predictable trends.",
        "Whether this scaling will continue indefinitely or eventually plateau remains an open",
        "question in the field. Researchers are now exploring ways to make these models more",
        "efficient through techniques like quantization pruning and knowledge distillation.",
    ] * 3)),
    ("code", """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return self.w_o(torch.matmul(self.dropout(attn), V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)), attn
""" * 4),
]


class AttentionCapture:
    def __init__(self, model):
        self.hooks  = []
        self.q_acts = {}
        self.k_acts = {}
        self.v_acts = {}
        self._register(model)

    def _register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            def make_hook(d, key):
                def h(module, inp, out):
                    # Llama q/k/v_proj return plain tensors
                    d[key] = (out[0] if isinstance(out, tuple) else out).detach().float()
                return h
            self.hooks.append(attn.q_proj.register_forward_hook(make_hook(self.q_acts, layer_idx)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_hook(self.k_acts, layer_idx)))
            self.hooks.append(attn.v_proj.register_forward_hook(make_hook(self.v_acts, layer_idx)))

    def clear(self):
        self.q_acts.clear(); self.k_acts.clear(); self.v_acts.clear()

    def remove(self):
        for h in self.hooks: h.remove()


def compute_tile_ratios(Q_raw, K_raw, num_q_heads, num_kv_heads, head_dim, block_size=BLOCK_SIZE):
    B, N, _ = Q_raw.shape
    scale = 1.0 / math.sqrt(head_dim)
    Q = Q_raw.view(B, N, num_q_heads, head_dim).permute(0, 2, 1, 3)
    K = K_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    if num_kv_heads != num_q_heads:
        K = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    ratios = []
    n_blocks = math.ceil(N / block_size)
    for i in range(n_blocks):
        Q_b = Q[:, :, i*block_size:min((i+1)*block_size, N), :]
        for j in range(n_blocks):
            K_b = K[:, :, j*block_size:min((j+1)*block_size, N), :]
            S = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale
            abs_S = S.abs()
            ratios.append(abs_S.max().item() / (abs_S.mean().item() + 1e-6))
    return ratios


def compute_channel_outliers(K_raw, V_raw, num_kv_heads, head_dim):
    B, N, _ = K_raw.shape
    K = K_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 3, 1)
    V = V_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 3, 1)
    total = 0; outliers = 0
    for tensor in [K, V]:
        ch_max  = tensor.abs().max(dim=-1).values.max(dim=0).values.max(dim=0).values
        ch_mean = tensor.abs().mean(dim=-1).mean(dim=0).mean(dim=0)
        total    += head_dim
        outliers += (ch_max / (ch_mean + 1e-6) > RATIO_THRESHOLD).sum().item()
    return outliers / total, outliers, total


if __name__ == "__main__":
    print("=" * 68, flush=True)
    print(f"Llama-3.2-1B Validation — Pre-Softmax Ratio Gate")
    print("=" * 68, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_ID} → /dev/shm ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Free /dev/shm cache — weights are in VRAM now
    shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
    print("  Cache cleared from /dev/shm.", flush=True)

    cfg = model.config
    n_layers     = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.hidden_size // num_q_heads
    print(f"  Layers={n_layers}  Q-heads={num_q_heads}  KV-heads={num_kv_heads}"
          f"  head_dim={head_dim}", flush=True)

    capture = AttentionCapture(model)

    all_ratios       = []
    per_layer_ratios = [[] for _ in range(n_layers)]
    per_layer_ch_out = [[] for _ in range(n_layers)]
    seq_lens_used    = []

    for prompt_name, prompt_text in LONG_PROMPTS:
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=512).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        seq_lens_used.append(seq_len)
        print(f"\nPrompt '{prompt_name}': {seq_len} tokens", flush=True)

        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(n_layers):
            Q_raw = capture.q_acts[layer_idx].cpu()
            K_raw = capture.k_acts[layer_idx].cpu()
            V_raw = capture.v_acts[layer_idx].cpu()

            ratios = compute_tile_ratios(Q_raw, K_raw, num_q_heads, num_kv_heads, head_dim)
            all_ratios.extend(ratios)
            per_layer_ratios[layer_idx].extend(ratios)

            pct, _, _ = compute_channel_outliers(K_raw, V_raw, num_kv_heads, head_dim)
            per_layer_ch_out[layer_idx].append(pct)

        capture.clear()
        print(f"  Layers processed: {n_layers}", flush=True)

    capture.remove()
    del model; torch.cuda.empty_cache()

    # Summary
    n_fp16 = sum(1 for r in all_ratios if r > RATIO_THRESHOLD)
    n_int8 = len(all_ratios) - n_fp16
    total  = len(all_ratios)

    print("\n=== Pre-Softmax Ratio Results ===", flush=True)
    print(f"  Total tiles:            {total}", flush=True)
    print(f"  Ratio > 10 (FP16):      {n_fp16} ({100*n_fp16/total:.1f}%)", flush=True)
    print(f"  Ratio ≤ 10 (INT8-safe): {n_int8} ({100*n_int8/total:.1f}%)", flush=True)
    print(f"  Median ratio:           {float(np.nanmedian(all_ratios)):.2f}", flush=True)
    print(f"  Max ratio:              {max(all_ratios):.1f}", flush=True)
    print(f"  Seq lengths used:       {seq_lens_used}", flush=True)

    print("\n=== Per-Layer Breakdown ===", flush=True)
    print(f"  {'Layer':>6} {'FP16%':>8} {'KV outlier%':>13}", flush=True)
    layer_fp16_pct = []
    layer_ch_pct   = []
    for l in range(n_layers):
        r_list   = per_layer_ratios[l]
        pct_fp16 = 100 * sum(1 for r in r_list if r > RATIO_THRESHOLD) / max(len(r_list), 1)
        pct_ch   = 100 * float(np.mean(per_layer_ch_out[l]))
        layer_fp16_pct.append(pct_fp16)
        layer_ch_pct.append(pct_ch)
        bar = "█" * int(pct_fp16 / 5)
        print(f"  {l:>6}  {pct_fp16:>6.1f}%  {pct_ch:>11.1f}%  {bar}", flush=True)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Phi-2 (2.7B) -- Pre-Softmax Ratio Gate Validation"
                 f"  (seq≈{int(np.mean(seq_lens_used))} tokens)", fontsize=12, fontweight="bold")

    ax = axes[0]
    log_ratios = [math.log10(max(r, 0.01)) for r in all_ratios]
    ax.hist(log_ratios, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(math.log10(RATIO_THRESHOLD), color="red", linestyle="--",
               linewidth=1.5, label=f"Threshold={RATIO_THRESHOLD}")
    ax.set_xlabel("log₁₀(max|S| / mean|S|)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Tile Ratio Distribution\nFP16: {100*n_fp16/total:.1f}%  INT8: {100*n_int8/total:.1f}%")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.barh(range(n_layers), layer_fp16_pct, color="#DD8452", alpha=0.85)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("FP16 tiles (%)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title("Per-Layer FP16 Tile %")
    ax.set_xlim(0, 105); ax.invert_yaxis()

    ax = axes[2]
    ax.barh(range(n_layers), layer_ch_pct, color="#55A868", alpha=0.85)
    ax.axvline(10, color="red", linestyle="--", linewidth=1, alpha=0.7, label="10% threshold")
    ax.set_xlabel("Outlier KV channels (%)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title("Per-Layer KV Channel Outliers")
    ax.set_xlim(0, 105); ax.invert_yaxis(); ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "phi2_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out_path}", flush=True)

    stats = {
        "model": MODEL_ID,
        "total_tiles": total, "n_fp16": n_fp16, "n_int8": n_int8,
        "pct_fp16": 100*n_fp16/total, "pct_int8": 100*n_int8/total,
        "median_ratio": float(np.nanmedian(all_ratios)),
        "max_ratio": float(max(all_ratios)),
        "seq_lens": seq_lens_used,
        "per_layer_fp16_pct": layer_fp16_pct,
        "per_layer_kv_outlier_pct": layer_ch_pct,
        "n_layers": n_layers, "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
    }
    with open(os.path.join(os.path.dirname(__file__), "phi2_validation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: phi2_validation_stats.json", flush=True)

    print("\n" + "=" * 68, flush=True)
    print("Done.", flush=True)
    print("=" * 68, flush=True)
