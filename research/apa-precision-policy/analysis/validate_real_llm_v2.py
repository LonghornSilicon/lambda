"""Real LLM validation v2 — pre-softmax scores + KV channel outliers + long sequences.

Fixes three problems with v1:
  1. Post-softmax entropy was the wrong metric — hooks q/k/v_proj outputs directly
  2. Short sequences (20-54 tokens) forced artificial concentration — uses 512+ tokens
  3. Didn't check K,V channel outliers (the actual INT8 failure mode)

Measures:
  A. Pre-softmax ratio = max(|S|) / mean(|S|) per tile — the actual signal on the chip
  B. K,V channel outlier rate — fraction of channels with max/mean > 10
  C. Per-layer breakdown of both, with 512-token sequences
"""

import sys
import os
import math
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODEL_ID    = "Qwen/Qwen2-0.5B"
BLOCK_SIZE  = 64   # tile size for score computation (matches d=64 kernel config)
RATIO_THRESHOLD = 10.0

# Long prompts — enough tokens to get 512+ per prompt
LONG_PROMPTS = [
    # Public domain prose (varied attention pattern)
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
    # Code — local structure dominates
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
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
""" * 2),
]


# ---------------------------------------------------------------------------
# Hooks for pre-softmax scores and KV activations
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Registers hooks on q/k/v projections to capture raw activations."""

    def __init__(self, model):
        self.hooks   = []
        self.q_acts  = {}   # layer_idx -> tensor [B, seq, q_dim]
        self.k_acts  = {}   # layer_idx -> tensor [B, seq, kv_dim]
        self.v_acts  = {}   # layer_idx -> tensor [B, seq, kv_dim]
        self._register(model)

    def _register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(d, key):
                def h(module, inp, out):
                    d[key] = out.detach().float()
                return h

            self.hooks.append(
                attn.q_proj.register_forward_hook(make_hook(self.q_acts, layer_idx))
            )
            self.hooks.append(
                attn.k_proj.register_forward_hook(make_hook(self.k_acts, layer_idx))
            )
            self.hooks.append(
                attn.v_proj.register_forward_hook(make_hook(self.v_acts, layer_idx))
            )

    def clear(self):
        self.q_acts.clear()
        self.k_acts.clear()
        self.v_acts.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def compute_tile_ratios(Q_raw, K_raw, num_q_heads, num_kv_heads, head_dim, block_size=BLOCK_SIZE):
    """Compute pre-softmax ratio per tile.

    Q_raw: [B, seq, num_q_heads * head_dim]
    K_raw: [B, seq, num_kv_heads * head_dim]
    Returns list of ratios, one per (q_block, k_block) pair.
    """
    B, N, _ = Q_raw.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Reshape to [B, heads, seq, head_dim]
    Q = Q_raw.view(B, N, num_q_heads, head_dim).permute(0, 2, 1, 3)   # [B, Hq, N, d]
    K = K_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)  # [B, Hkv, N, d]

    # Expand KV heads for GQA
    if num_kv_heads != num_q_heads:
        repeats = num_q_heads // num_kv_heads
        K = K.repeat_interleave(repeats, dim=1)  # [B, Hq, N, d]

    ratios = []
    n_blocks = math.ceil(N / block_size)
    for i in range(n_blocks):
        q_s = i * block_size
        q_e = min(q_s + block_size, N)
        Q_b = Q[:, :, q_s:q_e, :]  # [B, H, Bq, d]

        for j in range(n_blocks):
            k_s = j * block_size
            k_e = min(k_s + block_size, N)
            K_b = K[:, :, k_s:k_e, :]  # [B, H, Bk, d]

            S = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale  # [B, H, Bq, Bk]
            abs_S = S.abs()
            s_max  = abs_S.max().item()
            s_mean = abs_S.mean().item()
            ratio  = s_max / (s_mean + 1e-6)
            ratios.append(ratio)

    return ratios


def compute_channel_outliers(K_raw, V_raw, num_kv_heads, head_dim):
    """Check for outlier channels in K and V.

    A channel is an outlier if max(|channel|) / mean(|channel|) > 10.
    Returns fraction of outlier channels.
    """
    B, N, _ = K_raw.shape
    K = K_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 3, 1)  # [B, H, d, N]
    V = V_raw.view(B, N, num_kv_heads, head_dim).permute(0, 2, 3, 1)

    total_channels = 0
    outlier_channels = 0

    for tensor in [K, V]:
        # Per channel: max over (B, H, N) dimensions for each d
        # tensor: [B, H, d, N]
        ch_max  = tensor.abs().max(dim=-1).values.max(dim=0).values.max(dim=0).values  # [d]
        ch_mean = tensor.abs().mean(dim=-1).mean(dim=0).mean(dim=0)  # [d]
        ch_ratio = ch_max / (ch_mean + 1e-6)

        total_channels += head_dim
        outlier_channels += (ch_ratio > RATIO_THRESHOLD).sum().item()

    return outlier_channels / total_channels, outlier_channels, total_channels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68, flush=True)
    print("Real LLM Validation v2 — Pre-Softmax Scores + KV Channels")
    print("=" * 68, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    n_layers    = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim    = cfg.hidden_size // num_q_heads
    print(f"  Layers={n_layers}  Q-heads={num_q_heads}  KV-heads={num_kv_heads}"
          f"  head_dim={head_dim}", flush=True)

    capture = AttentionCapture(model)

    # Aggregate results
    all_ratios     = []         # all tile ratios across all layers/prompts
    per_layer_ratios = [[] for _ in range(n_layers)]
    per_layer_ch_out = []       # (pct, n_out, n_total) per layer
    seq_lens_used  = []

    for prompt_name, prompt_text in LONG_PROMPTS:
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=512).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        seq_lens_used.append(seq_len)
        print(f"\nPrompt '{prompt_name}': {seq_len} tokens", flush=True)

        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(n_layers):
            Q_raw = capture.q_acts[layer_idx].cpu()  # [1, seq, q_dim]
            K_raw = capture.k_acts[layer_idx].cpu()  # [1, seq, kv_dim]
            V_raw = capture.v_acts[layer_idx].cpu()

            # A: tile ratios (pre-softmax)
            ratios = compute_tile_ratios(Q_raw, K_raw, num_q_heads, num_kv_heads, head_dim)
            all_ratios.extend(ratios)
            per_layer_ratios[layer_idx].extend(ratios)

            # B: KV channel outliers
            pct, n_out, n_total = compute_channel_outliers(K_raw, V_raw, num_kv_heads, head_dim)
            if len(per_layer_ch_out) <= layer_idx:
                per_layer_ch_out.append([])
            per_layer_ch_out[layer_idx].append(pct)

        capture.clear()
        print(f"  Layers processed: {n_layers}", flush=True)

    capture.remove()

    # --- Summary ---
    print("\n=== A: Pre-Softmax Ratio Distribution ===", flush=True)
    n_fp16  = sum(1 for r in all_ratios if r > RATIO_THRESHOLD)
    n_int8  = len(all_ratios) - n_fp16
    total   = len(all_ratios)
    print(f"  Total tiles:           {total}", flush=True)
    print(f"  Ratio > 10 (FP16):     {n_fp16} ({100*n_fp16/total:.1f}%)", flush=True)
    print(f"  Ratio ≤ 10 (INT8-safe):{n_int8} ({100*n_int8/total:.1f}%)", flush=True)
    print(f"  Median ratio:          {float(np.median(all_ratios)):.2f}", flush=True)
    print(f"  Max ratio:             {max(all_ratios):.1f}", flush=True)
    print(f"  Seq lengths used:      {seq_lens_used}", flush=True)

    print("\n=== B: KV Channel Outliers per Layer ===", flush=True)
    print(f"  {'Layer':>6} {'FP16 tiles%':>12} {'KV outlier channels%':>21}", flush=True)
    print("  " + "-" * 44, flush=True)
    for layer_idx in range(n_layers):
        r_list = per_layer_ratios[layer_idx]
        pct_fp16 = 100 * sum(1 for r in r_list if r > RATIO_THRESHOLD) / max(len(r_list), 1)
        pct_ch   = 100 * np.mean(per_layer_ch_out[layer_idx])
        bar      = "█" * int(pct_fp16 / 5)
        print(f"  {layer_idx:>6}  {pct_fp16:>10.1f}%  {pct_ch:>19.1f}%  {bar}", flush=True)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Real LLM Pre-Softmax Analysis — Qwen2-0.5B"
                 f"  (seq_len≈{int(np.mean(seq_lens_used))} tokens)", fontsize=12, fontweight="bold")

    # Ratio distribution
    ax = axes[0]
    log_ratios = [math.log10(max(r, 0.01)) for r in all_ratios]
    ax.hist(log_ratios, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(math.log10(RATIO_THRESHOLD), color="red", linestyle="--",
               linewidth=1.5, label=f"Threshold = {RATIO_THRESHOLD}")
    ax.set_xlabel("log₁₀(max|S| / mean|S|)  — pre-softmax ratio", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Tile Ratio Distribution\nFP16: {100*n_fp16/total:.1f}%   INT8: {100*n_int8/total:.1f}%")
    ax.legend(fontsize=9)

    # Per-layer FP16% for tiles
    ax = axes[1]
    layer_fp16_pct = [
        100 * sum(1 for r in per_layer_ratios[l] if r > RATIO_THRESHOLD) / max(len(per_layer_ratios[l]), 1)
        for l in range(n_layers)
    ]
    bars = ax.barh(range(n_layers), layer_fp16_pct, color="#DD8452", alpha=0.85)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("FP16 tiles (%)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title("Per-Layer FP16 Tile %\n(pre-softmax ratio > 10)")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Per-layer KV channel outliers
    ax = axes[2]
    layer_ch_pct = [100 * np.mean(per_layer_ch_out[l]) for l in range(n_layers)]
    ax.barh(range(n_layers), layer_ch_pct, color="#55A868", alpha=0.85)
    ax.axvline(10, color="red", linestyle="--", linewidth=1, alpha=0.7, label="10% threshold")
    ax.set_xlabel("Outlier KV channels (%)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title("Per-Layer KV Channel Outliers\n(max/mean > 10 per channel)")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "real_llm_v2.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out_path}", flush=True)

    # Update dashboard
    import base64
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(out_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    section = f"""
<!-- Real LLM v2 Section -->
<div style="margin:30px 0; padding:20px; background:#f0f4f8; border-radius:8px; border-left:4px solid #55A868;">
  <h2 style="color:#2c3e50; margin-top:0;">Real LLM Validation v2 — Pre-Softmax Scores (Qwen2-0.5B)</h2>
  <p style="color:#555; font-size:14px;">
    Seq length ≈{int(np.mean(seq_lens_used))} tokens &nbsp;|&nbsp;
    <strong>FP16 tiles (ratio&gt;10):</strong> {100*n_fp16/total:.1f}% &nbsp;|&nbsp;
    <strong>INT8-safe tiles:</strong> {100*n_int8/total:.1f}% &nbsp;|&nbsp;
    Total tiles: {total}
  </p>
  <img src="data:image/png;base64,{img_b64}"
       style="max-width:100%; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
</div>
"""

    if os.path.exists(dashboard_path):
        with open(dashboard_path) as f:
            html = f.read()
        if "Real LLM v2 Section" in html:
            import re
            html = re.sub(r"<!-- Real LLM v2 Section.*?</div>", section.strip(),
                          html, flags=re.DOTALL)
        else:
            html = html.replace("</body>", section + "\n</body>")
        with open(dashboard_path, "w") as f:
            f.write(html)
        print(f"  Dashboard updated.", flush=True)

    # Save JSON
    stats = {
        "total_tiles": total, "n_fp16": n_fp16, "n_int8": n_int8,
        "pct_fp16": 100*n_fp16/total, "pct_int8": 100*n_int8/total,
        "median_ratio": float(np.median(all_ratios)),
        "max_ratio": float(max(all_ratios)),
        "seq_lens": seq_lens_used,
        "per_layer_fp16_pct": layer_fp16_pct,
        "per_layer_kv_outlier_pct": layer_ch_pct,
    }
    with open(os.path.join(os.path.dirname(__file__), "real_llm_v2_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 68, flush=True)
    print("Done.", flush=True)
    print("=" * 68, flush=True)
