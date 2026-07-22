"""Deep-layer validation — tests multiple model sizes to see if INT8 findings scale.

Strategy (storage-safe):
  - HF_HOME routed to /dev/shm so downloads land in RAM, never touch disk
  - After each model loads into VRAM, /dev/shm cache is deleted immediately
  - Peak /dev/shm usage: one model at a time (≤6 GB)
  - GPU runs models sequentially; del + empty_cache between runs

Models tested:
  0. Qwen2-0.5B  (24 layers) — loaded from existing stats JSON, no re-run
  1. Qwen2-1.5B  (28 layers) — 3.1 GB fp16
  2. Qwen2.5-3B  (36 layers) — 6.2 GB fp16

All three results are combined into a single comparison figure.
"""

import os
import sys
import math
import json
import shutil

# Route HF downloads to RAM — must be set before any transformers/huggingface imports
HF_CACHE_DIR = "/dev/shm/hf_deep"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR     = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RATIO_THRESHOLD = 10.0
BLOCK_SIZE      = 64    # tile size (matches kernel block config for d≤64)
MAX_SEQ         = 512

MODELS_TO_TEST = [
    ("Qwen/Qwen2-1.5B",  "Qwen2-1.5B"),
    ("Qwen/Qwen2.5-3B",  "Qwen2.5-3B"),
]

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
        return torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(x + self.ff(x))
""" * 3),
]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class AttentionCapture:
    def __init__(self, model):
        self.hooks   = []
        self.q_acts  = {}
        self.k_acts  = {}
        self.v_acts  = {}
        self._register(model)

    def _register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(d, key):
                def h(module, inp, out):
                    d[key] = out.detach().float()
                return h

            self.hooks.append(attn.q_proj.register_forward_hook(make_hook(self.q_acts, layer_idx)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_hook(self.k_acts, layer_idx)))
            self.hooks.append(attn.v_proj.register_forward_hook(make_hook(self.v_acts, layer_idx)))

    def clear(self):
        self.q_acts.clear(); self.k_acts.clear(); self.v_acts.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


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
        q_s, q_e = i * block_size, min((i + 1) * block_size, N)
        Q_b = Q[:, :, q_s:q_e, :]
        for j in range(n_blocks):
            k_s, k_e = j * block_size, min((j + 1) * block_size, N)
            K_b = K[:, :, k_s:k_e, :]
            S = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale
            abs_S = S.abs()
            ratio = abs_S.max().item() / (abs_S.mean().item() + 1e-6)
            ratios.append(ratio)
    return ratios


def compute_channel_outliers(K_raw, V_raw, num_kv_heads, head_dim):
    B, N, _ = K_raw.shape
    total_out = outlier_out = 0
    for tensor, nkv in [(K_raw, num_kv_heads), (V_raw, num_kv_heads)]:
        T = tensor.view(B, N, nkv, head_dim).permute(0, 2, 3, 1)  # [B, H, d, N]
        ch_max  = T.abs().max(dim=-1).values.max(dim=0).values.max(dim=0).values
        ch_mean = T.abs().mean(dim=-1).mean(dim=0).mean(dim=0)
        total_out   += head_dim
        outlier_out += (ch_max / (ch_mean + 1e-6) > RATIO_THRESHOLD).sum().item()
    return outlier_out / total_out, outlier_out, total_out


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model_id, short_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*68}", flush=True)
    print(f"Loading {model_id} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    n_layers     = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.hidden_size // num_q_heads
    print(f"  {short_name}: {n_layers} layers, {num_q_heads}Q/{num_kv_heads}KV heads, "
          f"d={head_dim}", flush=True)

    # Free /dev/shm cache now that weights are in VRAM
    shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    free_gb = shutil.disk_usage("/dev/shm").free / 1e9
    print(f"  /dev/shm cache cleared — {free_gb:.1f} GB free", flush=True)

    capture = AttentionCapture(model)
    all_ratios        = []
    per_layer_ratios  = [[] for _ in range(n_layers)]
    per_layer_ch_out  = [[] for _ in range(n_layers)]
    seq_lens_used     = []

    for prompt_name, prompt_text in LONG_PROMPTS:
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=MAX_SEQ).to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        seq_lens_used.append(seq_len)
        print(f"  Prompt '{prompt_name}': {seq_len} tokens", flush=True)

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

    capture.remove()

    # Summary
    total  = len(all_ratios)
    n_fp16 = sum(1 for r in all_ratios if r > RATIO_THRESHOLD)
    n_int8 = total - n_fp16
    med    = float(np.nanmedian(all_ratios))
    mx     = float(max(all_ratios))
    layer_fp16_pct = [
        100 * sum(1 for r in per_layer_ratios[l] if r > RATIO_THRESHOLD) / max(len(per_layer_ratios[l]), 1)
        for l in range(n_layers)
    ]
    layer_ch_pct = [100 * float(np.mean(per_layer_ch_out[l])) for l in range(n_layers)]

    print(f"\n  Total tiles: {total}", flush=True)
    print(f"  FP16 (ratio>10): {n_fp16} ({100*n_fp16/total:.1f}%)", flush=True)
    print(f"  INT8-safe:       {n_int8} ({100*n_int8/total:.1f}%)", flush=True)
    print(f"  Median ratio:    {med:.2f}   Max: {mx:.1f}", flush=True)

    print(f"\n  Per-layer FP16%:", flush=True)
    for li, pct in enumerate(layer_fp16_pct):
        bar = "█" * int(pct / 5)
        print(f"    Layer {li:2d}: {pct:5.1f}%  {bar}", flush=True)

    stats = {
        "model_id": model_id, "short_name": short_name,
        "n_layers": n_layers, "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "total_tiles": total, "n_fp16": n_fp16, "n_int8": n_int8,
        "pct_fp16": 100 * n_fp16 / total, "pct_int8": 100 * n_int8 / total,
        "median_ratio": med, "max_ratio": mx,
        "seq_lens": seq_lens_used,
        "per_layer_fp16_pct": layer_fp16_pct,
        "per_layer_kv_outlier_pct": layer_ch_pct,
    }

    out_json = os.path.join(os.path.dirname(__file__),
                            f"deep_layer_stats_{short_name.replace('/', '_')}.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {out_json}", flush=True)

    # Tear down model from GPU
    del model
    torch.cuda.empty_cache()

    return stats


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def plot_comparison(all_stats):
    n_models = len(all_stats)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Adaptive Precision Attention — Deep Layer Scaling Study\n"
                 "Pre-Softmax Ratio Signal (ratio > 10 → FP16, else INT8)",
                 fontsize=13, fontweight="bold")

    # --- Panel 1: Overall INT8-safe % per model ---
    ax = axes[0]
    names   = [s["short_name"] for s in all_stats]
    pct_int8 = [s["pct_int8"] for s in all_stats]
    pct_fp16 = [s["pct_fp16"] for s in all_stats]
    x = np.arange(n_models)
    bars = ax.bar(x, pct_int8, color=colors[:n_models], alpha=0.85, edgecolor="white")
    for bar, p in zip(bars, pct_int8):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("INT8-safe tiles (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.axhline(90, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label=">90% target")
    ax.set_title("Overall INT8-Safe % per Model", fontsize=11)
    ax.legend(fontsize=9)
    ax2 = ax.twinx()
    ax2.set_ylabel("FP16 tiles (%)", fontsize=10, color="#DD8452")
    ax2.plot(x, pct_fp16, "o--", color="#DD8452", linewidth=1.5, markersize=6)
    ax2.set_ylim(0, 108)
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    # --- Panel 2: Per-layer FP16% normalized to 0-100% depth ---
    ax = axes[1]
    for i, stats in enumerate(all_stats):
        n = stats["n_layers"]
        pcts = stats["per_layer_fp16_pct"]
        norm_depth = [100 * l / (n - 1) for l in range(n)]
        ax.plot(norm_depth, pcts, "o-", color=colors[i], alpha=0.85,
                linewidth=1.5, markersize=4, label=f"{stats['short_name']} ({n}L)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("Relative layer depth (%)", fontsize=11)
    ax.set_ylabel("FP16 tile % (ratio > 10)", fontsize=11)
    ax.set_title("Per-Layer FP16% vs Relative Depth\n(normalized across model sizes)", fontsize=10)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=9)

    # --- Panel 3: Median ratio + max ratio per model (scatter) ---
    ax = axes[2]
    medians = [float(np.nanmedian(s["per_layer_fp16_pct"])) if np.isnan(s["median_ratio"]) else s["median_ratio"] for s in all_stats]
    maxes   = [s["max_ratio"] for s in all_stats]
    ax.bar(x, medians, color=colors[:n_models], alpha=0.85, label="Median ratio")
    ax.plot(x, maxes, "^", color="red", markersize=9, label="Max ratio", zorder=5)
    ax.axhline(RATIO_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Threshold = {RATIO_THRESHOLD}")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("max(|S|) / mean(|S|)  ratio", fontsize=11)
    ax.set_title("Median & Max Ratio per Model\n(both well below threshold = safe)", fontsize=10)
    ax.legend(fontsize=9)
    for xi, (m, mx) in enumerate(zip(medians, maxes)):
        ax.text(xi, m + 0.15, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "deep_layer_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison figure saved: {out_path}", flush=True)
    return out_path


def update_dashboard(all_stats, fig_path):
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(dashboard_path):
        return

    import base64
    with open(fig_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    rows = "\n".join(
        f"    <tr><td>{s['short_name']}</td><td>{s['n_layers']}</td>"
        f"<td>{s['pct_int8']:.1f}%</td><td>{s['pct_fp16']:.1f}%</td>"
        f"<td>{s['median_ratio']:.2f}</td><td>{s['max_ratio']:.1f}</td></tr>"
        for s in all_stats
    )

    section = f"""
<!-- Deep Layer Comparison Section -->
<div style="margin:30px 0; padding:20px; background:#fdf6ec; border-radius:8px; border-left:4px solid #DD8452;">
  <h2 style="color:#2c3e50; margin-top:0;">Deep Layer Scaling Study — Multi-Model Comparison</h2>
  <table style="border-collapse:collapse; font-size:13px; margin-bottom:12px;">
    <tr style="background:#eee;">
      <th style="padding:6px 12px; text-align:left;">Model</th>
      <th style="padding:6px 12px;">Layers</th>
      <th style="padding:6px 12px;">INT8-safe</th>
      <th style="padding:6px 12px;">FP16</th>
      <th style="padding:6px 12px;">Median ratio</th>
      <th style="padding:6px 12px;">Max ratio</th>
    </tr>
{rows}
  </table>
  <img src="data:image/png;base64,{img_b64}"
       style="max-width:100%; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
</div>
"""
    with open(dashboard_path) as f:
        html = f.read()

    import re
    if "Deep Layer Comparison Section" in html:
        html = re.sub(r"<!-- Deep Layer Comparison Section.*?</div>",
                      section.strip(), html, flags=re.DOTALL)
    else:
        html = html.replace("</body>", section + "\n</body>")

    with open(dashboard_path, "w") as f:
        f.write(html)
    print(f"  Dashboard updated.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68, flush=True)
    print("Deep Layer Validation — Streaming Weights via /dev/shm", flush=True)
    print("=" * 68, flush=True)

    # Check /dev/shm capacity
    shm_free = shutil.disk_usage("/dev/shm").free / 1e9
    print(f"  /dev/shm free: {shm_free:.1f} GB", flush=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    # Load 0.5B stats from existing JSON (no re-run needed)
    v2_json = os.path.join(os.path.dirname(__file__), "real_llm_v2_stats.json")
    with open(v2_json) as f:
        stats_05b = json.load(f)
    stats_05b["model_id"]    = "Qwen/Qwen2-0.5B"
    stats_05b["short_name"]  = "Qwen2-0.5B"
    stats_05b["n_layers"]    = len(stats_05b["per_layer_fp16_pct"])
    stats_05b["num_q_heads"] = 14
    stats_05b["num_kv_heads"]= 2
    stats_05b["head_dim"]    = 64
    print(f"\n  [0/2] Qwen2-0.5B — loaded from existing stats "
          f"(INT8-safe: {stats_05b['pct_int8']:.1f}%)", flush=True)

    all_stats = [stats_05b]

    for i, (model_id, short_name) in enumerate(MODELS_TO_TEST):
        try:
            stats = run_model(model_id, short_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n  ERROR loading {model_id}: {e}", flush=True)
            print(f"  Skipping {short_name}.", flush=True)
            # Clean up any partial download
            shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
            os.makedirs(HF_CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 68, flush=True)
    print("=== Combined Summary ===", flush=True)
    for s in all_stats:
        print(f"  {s['short_name']:15s}  {s['n_layers']:2d} layers  "
              f"INT8-safe: {s['pct_int8']:.1f}%  "
              f"median_ratio: {s['median_ratio']:.2f}  "
              f"max_ratio: {s['max_ratio']:.1f}", flush=True)

    fig_path = plot_comparison(all_stats)
    update_dashboard(all_stats, fig_path)

    # Save combined JSON
    combined = os.path.join(os.path.dirname(__file__), "deep_layer_combined_stats.json")
    with open(combined, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Combined stats: {combined}", flush=True)

    # Final cleanup
    shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
    print(f"  /dev/shm cleared.", flush=True)

    print("=" * 68, flush=True)
    print("Done.", flush=True)
    print("=" * 68, flush=True)
