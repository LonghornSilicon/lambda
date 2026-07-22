"""Validate ratio signal on real LLM attention patterns (Qwen2-0.5B).

Downloads Qwen2-0.5B, runs forward passes on diverse prompts with eager
attention (so we get actual attention weights), then computes:
  - Entropy distribution per layer/head
  - max_prob * N (post-softmax proxy for pre-softmax ratio signal)
  - Fraction of heads that would be flagged as FP16 vs INT8

Compares to synthetic benchmark results and updates the dashboard.
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
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2-0.5B"

PROMPTS = [
    # Natural language — varied attention patterns
    "The transformer architecture introduced in 'Attention is All You Need' revolutionized natural language processing by replacing recurrence with self-attention mechanisms.",
    # Code — local dependencies dominate
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    # Math — dense cross-token reasoning
    "Prove that the square root of 2 is irrational. Assume for contradiction that sqrt(2) = p/q where p and q are integers with no common factor.",
    # Repetitive — tests periodic attention
    "one two three four five one two three four five one two three four five one two three four five",
    # Long context — tests how attention spreads
    "In the beginning God created the heavens and the earth. Now the earth was formless and empty, darkness was over the surface of the deep, and the Spirit of God was hovering over the waters. And God said, Let there be light, and there was light.",
]


def load_model():
    print(f"Loading {MODEL_ID}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",  # exposes attention weights
    )
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)
    print(f"  Layers: {model.config.num_hidden_layers}", flush=True)
    print(f"  Heads:  {model.config.num_attention_heads}", flush=True)
    print(f"  d_head: {model.config.hidden_size // model.config.num_attention_heads}", flush=True)
    return model, tokenizer


def collect_attention_stats(model, tokenizer):
    """Run all prompts, collect per-layer per-head stats."""
    from transformers import AutoTokenizer

    all_entropy   = []   # one float per (layer, head, prompt)
    all_max_prob  = []   # max attention weight per row, averaged
    all_conc      = []   # concentration: max_prob * seq_len ≈ ratio proxy

    print(f"\nRunning {len(PROMPTS)} prompts...", flush=True)
    for p_idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Prompt {p_idx+1}: {seq_len} tokens — {prompt[:50]}...", flush=True)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (1, H, N, N) per layer
        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (1, H, N, N) float32 post-softmax probabilities
            P = attn[0].float()  # (H, N, N)
            H, N, _ = P.shape

            for h in range(H):
                ph = P[h]  # (N, N)

                # Entropy per row, averaged
                log_p = torch.log(ph + 1e-10)
                ent = -(ph * log_p).sum(dim=-1).mean().item()

                # Max probability per row, averaged
                mp = ph.max(dim=-1).values.mean().item()

                # Concentration proxy: max_prob * N
                # For uniform dist: mp = 1/N → conc = 1
                # For peaked dist:  mp ≈ 1   → conc = N
                conc = mp * N

                all_entropy.append(ent)
                all_max_prob.append(mp)
                all_conc.append(conc)

    return all_entropy, all_max_prob, all_conc


def plot_results(all_entropy, all_max_prob, all_conc, n_layers, n_heads):
    """Generate figure for the dashboard."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Real LLM Attention Patterns — Qwen2-0.5B", fontsize=13, fontweight="bold")

    # --- Entropy distribution ---
    ax = axes[0]
    ax.hist(all_entropy, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(2.0, color="red", linestyle="--", linewidth=1.5, label="Threshold = 2.0")
    n_low  = sum(1 for e in all_entropy if e < 2.0)
    n_high = sum(1 for e in all_entropy if e >= 2.0)
    ax.set_xlabel("Attention Entropy (nats)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Entropy Distribution\nFP16-flagged: {n_low} ({100*n_low/len(all_entropy):.1f}%)  "
                 f"INT8-safe: {n_high} ({100*n_high/len(all_entropy):.1f}%)")
    ax.legend(fontsize=9)

    # --- Concentration proxy distribution ---
    ax = axes[1]
    log_conc = [math.log10(max(c, 0.1)) for c in all_conc]
    ax.hist(log_conc, bins=60, color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(math.log10(10), color="red", linestyle="--", linewidth=1.5, label="Threshold = 10")
    n_fp16 = sum(1 for c in all_conc if c > 10)
    n_int8 = sum(1 for c in all_conc if c <= 10)
    ax.set_xlabel("log₁₀(max_prob × seq_len)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Concentration Proxy (≈ ratio signal)\nFP16-flagged: {n_fp16} ({100*n_fp16/len(all_conc):.1f}%)  "
                 f"INT8-safe: {n_int8} ({100*n_int8/len(all_conc):.1f}%)")
    ax.legend(fontsize=9)

    # --- Entropy vs concentration scatter ---
    ax = axes[2]
    colors = ["#DD8452" if e < 2.0 else "#4C72B0" for e in all_entropy]
    ax.scatter(all_entropy, log_conc, c=colors, alpha=0.3, s=8)
    ax.axvline(2.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(math.log10(10), color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Entropy (nats)", fontsize=11)
    ax.set_ylabel("log₁₀(concentration)", fontsize=11)
    ax.set_title("Entropy vs Concentration\nOrange=FP16, Blue=INT8")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "real_llm_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}", flush=True)
    return out_path


def update_dashboard(stats_summary):
    """Append real-LLM section to the existing dashboard HTML."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(dashboard_path):
        print("  Dashboard not found, skipping update.", flush=True)
        return

    import base64
    fig_path = os.path.join(FIGURES_DIR, "real_llm_validation.png")
    with open(fig_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    section_html = f"""
<!-- Real LLM Validation Section (auto-injected) -->
<div style="margin:30px 0; padding:20px; background:#f8f9fa; border-radius:8px; border-left:4px solid #4C72B0;">
  <h2 style="color:#2c3e50; margin-top:0;">Real LLM Validation — Qwen2-0.5B</h2>
  <p style="color:#555; font-size:14px;">
    <strong>Model:</strong> {MODEL_ID} &nbsp;|&nbsp;
    <strong>FP16-flagged heads:</strong> {stats_summary['pct_fp16']:.1f}% &nbsp;|&nbsp;
    <strong>INT8-safe heads:</strong> {stats_summary['pct_int8']:.1f}% &nbsp;|&nbsp;
    <strong>Total (layer×head×prompt):</strong> {stats_summary['total']}
  </p>
  <p style="color:#555; font-size:14px;">
    Median entropy: {stats_summary['median_entropy']:.3f} nats &nbsp;|&nbsp;
    Low-entropy heads (&lt;2.0): {stats_summary['n_low_entropy']} &nbsp;|&nbsp;
    Bimodal gap present: {'Yes ✓' if stats_summary['bimodal'] else 'Unclear'}
  </p>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
</div>
"""

    with open(dashboard_path, "r") as f:
        html = f.read()

    # Insert before closing body tag
    if "Real LLM Validation Section" in html:
        # Already has it — replace
        import re
        html = re.sub(
            r"<!-- Real LLM Validation Section.*?</div>\s*(?=<)",
            section_html,
            html, flags=re.DOTALL
        )
    else:
        html = html.replace("</body>", section_html + "\n</body>")

    with open(dashboard_path, "w") as f:
        f.write(html)
    print(f"  Dashboard updated: {dashboard_path}", flush=True)


if __name__ == "__main__":
    print("=" * 64, flush=True)
    print("Real LLM Activation Validation")
    print("=" * 64, flush=True)

    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads

    all_entropy, all_max_prob, all_conc = collect_attention_stats(model, tokenizer)

    # Summary stats
    total = len(all_entropy)
    n_low  = sum(1 for e in all_entropy if e < 2.0)
    n_high = total - n_low
    median_ent = float(np.median(all_entropy))
    n_fp16_conc = sum(1 for c in all_conc if c > 10)

    # Bimodal check: are low-entropy and high-entropy heads clearly separated?
    # Simple check: is the gap between 1.5 and 3.5 sparse?
    gap_count = sum(1 for e in all_entropy if 1.5 < e < 3.5)
    bimodal = gap_count < 0.1 * total

    print(f"\n=== Summary ===", flush=True)
    print(f"  Total (layer×head×prompt): {total}", flush=True)
    print(f"  Entropy < 2.0 (FP16-flagged): {n_low} ({100*n_low/total:.1f}%)", flush=True)
    print(f"  Entropy ≥ 2.0 (INT8-safe):    {n_high} ({100*n_high/total:.1f}%)", flush=True)
    print(f"  Median entropy: {median_ent:.3f} nats", flush=True)
    print(f"  Concentration proxy > 10:     {n_fp16_conc} ({100*n_fp16_conc/total:.1f}%)", flush=True)
    print(f"  Bimodal gap (1.5-3.5 sparse): {'Yes' if bimodal else 'No'}", flush=True)

    # Per-layer breakdown
    print(f"\n=== Per-Layer FP16% (averaged over heads and prompts) ===", flush=True)
    per_layer_ent = [[] for _ in range(n_layers)]
    idx = 0
    for p_idx in range(len(PROMPTS)):
        for layer_idx in range(n_layers):
            for h in range(n_heads):
                per_layer_ent[layer_idx].append(all_entropy[idx])
                idx += 1

    for layer_idx, ents in enumerate(per_layer_ent):
        pct_fp16 = 100 * sum(1 for e in ents if e < 2.0) / len(ents)
        bar = "█" * int(pct_fp16 / 5)
        print(f"  Layer {layer_idx:2d}: {pct_fp16:5.1f}% FP16  {bar}", flush=True)

    stats_summary = {
        "total": total,
        "n_low_entropy": n_low,
        "pct_fp16": 100 * n_low / total,
        "pct_int8": 100 * n_high / total,
        "median_entropy": median_ent,
        "bimodal": bimodal,
    }

    plot_results(all_entropy, all_max_prob, all_conc, n_layers, n_heads)
    update_dashboard(stats_summary)

    # Save JSON
    out_json = os.path.join(os.path.dirname(__file__), "real_llm_stats.json")
    with open(out_json, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"  Stats saved: {out_json}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("Validation complete. Check dashboard for figures.", flush=True)
    print("=" * 64, flush=True)
