"""Block size sweep — finds optimal tile dimensions for FPGA SRAM sizing.

Tests BLOCK_M × BLOCK_N in {32, 64, 128} and ratio thresholds in {6, 8, 10, 15, 20}.

Key questions:
  1. Does INT8 coverage change with tile granularity?
     (smaller tiles = more decisions, finer grain, but noisier mean estimate)
  2. Does the safe threshold shift at smaller block sizes?
     (32×32 has 1024 elements vs 4096 for 64×64 — mean is less stable)
  3. What register widths does each block size require?
     (sum accumulator = score_bits + log2(N_tile))
"""

import os, sys, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BLOCK_SIZES   = [32, 64, 128]
THRESHOLDS    = [6, 8, 10, 15, 20]
N_TILES       = 10_000
SCORE_BITS    = 8        # fixed-point width (proven safe in fixed_point_sim.py)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tile generation — same distributions as fixed_point_sim.py
# ---------------------------------------------------------------------------

def make_tiles(block_m, block_n, n_tiles=N_TILES, fp16_frac=0.08):
    n_fp16 = int(n_tiles * fp16_frac)
    n_int8 = n_tiles - n_fp16
    tiles = []

    for _ in range(n_int8):
        s = rng.uniform(-3.0, 3.0, size=(block_m, block_n)).astype(np.float32)
        tiles.append(("int8", s))

    for _ in range(n_fp16):
        s = rng.uniform(-1.0, 1.0, size=(block_m, block_n)).astype(np.float32)
        n_spikes = rng.integers(1, 5)
        idx_m = rng.integers(0, block_m, size=n_spikes)
        idx_n = rng.integers(0, block_n, size=n_spikes)
        s[idx_m, idx_n] = rng.uniform(15.0, 60.0) * rng.choice([-1, 1])
        tiles.append(("fp16", s))

    rng.shuffle(tiles)
    return tiles


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------

def float_ratio(s):
    abs_s = np.abs(s)
    return abs_s.max() / (abs_s.mean() + 1e-6)


def int_decision(s, threshold, n_bits=SCORE_BITS):
    """No-division integer comparison: max*N > threshold*sum."""
    max_abs = np.abs(s).max()
    if max_abs < 1e-9:
        return "int8", 0.0
    max_int_val = (1 << (n_bits - 1)) - 1
    scale = max_abs / max_int_val
    s_int = np.round(np.clip(s / scale, -max_int_val, max_int_val)).astype(np.int64)
    abs_int = np.abs(s_int)
    N = s.shape[0] * s.shape[1]
    lhs = int(abs_int.max()) * N
    rhs = int(threshold) * int(abs_int.sum())
    return ("fp16" if lhs > rhs else "int8"), float(lhs) / max(rhs, 1)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

results = {}

for bsize in BLOCK_SIZES:
    N_tile = bsize * bsize
    tiles = make_tiles(bsize, bsize)
    total = len(tiles)
    n_gt_fp16 = sum(1 for l, _ in tiles if l == "fp16")

    # Float ratios for each tile
    ratios = [float_ratio(s) for _, s in tiles]

    results[bsize] = {
        "n_tiles": total,
        "n_gt_fp16": n_gt_fp16,
        "N_tile": N_tile,
        "sum_bits": SCORE_BITS + math.ceil(math.log2(N_tile)),
        "comparator_bits": SCORE_BITS + math.ceil(math.log2(N_tile)) + 4,
        "ratio_stats": {
            "fp16_median": float(np.median([r for (l, _), r in zip(tiles, ratios) if l == "fp16"])),
            "int8_median": float(np.median([r for (l, _), r in zip(tiles, ratios) if l == "int8"])),
            "fp16_min":    float(np.min([r for (l, _), r in zip(tiles, ratios) if l == "fp16"])),
            "int8_max":    float(np.max([r for (l, _), r in zip(tiles, ratios) if l == "int8"])),
        },
        "thresholds": {},
    }

    print(f"\nBlock {bsize}×{bsize} (N={N_tile})  "
          f"sum_reg={SCORE_BITS + math.ceil(math.log2(N_tile))}-bit  "
          f"cmp={SCORE_BITS + math.ceil(math.log2(N_tile)) + 4}-bit", flush=True)
    fp16_med = results[bsize]["ratio_stats"]["fp16_median"]
    int8_med  = results[bsize]["ratio_stats"]["int8_median"]
    int8_max  = results[bsize]["ratio_stats"]["int8_max"]
    fp16_min  = results[bsize]["ratio_stats"]["fp16_min"]
    print(f"  FP16 tiles: median ratio = {fp16_med:.1f}  |  "
          f"INT8 tiles: median = {int8_med:.2f}  max = {int8_max:.2f}  |  "
          f"Gap: [{int8_max:.1f}, {fp16_min:.1f}]", flush=True)
    print(f"  {'Thresh':>7} {'Agree%':>8} {'Danger':>8} {'FalsePos':>9}", flush=True)
    print(f"  {'-'*38}", flush=True)

    for thresh in THRESHOLDS:
        agree = danger = fp_count = 0
        for (gt_label, s), ratio in zip(tiles, ratios):
            gt_fp16 = gt_label == "fp16"
            pred_label, _ = int_decision(s, thresh)
            pred_fp16 = pred_label == "fp16"
            if pred_fp16 == gt_fp16:
                agree += 1
            elif gt_fp16 and not pred_fp16:
                danger += 1   # missed FP16 → catastrophic
            else:
                fp_count += 1

        pct_agree  = 100 * agree   / total
        pct_danger = 100 * danger  / total
        pct_fp     = 100 * fp_count / total
        safe = "✓" if danger == 0 else "✗"

        print(f"  {thresh:>7}  {pct_agree:>7.2f}%  "
              f"{danger:>4} ({pct_danger:.2f}%)  "
              f"{fp_count:>4} ({pct_fp:.2f}%)  {safe}", flush=True)

        results[bsize]["thresholds"][thresh] = {
            "agreement_pct": pct_agree,
            "danger_n": danger,
            "danger_pct": pct_danger,
            "false_pos_n": fp_count,
            "false_pos_pct": pct_fp,
            "safe": danger == 0,
        }


# ---------------------------------------------------------------------------
# Per-block-size: what threshold range is safe?
# ---------------------------------------------------------------------------

print("\n=== Safe threshold range per block size ===", flush=True)
print(f"  {'Block':>8}  {'Min safe thresh':>16}  {'Max safe thresh':>16}  {'Sum reg':>8}  {'Cmp reg':>8}", flush=True)
for bsize in BLOCK_SIZES:
    safe_thresholds = [t for t in THRESHOLDS
                       if results[bsize]["thresholds"][t]["safe"]]
    min_safe = min(safe_thresholds) if safe_thresholds else "none"
    max_safe = max(safe_thresholds) if safe_thresholds else "none"
    sr = results[bsize]["sum_bits"]
    cr = results[bsize]["comparator_bits"]
    print(f"  {bsize}×{bsize:>3}  {str(min_safe):>16}  {str(max_safe):>16}  {sr:>7}-bit  {cr:>7}-bit", flush=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Block Size Sweep — Ratio Threshold vs Tile Granularity\n"
             "Hardware: max×N > threshold×sum  (no division)",
             fontsize=12, fontweight="bold")

colors_thresh = {6: "#C44E52", 8: "#DD8452", 10: "#4C72B0", 15: "#55A868", 20: "#8172B3"}

# Panel 1: Agreement % vs threshold for each block size
ax = axes[0]
x = THRESHOLDS
for bsize in BLOCK_SIZES:
    agree = [results[bsize]["thresholds"][t]["agreement_pct"] for t in THRESHOLDS]
    ax.plot(x, agree, "o-", linewidth=2, markersize=6, label=f"{bsize}×{bsize}")
ax.axhline(100, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="100% target")
ax.set_xlabel("Ratio threshold", fontsize=11)
ax.set_ylabel("Agreement with ground truth (%)", fontsize=11)
ax.set_title("Agreement vs Threshold\n(all block sizes)", fontsize=11)
ax.set_xticks(THRESHOLDS)
ax.set_ylim(88, 101)
ax.legend(fontsize=9)

# Panel 2: Dangerous miss count vs threshold
ax = axes[1]
for bsize in BLOCK_SIZES:
    danger = [results[bsize]["thresholds"][t]["danger_n"] for t in THRESHOLDS]
    ax.plot(x, danger, "o-", linewidth=2, markersize=6, label=f"{bsize}×{bsize}")
ax.axhline(0, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="Target: 0")
ax.set_xlabel("Ratio threshold", fontsize=11)
ax.set_ylabel("Dangerous misses (count)", fontsize=11)
ax.set_title("Dangerous Misses vs Threshold\n(FP16 called INT8 → RMSE catastrophe)", fontsize=11)
ax.set_xticks(THRESHOLDS)
ax.legend(fontsize=9)

# Panel 3: Ratio gap visualization per block size
ax = axes[2]
bsizes_labels = [f"{b}×{b}" for b in BLOCK_SIZES]
fp16_meds = [results[b]["ratio_stats"]["fp16_median"] for b in BLOCK_SIZES]
int8_maxs = [results[b]["ratio_stats"]["int8_max"]    for b in BLOCK_SIZES]
int8_meds = [results[b]["ratio_stats"]["int8_median"] for b in BLOCK_SIZES]

x_pos = np.arange(len(BLOCK_SIZES))
ax.bar(x_pos - 0.2, fp16_meds, 0.35, label="FP16 median ratio", color="#C44E52", alpha=0.85)
ax.bar(x_pos + 0.2, int8_maxs, 0.35, label="INT8 max ratio",    color="#4C72B0", alpha=0.85)
ax.axhline(10, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Threshold = 10")
ax.set_xticks(x_pos); ax.set_xticklabels(bsizes_labels, fontsize=10)
ax.set_ylabel("Ratio value", fontsize=11)
ax.set_title("Signal Gap per Block Size\n(FP16 median vs INT8 max — gap must straddle threshold)", fontsize=10)
ax.legend(fontsize=9)
for xi, (fm, im) in enumerate(zip(fp16_meds, int8_maxs)):
    ax.text(xi - 0.2, fm + 5, f"{fm:.0f}", ha="center", fontsize=8, color="#C44E52")
    ax.text(xi + 0.2, im + 5, f"{im:.1f}", ha="center", fontsize=8, color="#4C72B0")

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "block_size_sweep.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure: {out_path}", flush=True)

# Save JSON
out_json = os.path.join(os.path.dirname(__file__), "block_size_sweep_results.json")
with open(out_json, "w") as f:
    json.dump({
        "block_sizes": BLOCK_SIZES,
        "thresholds": THRESHOLDS,
        "score_bits": SCORE_BITS,
        "n_tiles_per_config": N_TILES,
        "results": {str(b): results[b] for b in BLOCK_SIZES},
    }, f, indent=2)
print(f"  JSON:   {out_json}", flush=True)
