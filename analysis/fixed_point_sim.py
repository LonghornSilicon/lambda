"""Fixed-point simulation of the ratio threshold decision.

The ratio check in hardware is NOT a division. It is:

    max(|S|) > threshold × mean(|S|)
  = max(|S|) > threshold × sum(|S|) / N
  = max(|S|) × N > threshold × sum(|S|)

With N = BLOCK_M × BLOCK_N (compile-time constant, power of 2):
  - max × N    is a left shift   (free)
  - 10 × sum   is sum<<3 + sum<<1 (two shifts + one adder)
  - comparison is one comparator

This script quantizes the attention scores S to N-bit integers at various
bit widths and checks whether the integer comparison gives the same FP16/INT8
decision as the float reference. A "dangerous miss" is calling INT8 when the
float reference says FP16 (RMSE catastrophe). A "false positive" is calling
FP16 when INT8 would be fine (wasteful but safe).
"""

import os, sys, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RATIO_THRESHOLD = 10.0   # float reference threshold
BLOCK_M = BLOCK_N = 64  # tile dimensions (matches our kernel config for d=64)
N_TILE  = BLOCK_M * BLOCK_N   # = 4096; left-shift amount = 12 bits

BIT_WIDTHS = [4, 6, 8, 10, 12, 16]   # score representation widths to test

# ---------------------------------------------------------------------------
# Score tile generation — reproduces the distributions from our benchmarks
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

def make_tiles(n_tiles=8000):
    """Generate a mix of INT8-safe and FP16-requiring score tiles.

    INT8-safe  (ratio ≤ 10): uniform-ish scores, median ratio ≈ 5
    FP16-req   (ratio > 10): one or few dominant scores, median ratio ≈ 957
    Mix matches our benchmark: ~8% FP16, ~92% INT8.
    """
    tiles = []
    n_fp16  = int(n_tiles * 0.08)
    n_int8  = n_tiles - n_fp16

    # INT8-safe: scores drawn from a narrow uniform distribution
    for _ in range(n_int8):
        s = rng.uniform(-3.0, 3.0, size=(BLOCK_M, BLOCK_N)).astype(np.float32)
        tiles.append(("int8", s))

    # FP16-requiring: one dominant score spike embedded in background noise
    for _ in range(n_fp16):
        s = rng.uniform(-1.0, 1.0, size=(BLOCK_M, BLOCK_N)).astype(np.float32)
        # Plant a spike: 1-4 elements get a value 10-50× larger than the rest
        n_spikes = rng.integers(1, 5)
        idx_m = rng.integers(0, BLOCK_M, size=n_spikes)
        idx_n = rng.integers(0, BLOCK_N, size=n_spikes)
        spike_val = rng.uniform(15.0, 60.0)
        s[idx_m, idx_n] = spike_val * rng.choice([-1, 1])
        tiles.append(("fp16", s))

    rng.shuffle(tiles)   # type: ignore[arg-type]
    return tiles


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------

def float_decision(s: np.ndarray) -> str:
    """Reference float decision (what our Triton kernel does)."""
    abs_s = np.abs(s)
    ratio = abs_s.max() / (abs_s.mean() + 1e-6)
    return "fp16" if ratio > RATIO_THRESHOLD else "int8"


def fixed_point_decision(s: np.ndarray, n_bits: int) -> str:
    """Integer-only decision using the no-division formulation.

    Hardware:
      1. max_acc: running max tree over |S| values (N-bit register)
      2. sum_acc: adder tree over |S| values (N + log2(N_TILE) bit register)
      3. Compare: (max_acc << log2(N_TILE)) > (RATIO_THRESHOLD * sum_acc)
         which is:  max_acc * N_TILE > 10 * sum_acc
    """
    # Quantize scores to N-bit integers (symmetric, per-tile scale)
    max_abs = np.abs(s).max()
    if max_abs < 1e-9:
        return "int8"   # all-zero tile is always safe

    max_int = (1 << (n_bits - 1)) - 1   # e.g. 127 for 8-bit
    scale   = max_abs / max_int           # float scale factor (stored in a register)

    s_int = np.round(np.clip(s / scale, -max_int, max_int)).astype(np.int64)
    abs_int = np.abs(s_int)

    max_acc = int(abs_int.max())   # N-bit register
    sum_acc = int(abs_int.sum())   # (N + 12)-bit register for 64×64 tiles

    # Compare without division: max × N_TILE > threshold × sum
    # max × 4096 = max << 12  (free in hardware)
    # 10 × sum   = (sum << 3) + (sum << 1)  (two shifts + one adder)
    lhs = max_acc * N_TILE             # max << 12
    rhs = int(RATIO_THRESHOLD) * sum_acc  # 10 × sum

    return "fp16" if lhs > rhs else "int8"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64, flush=True)
    print("Fixed-Point Ratio Threshold Simulation", flush=True)
    print("  Hardware formulation: max×N > 10×sum  (no divider)", flush=True)
    print(f"  Tile size: {BLOCK_M}×{BLOCK_N} = {N_TILE} elements", flush=True)
    print(f"  Threshold: {RATIO_THRESHOLD}", flush=True)
    print("=" * 64, flush=True)

    tiles = make_tiles(n_tiles=8000)
    total = len(tiles)
    n_gt_fp16 = sum(1 for label, _ in tiles if label == "fp16")
    n_gt_int8 = total - n_gt_fp16
    print(f"\n  Tiles: {total}  ({n_gt_fp16} FP16-ground-truth, {n_gt_int8} INT8-ground-truth)",
          flush=True)

    # Compute float reference for each tile
    float_decisions = []
    for _, s in tiles:
        float_decisions.append(float_decision(s))

    results = {}
    print(f"\n{'Bits':>6} {'Agreement':>12} {'Danger misses':>14} {'False pos':>10}", flush=True)
    print("-" * 48, flush=True)

    for n_bits in BIT_WIDTHS:
        agree = 0
        danger_miss = 0   # float=FP16, fixed=INT8 → catastrophic
        false_pos   = 0   # float=INT8, fixed=FP16 → safe but wasteful

        for (label, s), f_dec in zip(tiles, float_decisions):
            fx_dec = fixed_point_decision(s, n_bits)

            if fx_dec == f_dec:
                agree += 1
            elif f_dec == "fp16" and fx_dec == "int8":
                danger_miss += 1
            else:
                false_pos += 1

        pct_agree  = 100 * agree       / total
        pct_danger = 100 * danger_miss / total
        pct_fp     = 100 * false_pos   / total

        results[n_bits] = {
            "agreement_pct":   pct_agree,
            "danger_miss_pct": pct_danger,
            "danger_miss_n":   danger_miss,
            "false_pos_pct":   pct_fp,
            "false_pos_n":     false_pos,
        }

        danger_str = f"{danger_miss:4d} ({pct_danger:.2f}%)"
        fp_str     = f"{false_pos:4d} ({pct_fp:.2f}%)"
        safe_mark  = "✓ SAFE" if danger_miss == 0 else "✗ DANGEROUS"
        print(f"{n_bits:>6}    {pct_agree:>8.3f}%    {danger_str:>14}  {fp_str:>10}  {safe_mark}",
              flush=True)

    # Minimum safe bit width
    min_safe = next((b for b in BIT_WIDTHS if results[b]["danger_miss_n"] == 0), None)
    print(f"\n  Minimum safe bit width: {min_safe}-bit", flush=True)
    print(f"  Accumulator width at {min_safe}-bit: {min_safe + math.ceil(math.log2(N_TILE))}-bit "
          f"(score bits + {math.ceil(math.log2(N_TILE))} bits for tile size {N_TILE})", flush=True)
    print(f"\n  Hardware gate count for the comparison:", flush=True)
    print(f"    max register:    {min_safe}-bit  (running max tree — log2({N_TILE})={int(math.log2(N_TILE))} comparator levels)", flush=True)
    acc_bits = min_safe + math.ceil(math.log2(N_TILE))
    print(f"    sum register:    {acc_bits}-bit  (adder tree over {N_TILE} elements)", flush=True)
    print(f"    LHS = max << 12: {min_safe+12}-bit  (free — just wire routing)", flush=True)
    print(f"    RHS = 10×sum:    {acc_bits+4}-bit  (sum<<3 + sum<<1)", flush=True)
    print(f"    comparator:      {max(min_safe+12, acc_bits+4)}-bit  (one comparator, result = 1 bit)", flush=True)
    print(f"    total gate count: O(N log N) adder tree + O(log N) max tree + 1 comparator", flush=True)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fixed-Point Ratio Threshold Simulation\n"
                 "Hardware: max×N > 10×sum  (no divider, no division gate)",
                 fontsize=12, fontweight="bold")

    bits_list  = BIT_WIDTHS
    agree_list = [results[b]["agreement_pct"] for b in bits_list]
    danger_list= [results[b]["danger_miss_pct"] for b in bits_list]
    fp_list    = [results[b]["false_pos_pct"] for b in bits_list]

    ax = axes[0]
    ax.plot(bits_list, agree_list, "o-", color="#4C72B0", linewidth=2, markersize=7, label="Agreement %")
    ax.axhline(100, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="100% target")
    ax.set_xlabel("Score bit width (fixed-point)", fontsize=11)
    ax.set_ylabel("Agreement with float reference (%)", fontsize=11)
    ax.set_title("Agreement vs Bit Width", fontsize=11)
    ax.set_xticks(bits_list)
    ax.set_ylim(85, 101)
    ax.legend(fontsize=9)
    if min_safe:
        ax.axvline(min_safe, color="red", linestyle=":", linewidth=1.5,
                   label=f"Min safe: {min_safe}-bit")
        ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(bits_list, danger_list, color="#C44E52", alpha=0.85, label="Dangerous misses (→ RMSE catastrophe)")
    ax.bar(bits_list, fp_list, bottom=danger_list, color="#DD8452", alpha=0.85,
           label="False positives (wasteful but safe)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Score bit width (fixed-point)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("Error Breakdown vs Bit Width", fontsize=11)
    ax.set_xticks(bits_list)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fixed_point_sim.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out_path}", flush=True)

    # Save JSON
    out_json = os.path.join(os.path.dirname(__file__), "fixed_point_sim_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_tiles": total, "block_size": BLOCK_M,
            "threshold": RATIO_THRESHOLD,
            "hardware_formulation": "max*N > threshold*sum (no division)",
            "min_safe_bits": min_safe,
            "results": {str(b): results[b] for b in BIT_WIDTHS},
        }, f, indent=2)
    print(f"  Results saved: {out_json}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("Done.", flush=True)
    print("=" * 64, flush=True)
