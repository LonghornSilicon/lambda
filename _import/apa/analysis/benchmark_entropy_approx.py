"""Benchmark: can we replace entropy with cheaper hardware approximations?

For each 128×128 attention block we compare 4 precision signals:
  1. Full entropy (gold standard, requires exp+log)
  2. has_outlier flag (requires max, mean, std — needs sqrt)
  3. max_abs / mean_abs ratio (requires max, adder tree — no sqrt/log/exp)
  4. score variance (requires adder tree + multiply — no sqrt/log/exp)

For each approximation we sweep thresholds and report:
  - Decision agreement with the evolved policy (INT8 vs FP16)
  - RMSE penalty when a block is misclassified
  - Optimal threshold

Goal: find the simplest signal that gives 100% agreement with the evolved
policy, eliminating log/exp from the precision controller.
"""

import sys
import os
import math
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.workloads import ALL_WORKLOADS, generate_workload
from common.quantization import quantize_dequantize

BLOCK_SIZE = 128


# -----------------------------------------------------------------------
# Per-block feature extraction
# -----------------------------------------------------------------------

def extract_block_features(Q, K, block_size=BLOCK_SIZE):
    """Return per-block dict with entropy and all candidate approximations."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Tr = math.ceil(N / block_size)
    Tc = math.ceil(N / block_size)

    blocks = []
    for i in range(Tr):
        q_s = i * block_size
        q_e = min(q_s + block_size, N)
        Q_b = Q[:, :, q_s:q_e, :].float()

        for j in range(Tc):
            k_s = j * block_size
            k_e = min(k_s + block_size, N)
            K_b = K[:, :, k_s:k_e, :].float()

            S = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale  # [B,H,Br,Bc]

            # --- raw score stats (no transcendental functions) ---
            abs_S = S.abs()
            abs_max  = abs_S.max().item()
            abs_mean = abs_S.mean().item()
            abs_std  = abs_S.std().item()
            abs_var  = abs_S.var().item()
            score_range = (S.max() - S.min()).item()

            has_outlier = bool(abs_max > abs_mean + 10 * abs_std) if abs_std > 0 else False
            ratio = abs_max / (abs_mean + 1e-10)  # max/mean ratio, no sqrt

            # --- full entropy (requires softmax + log, expensive in HW) ---
            P = torch.softmax(S, dim=-1)
            log_P = torch.log(P + 1e-10)
            entropy = -(P * log_P).sum(dim=-1).mean().item()

            # Ground-truth evolved policy decision
            gt_fp16 = has_outlier and (entropy < 2.0)

            blocks.append({
                "block": (i, j),
                "entropy": entropy,
                "has_outlier": has_outlier,
                "ratio": ratio,           # abs_max / abs_mean  (no sqrt/log/exp)
                "abs_var": abs_var,       # variance            (no sqrt/log/exp)
                "score_range": score_range,
                "gt_fp16": gt_fp16,       # ground truth: True=FP16, False=INT8
            })
    return blocks


# -----------------------------------------------------------------------
# RMSE cost of a wrong precision decision
# -----------------------------------------------------------------------

def rmse_cost(Q, K, V, correct_prec, wrong_prec, block_size=BLOCK_SIZE):
    """Compute RMSE difference when one block uses wrong_prec instead of correct_prec."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Reference: full FP32 attention
    S_full = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    P_full = torch.softmax(S_full, dim=-1)
    O_ref  = torch.matmul(P_full, V.float())

    # Correct-precision output
    Qc = quantize_dequantize(Q, correct_prec)
    Kc = quantize_dequantize(K, correct_prec)
    Vc = quantize_dequantize(V, correct_prec)
    Sc = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale
    Pc = torch.softmax(Sc, dim=-1)
    O_correct = torch.matmul(Pc, Vc.float())

    # Wrong-precision output
    Qw = quantize_dequantize(Q, wrong_prec)
    Kw = quantize_dequantize(K, wrong_prec)
    Vw = quantize_dequantize(V, wrong_prec)
    Sw = torch.matmul(Qw.float(), Kw.float().transpose(-2, -1)) * scale
    Pw = torch.softmax(Sw, dim=-1)
    O_wrong = torch.matmul(Pw, Vw.float())

    rms_ref = O_ref.pow(2).mean().sqrt().item() + 1e-10
    rmse_correct = (O_correct - O_ref).pow(2).mean().sqrt().item() / rms_ref
    rmse_wrong   = (O_wrong   - O_ref).pow(2).mean().sqrt().item() / rms_ref
    return rmse_correct, rmse_wrong


# -----------------------------------------------------------------------
# Sweep thresholds for a given feature
# -----------------------------------------------------------------------

def sweep_threshold(all_blocks, feature_key, thresholds, invert=False):
    """
    For each threshold t, classify blocks as fp16 if feature > t (or < t if invert).
    Return (threshold, agreement_pct, n_wrong_fp16, n_wrong_int8) for each t.
    """
    results = []
    total = len(all_blocks)
    for t in thresholds:
        n_agree = 0
        n_wrong_fp16  = 0  # should be fp16 (gt=True), predicted int8
        n_wrong_int8  = 0  # should be int8 (gt=False), predicted fp16
        for b in all_blocks:
            v = b[feature_key]
            if invert:
                pred_fp16 = v < t
            else:
                pred_fp16 = v > t
            gt = b["gt_fp16"]
            if pred_fp16 == gt:
                n_agree += 1
            elif gt and not pred_fp16:
                n_wrong_fp16 += 1   # dangerous: used INT8 when should be FP16
            else:
                n_wrong_int8 += 1   # wasteful: used FP16 when INT8 would have been fine
        results.append({
            "threshold": t,
            "agreement": 100 * n_agree / total,
            "wrong_fp16": n_wrong_fp16,   # dangerous misses
            "wrong_int8": n_wrong_int8,   # safe but wasteful
        })
    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("Entropy Approximation Benchmark — Hardware Simplification Study")
    print("=" * 72, flush=True)

    # --- Collect block-level features across all 12 workloads ---
    print("\nExtracting block features from all 12 workloads...", flush=True)
    all_blocks = []
    workload_blocks = []  # per-workload for RMSE analysis

    for idx, cfg in enumerate(ALL_WORKLOADS):
        Q, K, V = generate_workload(cfg, seed=42 + idx)
        blocks = extract_block_features(Q, K)
        all_blocks.extend(blocks)
        workload_blocks.append((cfg, Q, K, V, blocks))
        n_fp16 = sum(1 for b in blocks if b["gt_fp16"])
        print(f"  Workload {idx+1:2d}: N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
              f" outliers={str(cfg['outliers']):<5}  "
              f"{len(blocks):4d} blocks  FP16={n_fp16}  INT8={len(blocks)-n_fp16}", flush=True)

    total_blocks = len(all_blocks)
    total_fp16   = sum(1 for b in all_blocks if b["gt_fp16"])
    total_int8   = total_blocks - total_fp16
    print(f"\nTotal: {total_blocks} blocks  FP16={total_fp16} ({100*total_fp16/total_blocks:.1f}%)"
          f"  INT8={total_int8} ({100*total_int8/total_blocks:.1f}%)", flush=True)

    # --- Section 1: Feature distributions ---
    print("\n=== Section 1: Feature Distributions ===", flush=True)
    print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}", flush=True)
    print("-" * 65, flush=True)

    for key in ["entropy", "ratio", "abs_var"]:
        vals = [b[key] for b in all_blocks]
        print(f"  {key:<18} {min(vals):>10.4f} {max(vals):>10.4f}"
              f" {sum(vals)/len(vals):>10.4f} {torch.tensor(vals).std().item():>10.4f}", flush=True)

    # Entropy by class
    ent_fp16 = [b["entropy"] for b in all_blocks if b["gt_fp16"]]
    ent_int8 = [b["entropy"] for b in all_blocks if not b["gt_fp16"]]
    ratio_fp16 = [b["ratio"] for b in all_blocks if b["gt_fp16"]]
    ratio_int8 = [b["ratio"] for b in all_blocks if not b["gt_fp16"]]
    var_fp16 = [b["abs_var"] for b in all_blocks if b["gt_fp16"]]
    var_int8 = [b["abs_var"] for b in all_blocks if not b["gt_fp16"]]

    print(f"\n  Entropy   — FP16 blocks: mean={sum(ent_fp16)/len(ent_fp16):.3f}  "
          f"INT8 blocks: mean={sum(ent_int8)/len(ent_int8):.3f}", flush=True)
    print(f"  Ratio     — FP16 blocks: mean={sum(ratio_fp16)/len(ratio_fp16):.2f}  "
          f"INT8 blocks: mean={sum(ratio_int8)/len(ratio_int8):.2f}", flush=True)
    print(f"  Abs_var   — FP16 blocks: mean={sum(var_fp16)/len(var_fp16):.4f}  "
          f"INT8 blocks: mean={sum(var_int8)/len(var_int8):.4f}", flush=True)

    # --- Section 2: has_outlier flag alone ---
    print("\n=== Section 2: has_outlier Flag Alone (no entropy) ===", flush=True)
    n_agree_outlier = sum(1 for b in all_blocks if b["has_outlier"] == b["gt_fp16"])
    wrong_fp16 = sum(1 for b in all_blocks if b["gt_fp16"] and not b["has_outlier"])
    wrong_int8 = sum(1 for b in all_blocks if not b["gt_fp16"] and b["has_outlier"])
    print(f"  Agreement:    {100*n_agree_outlier/total_blocks:.2f}%  ({n_agree_outlier}/{total_blocks})", flush=True)
    print(f"  Miss FP16:    {wrong_fp16}  (dangerous — INT8 used when FP16 needed)", flush=True)
    print(f"  False FP16:   {wrong_int8}  (safe — FP16 used when INT8 would work)", flush=True)

    # --- Section 3: max/mean ratio threshold sweep ---
    print("\n=== Section 3: max/mean Ratio Threshold Sweep (no sqrt/log/exp) ===", flush=True)
    print(f"  Hardware: one max tree + one adder tree + one comparator", flush=True)
    print(f"\n  {'Threshold':>10} {'Agreement':>10} {'Miss FP16':>10} {'False FP16':>11}", flush=True)
    print("  " + "-" * 46, flush=True)

    ratio_thresholds = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
    ratio_results = sweep_threshold(all_blocks, "ratio", ratio_thresholds, invert=False)
    best_ratio = max(ratio_results, key=lambda r: r["agreement"])
    for r in ratio_results:
        marker = " <-- best" if r["threshold"] == best_ratio["threshold"] else ""
        print(f"  {r['threshold']:>10}  {r['agreement']:>9.2f}%"
              f"  {r['wrong_fp16']:>9}  {r['wrong_int8']:>10}{marker}", flush=True)

    # --- Section 4: variance threshold sweep ---
    print("\n=== Section 4: Score Variance Threshold Sweep (no sqrt/log/exp) ===", flush=True)
    print(f"  Hardware: adder tree + multiply-accumulate + one comparator", flush=True)
    print(f"\n  {'Threshold':>12} {'Agreement':>10} {'Miss FP16':>10} {'False FP16':>11}", flush=True)
    print("  " + "-" * 48, flush=True)

    var_vals = sorted([b["abs_var"] for b in all_blocks])
    # pick thresholds spanning the range
    var_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    var_results = sweep_threshold(all_blocks, "abs_var", var_thresholds, invert=True)
    best_var = max(var_results, key=lambda r: r["agreement"])
    for r in var_results:
        marker = " <-- best" if r["threshold"] == best_var["threshold"] else ""
        print(f"  {r['threshold']:>12.3f}  {r['agreement']:>9.2f}%"
              f"  {r['wrong_fp16']:>9}  {r['wrong_int8']:>10}{marker}", flush=True)

    # --- Section 5: Entropy threshold sweep (to confirm 2.0 is optimal) ---
    print("\n=== Section 5: Entropy Threshold Sweep (gold standard, for reference) ===", flush=True)
    print(f"  Hardware: softmax + log = 32K transcendental ops per block", flush=True)

    # entropy is combined with has_outlier, so we need a special sweep
    ent_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    print(f"\n  {'Threshold':>10} {'Agreement':>10} {'Miss FP16':>10} {'False FP16':>11}", flush=True)
    print("  " + "-" * 46, flush=True)
    for t in ent_thresholds:
        n_agree = 0
        wrong_fp16_e = 0
        wrong_int8_e = 0
        for b in all_blocks:
            pred = b["has_outlier"] and b["entropy"] < t
            gt = b["gt_fp16"]
            if pred == gt:
                n_agree += 1
            elif gt and not pred:
                wrong_fp16_e += 1
            else:
                wrong_int8_e += 1
        agr = 100 * n_agree / total_blocks
        marker = " <-- evolved policy" if t == 2.0 else ""
        print(f"  {t:>10.1f}  {agr:>9.2f}%  {wrong_fp16_e:>9}  {wrong_int8_e:>10}{marker}", flush=True)

    # --- Section 6: RMSE cost of wrong decisions ---
    print("\n=== Section 6: RMSE Cost of Wrong Precision Decisions ===", flush=True)
    print("  (What's the accuracy penalty for one misclassified block?)", flush=True)
    print(f"\n  {'Workload':<35} {'INT8→FP16 cost':>15} {'FP16→INT8 cost':>15}", flush=True)
    print("  " + "-" * 68, flush=True)

    for cfg, Q, K, V, blocks in workload_blocks:
        correct_prec = "fp16" if cfg["outliers"] else "int8"
        wrong_prec   = "int8" if cfg["outliers"] else "fp16"
        rmse_c, rmse_w = rmse_cost(Q, K, V, correct_prec, wrong_prec)

        label = (f"N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
                 f"{' outliers' if cfg['outliers'] else ''}")

        if cfg["outliers"]:
            print(f"  {label:<35} {'(correct)':>15} {rmse_w:>14.6f}  DANGEROUS", flush=True)
        else:
            print(f"  {label:<35} {rmse_w:>14.6f}  waste {'(correct)':>14}", flush=True)

    # --- Section 7: Summary ---
    print("\n=== Section 7: Summary — Hardware Recommendation ===", flush=True)
    print(flush=True)

    approx_results = [
        ("has_outlier only",       100*n_agree_outlier/total_blocks, "sqrt (for std)",        "max+adder+divider+sqrt"),
        (f"ratio > {best_ratio['threshold']}",
                                   best_ratio["agreement"],          "none",                  "max+adder+divider"),
        (f"abs_var < {best_var['threshold']}",
                                   best_var["agreement"],            "none",                  "adder+multiply"),
        ("entropy < 2.0 + outlier",100.0,                            "exp+log (32K per block)","full softmax+log"),
    ]

    print(f"  {'Approximation':<28} {'Agreement':>10} {'Extra HW':>20} {'Operations':>30}", flush=True)
    print("  " + "-" * 93, flush=True)
    for name, agr, extra, ops in approx_results:
        print(f"  {name:<28} {agr:>9.2f}%  {extra:>20}  {ops:>30}", flush=True)

    print(flush=True)
    print("  Miss FP16 = dangerous (INT8 where FP16 needed → possible output corruption)", flush=True)
    print("  False FP16 = safe but wasteful (FP16 where INT8 would compress)", flush=True)
    print("=" * 72, flush=True)
