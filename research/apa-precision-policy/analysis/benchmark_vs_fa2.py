"""Benchmark: evolved precision policy vs FlashAttention-2 (PyTorch SDPA).

Measures:
  1. Accuracy — RMSE vs FP64 reference for FA-2 and evolved policy
  2. KV-cache memory — quantized cache bytes vs standard FP16 cache
  3. Scaling — accuracy and memory savings across sequence lengths
"""

import sys
import os
import time
import math
import json

import torch
import torch.backends.cuda

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.workloads import ALL_WORKLOADS, generate_workload
from common.reference import reference_attention, mixed_precision_attention
from kv_cache.quantized_cache import QuantizedKVCache

# Load evolved policy
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evolved",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "phase1_policy", "openevolve_output", "best", "best_program.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evolved_policy = _mod.get_policy()


# -----------------------------------------------------------------------
# FA-2 wrapper (PyTorch SDPA dispatches to FA-2 on Ampere with CUDA)
# -----------------------------------------------------------------------

def fa2_attention(Q, K, V, causal=False):
    """PyTorch SDPA — dispatches to Flash Attention 2 on Ampere GPUs."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                Q.cuda().half(), K.cuda().half(), V.cuda().half(),
                is_causal=causal,
            ).float().cpu()
    except Exception:
        # Fall back to standard SDPA (math backend)
        return torch.nn.functional.scaled_dot_product_attention(
            Q.cuda().half(), K.cuda().half(), V.cuda().half(),
            is_causal=causal,
        ).float().cpu()


def fa2_available():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        Q = torch.randn(1, 1, 64, 64, dtype=torch.float16, device="cuda")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(Q, Q, Q)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------
# Benchmark 1: Accuracy comparison
# -----------------------------------------------------------------------

def benchmark_accuracy():
    print("\n=== Benchmark 1: Accuracy vs FP32 Reference ===", flush=True)
    print(f"{'Workload':<35} {'FA-2 RMSE':>12} {'Policy RMSE':>12} {'Match':>8}", flush=True)
    print("-" * 72, flush=True)

    fa2_ok = fa2_available()
    if not fa2_ok:
        print("  Flash SDP kernel not available — using standard SDPA instead.")

    results = []
    for idx, cfg in enumerate(ALL_WORKLOADS):
        Q, K, V = generate_workload(cfg, seed=42 + idx)

        with torch.no_grad():
            O_ref = reference_attention(Q, K, V, causal=cfg["causal"])

        # FA-2 via SDPA
        if fa2_ok:
            O_fa2 = fa2_attention(Q, K, V, causal=cfg["causal"])
        else:
            O_fa2 = torch.nn.functional.scaled_dot_product_attention(
                Q.cuda().half(), K.cuda().half(), V.cuda().half(), is_causal=cfg["causal"]
            ).float().cpu()

        # Evolved policy
        with torch.no_grad():
            O_ev, avg_bits = mixed_precision_attention(
                Q, K, V, evolved_policy, causal=cfg["causal"]
            )

        rms_ref = O_ref.float().pow(2).mean().sqrt().item() + 1e-10
        rmse_fa2 = (O_fa2.cpu() - O_ref.float()).pow(2).mean().sqrt().item() / rms_ref
        rmse_ev  = (O_ev  - O_ref.float()).pow(2).mean().sqrt().item() / rms_ref
        # How close is the evolved policy to FA-2 specifically
        rmse_vs_fa2 = (O_ev - O_fa2.cpu()).pow(2).mean().sqrt().item() / rms_ref

        label = (f"N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
                 f"{' causal' if cfg['causal'] else ''}"
                 f"{' outliers' if cfg['outliers'] else ''}")
        match = "✓" if rmse_vs_fa2 < 0.05 else "✗"
        print(f"  {label:<33} {rmse_fa2:>12.6f} {rmse_ev:>12.6f} {match:>8}", flush=True)

        results.append({
            "config": cfg,
            "rmse_fa2": rmse_fa2,
            "rmse_evolved": rmse_ev,
            "rmse_evolved_vs_fa2": rmse_vs_fa2,
            "avg_bits": avg_bits,
        })

    avg_fa2  = sum(r["rmse_fa2"] for r in results) / len(results)
    avg_ev   = sum(r["rmse_evolved"] for r in results) / len(results)
    avg_diff = sum(r["rmse_evolved_vs_fa2"] for r in results) / len(results)
    print("-" * 72, flush=True)
    print(f"  {'Average':<33} {avg_fa2:>12.6f} {avg_ev:>12.6f}", flush=True)
    print(f"\n  Avg RMSE between evolved policy and FA-2: {avg_diff:.6f}", flush=True)
    print(f"  (Both measured against FP32 reference)\n", flush=True)
    return results


# -----------------------------------------------------------------------
# Benchmark 2: KV-cache memory savings
# -----------------------------------------------------------------------

def benchmark_kv_memory():
    print("\n=== Benchmark 2: KV-Cache Memory Savings ===", flush=True)
    print(f"{'Workload':<35} {'FP16 bytes':>12} {'Quant bytes':>12} {'Ratio':>8} {'Bits':>6}", flush=True)
    print("-" * 78, flush=True)

    results = []
    for idx, cfg in enumerate(ALL_WORKLOADS):
        Q, K, V = generate_workload(cfg, seed=42 + idx)
        B, H, N, d = (cfg["batch"], cfg["num_heads"], cfg["seq_len"], cfg["head_dim"])

        # Standard FP16 KV cache: 2 tensors × B × H × N × d × 2 bytes
        fp16_bytes = 2 * B * H * N * d * 2

        # Quantized KV cache
        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)
        cache.update(K, V, layer_idx=0)
        stats = cache.memory_stats()
        quant_bytes = stats["total_bytes"]
        ratio = fp16_bytes / quant_bytes
        bits = stats["bits_per_element"]

        label = (f"N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
                 f"{' causal' if cfg['causal'] else ''}"
                 f"{' outliers' if cfg['outliers'] else ''}")
        print(f"  {label:<33} {fp16_bytes:>12,} {quant_bytes:>12,} {ratio:>7.2f}x {bits:>5.1f}", flush=True)

        results.append({
            "config": cfg,
            "fp16_bytes": fp16_bytes,
            "quant_bytes": quant_bytes,
            "compression_ratio": ratio,
            "bits_per_element": bits,
        })

    avg_ratio = sum(r["compression_ratio"] for r in results) / len(results)
    avg_bits  = sum(r["bits_per_element"] for r in results) / len(results)
    total_fp16 = sum(r["fp16_bytes"] for r in results)
    total_quant = sum(r["quant_bytes"] for r in results)
    print("-" * 78, flush=True)
    print(f"  {'Total / Average':<33} {total_fp16:>12,} {total_quant:>12,} {total_fp16/total_quant:>7.2f}x {avg_bits:>5.1f}", flush=True)
    print(f"\n  Memory saved across all workloads: "
          f"{(1 - total_quant/total_fp16)*100:.1f}%\n", flush=True)
    return results


# -----------------------------------------------------------------------
# Benchmark 3: Scaling with sequence length
# -----------------------------------------------------------------------

def benchmark_scaling():
    print("\n=== Benchmark 3: Accuracy and Memory Scaling with Sequence Length ===", flush=True)
    print(f"{'Seq Len':>8} {'Outliers':>10} {'FA-2 RMSE':>12} {'Policy RMSE':>13} {'Compression':>12} {'Bits':>6}", flush=True)
    print("-" * 68, flush=True)

    seq_lens = [512, 1024, 2048, 4096, 8192]
    results = []

    fa2_ok = fa2_available()

    for seq_len in seq_lens:
        for outliers in [False, True]:
            cfg = {
                "seq_len": seq_len, "head_dim": 64, "causal": False,
                "outliers": outliers, "batch": 1, "num_heads": 4,
            }
            Q, K, V = generate_workload(cfg, seed=999)

            with torch.no_grad():
                O_ref = reference_attention(Q, K, V, causal=False)

            if fa2_ok:
                O_fa2 = fa2_attention(Q, K, V, causal=False)
            else:
                O_fa2 = torch.nn.functional.scaled_dot_product_attention(
                    Q.cuda().half(), K.cuda().half(), V.cuda().half(), is_causal=False
                ).float().cpu()

            with torch.no_grad():
                O_ev, avg_bits = mixed_precision_attention(
                    Q, K, V, evolved_policy, causal=False
                )

            rms_ref = O_ref.float().pow(2).mean().sqrt().item() + 1e-10
            rmse_fa2 = (O_fa2.cpu() - O_ref.float()).pow(2).mean().sqrt().item() / rms_ref
            rmse_ev  = (O_ev  - O_ref.float()).pow(2).mean().sqrt().item() / rms_ref

            fp16_bytes = 2 * 1 * 4 * seq_len * 64 * 2
            cache = QuantizedKVCache(block_size=128)
            cache.update(K, V, layer_idx=0)
            quant_bytes = cache.memory_stats()["total_bytes"]
            ratio = fp16_bytes / quant_bytes

            out_lbl = "Yes" if outliers else "No"
            print(f"  {seq_len:>6}   {out_lbl:>8}   {rmse_fa2:>10.6f}   {rmse_ev:>11.6f}   "
                  f"{ratio:>10.2f}x   {avg_bits:>4.1f}", flush=True)

            results.append({
                "seq_len": seq_len, "outliers": outliers,
                "rmse_fa2": rmse_fa2, "rmse_evolved": rmse_ev,
                "compression_ratio": ratio, "avg_bits": avg_bits,
            })
    print(flush=True)
    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("Adaptive Precision Attention vs FA-2 Benchmark", flush=True)
    print("Hardware:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Reference: FP32 (single-precision), Baseline: PyTorch SDPA (FA-2)", flush=True)
    print("=" * 72, flush=True)

    acc_results  = benchmark_accuracy()
    mem_results  = benchmark_kv_memory()
    scale_results = benchmark_scaling()

    # Save results for paper
    out = {
        "accuracy": acc_results,
        "kv_memory": mem_results,
        "scaling": scale_results,
    }
    # Remove non-serialisable config objects
    for r in out["accuracy"] + out["kv_memory"]:
        r["config"] = {k: str(v) for k, v in r["config"].items()}

    with open(os.path.join(os.path.dirname(__file__), "benchmark_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("Results saved to analysis/benchmark_results.json")
    print("=" * 72)
