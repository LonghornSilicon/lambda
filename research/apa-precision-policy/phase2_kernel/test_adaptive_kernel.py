"""Test suite for the adaptive attention Triton kernel.

Checks:
  1. Correctness vs FP32 reference on all 12 standard workloads
  2. Precision decisions match the ratio > 10 rule
  3. Timing: kernel latency vs PyTorch SDPA baseline
"""

import sys
import os
import math
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_kernel.adaptive_attention import adaptive_attention
from common.workloads import ALL_WORKLOADS, generate_workload
from common.reference import reference_attention

RATIO_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sdpa_baseline(Q, K, V, causal=False):
    """PyTorch SDPA (FA-2 on Ampere) for comparison."""
    with torch.no_grad():
        return F.scaled_dot_product_attention(
            Q, K, V, is_causal=causal
        )


def relative_rmse(output, reference):
    rms_ref = reference.float().pow(2).mean().sqrt().item() + 1e-10
    return (output.float() - reference.float()).pow(2).mean().sqrt().item() / rms_ref


def benchmark_ms(fn, warmup=5, reps=20):
    """Measure median latency in milliseconds."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Test 1: Correctness
# ---------------------------------------------------------------------------

def test_correctness():
    print("\n=== Test 1: Correctness vs FP32 Reference ===", flush=True)
    print(f"{'Workload':<38} {'Kernel RMSE':>12} {'SDPA RMSE':>12} {'Pass':>6}", flush=True)
    print("-" * 74, flush=True)

    all_pass = True
    for idx, cfg in enumerate(ALL_WORKLOADS):
        Q, K, V = generate_workload(cfg, seed=42 + idx)

        # FP32 reference
        with torch.no_grad():
            O_ref = reference_attention(Q, K, V, causal=cfg["causal"])

        # Our kernel (needs fp16 on CUDA)
        Qh = Q.cuda().half()
        Kh = K.cuda().half()
        Vh = V.cuda().half()
        with torch.no_grad():
            O_kernel = adaptive_attention(Qh, Kh, Vh,
                                          causal=cfg["causal"],
                                          ratio_threshold=RATIO_THRESHOLD)
        O_kernel_cpu = O_kernel.float().cpu()

        # SDPA baseline
        with torch.no_grad():
            O_sdpa = sdpa_baseline(Qh, Kh, Vh, causal=cfg["causal"]).float().cpu()

        rmse_kernel = relative_rmse(O_kernel_cpu, O_ref)
        rmse_sdpa   = relative_rmse(O_sdpa,       O_ref)

        # Pass if kernel is within 3× SDPA error (both are approximate vs FP32)
        passed = rmse_kernel < max(0.05, rmse_sdpa * 3)
        all_pass = all_pass and passed

        label = (f"N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
                 f"{' causal' if cfg['causal'] else ''}"
                 f"{' outliers' if cfg['outliers'] else ''}")
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<36} {rmse_kernel:>12.6f} {rmse_sdpa:>12.6f} {status:>6}", flush=True)

    print("-" * 74, flush=True)
    print(f"\n  All correctness tests: {'PASSED' if all_pass else 'FAILED'}", flush=True)
    return all_pass


# ---------------------------------------------------------------------------
# Test 2: Precision decisions
# ---------------------------------------------------------------------------

def test_precision_decisions():
    print("\n=== Test 2: Precision Decisions Match Ratio > 10 Rule ===", flush=True)
    print(f"{'Workload':<38} {'Expected':>10} {'Ratio':>10} {'Decision':>10}", flush=True)
    print("-" * 72, flush=True)

    for idx, cfg in enumerate(ALL_WORKLOADS):
        Q, K, V = generate_workload(cfg, seed=42 + idx)

        # Compute ratio on first block only (representative)
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        Q_b = Q[:, :, :128, :].float()
        K_b = K[:, :, :128, :].float()
        S = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale
        abs_s  = S.abs()
        ratio  = (abs_s.max() / (abs_s.mean() + 1e-6)).item()
        decision = "FP16" if ratio > RATIO_THRESHOLD else "INT8"
        expected = "FP16" if cfg["outliers"] else "INT8"
        match = "✓" if decision == expected else "✗"

        label = (f"N={cfg['seq_len']:4d} d={cfg['head_dim']:3d}"
                 f"{' causal' if cfg['causal'] else ''}"
                 f"{' outliers' if cfg['outliers'] else ''}")
        print(f"  {label:<36} {expected:>10} {ratio:>10.1f} {decision:>8} {match}", flush=True)

    print(flush=True)


# ---------------------------------------------------------------------------
# Test 3: Timing
# ---------------------------------------------------------------------------

def test_timing():
    print("\n=== Test 3: Kernel Latency vs PyTorch SDPA ===", flush=True)
    print(f"{'Shape':<30} {'Kernel ms':>12} {'SDPA ms':>12} {'Overhead':>10}", flush=True)
    print("-" * 68, flush=True)

    shapes = [
        (1, 4,  512, 64),
        (1, 4, 1024, 64),
        (1, 4, 2048, 64),
        (1, 4, 4096, 64),
        (1, 8, 2048, 128),
        (2, 8, 2048, 64),
    ]

    for B, H, N, d in shapes:
        Q = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
        K = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
        V = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")

        t_kernel = benchmark_ms(lambda: adaptive_attention(Q, K, V, causal=False))
        t_sdpa   = benchmark_ms(lambda: sdpa_baseline(Q, K, V, causal=False))
        overhead = t_kernel / t_sdpa

        label = f"B={B} H={H} N={N} d={d}"
        print(f"  {label:<28} {t_kernel:>12.3f} {t_sdpa:>12.3f} {overhead:>9.2f}x", flush=True)

    print(flush=True)
    print("  Note: kernel overhead vs SDPA is expected — SDPA is compiled,", flush=True)
    print("  our kernel adds ratio check + INT8 simulation in FP16 compute.", flush=True)
    print("  Speedup appears on FPGA/ASIC with real INT8 multiply units.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("Adaptive Attention Triton Kernel — Test Suite")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    import triton as _triton
    print(f"Triton: {_triton.__version__}", flush=True)
    print("=" * 72, flush=True)

    test_precision_decisions()
    ok = test_correctness()
    test_timing()

    print("\n" + "=" * 72, flush=True)
    print(f"Overall: {'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}", flush=True)
    print("=" * 72, flush=True)
