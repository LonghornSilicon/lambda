"""
Evaluator for FlashAttention-5 kernel evolution.

Scores a candidate kernel on throughput (TFLOPS/s) and accuracy (relative RMSE
vs FP32 reference). Combined: 60% throughput + 40% accuracy.

Hardware: NVIDIA RTX A4000 (Ampere), ~150 TFLOPS FP16 tensor core peak.
"""

import importlib.util
import sys
import os
import time
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.reference import reference_attention
from common.workloads import ALL_WORKLOADS, generate_workload
from openevolve.evaluation_result import EvaluationResult

A4000_PEAK_TFLOPS = 150.0

EVAL_WORKLOADS = [
    ALL_WORKLOADS[0],   # 512, d64, no causal, no outliers
    ALL_WORKLOADS[1],   # 512, d128, causal, outliers
    ALL_WORKLOADS[3],   # 2048, d128, causal, no outliers
    ALL_WORKLOADS[6],   # 2048, d64, causal, outliers
    ALL_WORKLOADS[8],   # 4096, d64, no causal, no outliers
]

WARMUP_RUNS = 3
TIMED_RUNS = 5


def _attention_flops(B, H, N, d, causal):
    flops = 4 * B * H * N * N * d
    if causal:
        flops = flops // 2
    return flops


def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("kernel_module", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_kernel"):
            return _error_result("Missing get_kernel() function")

        kernel_fn = module.get_kernel()
    except Exception as e:
        return _error_result(f"Failed to load kernel: {e}\n{traceback.format_exc()}")

    accuracy_scores = []
    throughput_scores = []
    successes = 0
    total = len(EVAL_WORKLOADS)
    details = []

    for idx, cfg in enumerate(EVAL_WORKLOADS):
        try:
            Q, K, V = generate_workload(cfg, seed=42 + idx)
            if torch.cuda.is_available():
                Q, K, V = Q.cuda(), K.cuda(), V.cuda()

            B, H, N, d = Q.shape
            flops = _attention_flops(B, H, N, d, cfg["causal"])

            with torch.no_grad():
                O_ref = reference_attention(Q, K, V, causal=cfg["causal"])

                for _ in range(WARMUP_RUNS):
                    try:
                        _ = kernel_fn(Q, K, V, causal=cfg["causal"])
                    except Exception:
                        break
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(TIMED_RUNS):
                    O_kernel = kernel_fn(Q, K, V, causal=cfg["causal"])
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event) / TIMED_RUNS

            if torch.isnan(O_kernel).any() or torch.isinf(O_kernel).any():
                details.append(f"Workload {idx}: NaN/Inf in output")
                continue

            rmse = (O_kernel - O_ref).pow(2).mean().sqrt().item()
            rms_ref = O_ref.pow(2).mean().sqrt().item() + 1e-10
            rel_rmse = rmse / rms_ref
            acc = 1.0 / (1.0 + rel_rmse * 10)

            tflops = flops / (elapsed_ms * 1e-3 * 1e12)
            tp_score = min(tflops / A4000_PEAK_TFLOPS, 1.0)

            accuracy_scores.append(acc)
            throughput_scores.append(tp_score)
            successes += 1
            details.append(
                f"Workload {idx} (N={N},d={d},causal={cfg['causal']},"
                f"outliers={cfg['outliers']}): "
                f"rel_RMSE={rel_rmse:.6f}, acc={acc:.4f}, "
                f"time={elapsed_ms:.2f}ms, TFLOPS={tflops:.3f}, tp={tp_score:.4f}"
            )

        except torch.cuda.OutOfMemoryError:
            details.append(f"Workload {idx}: OOM")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            details.append(f"Workload {idx}: ERROR {e}")
            traceback.print_exc()
            continue

    if successes == 0:
        return _error_result("All workloads failed", details=details)

    accuracy_score = sum(accuracy_scores) / len(accuracy_scores)
    throughput_score = sum(throughput_scores) / len(throughput_scores)
    reliability_score = successes / total

    combined_score = (
        0.60 * throughput_score
        + 0.30 * accuracy_score
        + 0.10 * reliability_score
    )

    return EvaluationResult(
        metrics={
            "accuracy_score": round(accuracy_score, 6),
            "throughput_score": round(throughput_score, 6),
            "reliability_score": round(reliability_score, 6),
            "combined_score": round(combined_score, 6),
        },
        artifacts={
            "workload_details": "\n".join(details),
            "num_workloads_passed": str(successes),
        },
    )


def _error_result(msg, details=None):
    artifacts = {"error": msg}
    if details:
        artifacts["details"] = "\n".join(details)
    return EvaluationResult(
        metrics={
            "accuracy_score": 0.0,
            "throughput_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "error": 1.0,
        },
        artifacts=artifacts,
    )
