"""
Evaluator for FlashAttention-5 precision policy evolution.

Scores a candidate policy on accuracy (relative RMSE vs FP32 reference),
compression (avg bits), and reliability (no NaN/Inf outputs).
"""

import importlib.util
import sys
import os
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.reference import reference_attention, mixed_precision_attention
from common.workloads import ALL_WORKLOADS, generate_workload
from openevolve.evaluation_result import EvaluationResult

EVAL_WORKLOADS = [
    ALL_WORKLOADS[0],   # 512, d64, no causal, no outliers
    ALL_WORKLOADS[1],   # 512, d128, causal, outliers
    ALL_WORKLOADS[2],   # 2048, d64, no causal, outliers
    ALL_WORKLOADS[3],   # 2048, d128, causal, no outliers
    ALL_WORKLOADS[6],   # 2048, d64, causal, outliers
    ALL_WORKLOADS[8],   # 4096, d64, no causal, no outliers
]


def evaluate(program_path):
    """Evaluate a precision policy program.

    Args:
        program_path: path to Python file containing get_policy() function

    Returns:
        EvaluationResult with metrics: accuracy_score, compression_score,
        reliability_score, combined_score
    """
    try:
        spec = importlib.util.spec_from_file_location("policy_module", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_policy"):
            return _error_result("Missing get_policy() function")

        policy_fn = module.get_policy()
    except Exception as e:
        return _error_result(f"Failed to load policy: {e}")

    accuracy_scores = []
    compression_scores = []
    successes = 0
    total = len(EVAL_WORKLOADS)
    details = []

    for idx, cfg in enumerate(EVAL_WORKLOADS):
        try:
            Q, K, V = generate_workload(cfg, seed=42 + idx)

            with torch.no_grad():
                O_ref = reference_attention(Q, K, V, causal=cfg["causal"])
                O_mp, avg_bits = mixed_precision_attention(
                    Q, K, V, policy_fn, causal=cfg["causal"]
                )

            if torch.isnan(O_mp).any() or torch.isinf(O_mp).any():
                details.append(f"Workload {idx}: NaN/Inf in output")
                continue

            rmse = (O_mp - O_ref).pow(2).mean().sqrt().item()
            rms_ref = O_ref.pow(2).mean().sqrt().item() + 1e-10
            rel_rmse = rmse / rms_ref
            acc = 1.0 / (1.0 + rel_rmse * 10)
            comp = (16.0 - avg_bits) / 14.0

            accuracy_scores.append(acc)
            compression_scores.append(comp)
            successes += 1
            details.append(
                f"Workload {idx} (N={cfg['seq_len']},d={cfg['head_dim']},"
                f"causal={cfg['causal']},outliers={cfg['outliers']}): "
                f"rel_RMSE={rel_rmse:.6f}, bits={avg_bits:.1f}, acc={acc:.4f}, comp={comp:.4f}"
            )

        except Exception as e:
            details.append(f"Workload {idx}: ERROR {e}")
            traceback.print_exc()
            continue

    if successes == 0:
        return _error_result("All workloads failed", details=details)

    accuracy_score = sum(accuracy_scores) / len(accuracy_scores)
    compression_score = sum(compression_scores) / len(compression_scores)
    reliability_score = successes / total

    combined_score = (
        0.50 * accuracy_score
        + 0.30 * compression_score
        + 0.20 * reliability_score
    )

    return EvaluationResult(
        metrics={
            "accuracy_score": round(accuracy_score, 6),
            "compression_score": round(compression_score, 6),
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
            "compression_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "error": 1.0,
        },
        artifacts=artifacts,
    )
