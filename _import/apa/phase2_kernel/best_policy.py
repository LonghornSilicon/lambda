"""Phase 1 best evolved precision policy.

Discovered through 80-iteration OpenEvolve search. Combined score: 0.7451
- accuracy_score: 0.9188 (relative RMSE vs FP32, k=10)
- compression_score: 0.2857 (avg 12 bits per element)
- reliability_score: 1.0 (no NaN/Inf on any workload)

Strategy: int8 for all blocks except outlier blocks with peaked attention (entropy < 2.0).
"""


def precision_policy(block_stats):
    has_outlier = block_stats.get('has_outlier', False)
    entropy = block_stats.get('entropy', 4.3)
    if has_outlier and entropy < 2.0:
        return "fp16"
    return "int8"
