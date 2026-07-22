import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.block_stats import compute_block_stats


def test_block_stats_keys():
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 256, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float32)
    K = torch.randn(B, H, N, d, dtype=torch.float32)
    V = torch.randn(B, H, N, d, dtype=torch.float32)

    stats_grid = compute_block_stats(Q, K, V, block_size=128)
    assert len(stats_grid) == 2
    assert len(stats_grid[0]) == 2

    s = stats_grid[0][0]
    required_keys = [
        "q_norm", "k_norm", "v_norm",
        "score_max", "score_mean", "score_var",
        "entropy", "causal_dist", "block_row", "block_col",
        "has_outlier", "seq_len", "head_dim", "num_heads",
    ]
    for key in required_keys:
        assert key in s, f"Missing key: {key}"


def test_block_stats_dimensions():
    torch.manual_seed(42)
    B, H, N, d = 2, 4, 384, 128
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    stats = compute_block_stats(Q, K, V, block_size=128)
    assert len(stats) == 3
    assert len(stats[0]) == 3


def test_outlier_detection():
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 128, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    Q[0, 0, 0, 0] = 500.0

    stats = compute_block_stats(Q, K, V, block_size=128)
    assert stats[0][0]["has_outlier"] is True


def test_entropy_is_positive():
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 256, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    stats = compute_block_stats(Q, K, V, block_size=128)
    for row in stats:
        for s in row:
            assert s["entropy"] >= 0.0
