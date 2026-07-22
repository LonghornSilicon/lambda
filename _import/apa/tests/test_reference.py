import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.reference import reference_attention


def test_reference_matches_naive():
    """Reference tiled attention must match naive O(N^2) attention in FP64."""
    torch.manual_seed(42)
    B, H, N, d = 1, 2, 256, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_naive = torch.matmul(P, V)

    O_ref = reference_attention(Q, K, V, causal=False)
    assert torch.allclose(O_naive, O_ref, atol=1e-10), f"Max diff: {(O_naive - O_ref).abs().max()}"


def test_reference_causal():
    """Causal reference attention must zero out future positions."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 128, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O_naive = torch.matmul(P, V)

    O_ref = reference_attention(Q, K, V, causal=True)
    assert torch.allclose(O_naive, O_ref, atol=1e-10), f"Max diff: {(O_naive - O_ref).abs().max()}"


def test_reference_block_size_independent():
    """Result must be identical regardless of internal block size."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 512, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    O_64 = reference_attention(Q, K, V, causal=False, block_size=64)
    O_128 = reference_attention(Q, K, V, causal=False, block_size=128)
    assert torch.allclose(O_64, O_128, atol=1e-10)
