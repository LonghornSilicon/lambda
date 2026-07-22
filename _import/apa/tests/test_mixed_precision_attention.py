import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.reference import reference_attention
from common.quantization import quantize_dequantize, PRECISION_BITS
from common.block_stats import compute_block_stats
from common.workloads import generate_workload, ALL_WORKLOADS


def mixed_precision_attention(Q, K, V, policy_fn, causal=False, block_size=128):
    """Attention with per-block precision selection."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    stats_grid = compute_block_stats(Q, K, V, block_size=block_size)
    total_bits = 0
    total_elements = 0

    O = torch.zeros(B, H, N, d, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=torch.float32, device=Q.device)
    L = torch.zeros(B, H, N, 1, dtype=torch.float32, device=Q.device)

    for i in range(Tr):
        q_s = i * Br
        q_e = min(q_s + Br, N)
        Q_block = Q[:, :, q_s:q_e, :].float()

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_s = j * Bc
            k_e = min(k_s + Bc, N)
            K_block = K[:, :, k_s:k_e, :].float()
            V_block = V[:, :, k_s:k_e, :].float()

            prec = policy_fn(stats_grid[i][j])
            Q_q = quantize_dequantize(Q_block, prec)
            K_q = quantize_dequantize(K_block, prec)
            V_q = quantize_dequantize(V_block, prec)

            S = torch.matmul(Q_q, K_q.transpose(-2, -1)) * scale
            if causal:
                q_idx = torch.arange(q_s, q_e, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_s, k_e, device=Q.device).unsqueeze(0)
                S.masked_fill_(q_idx < k_idx, float("-inf"))

            m_old = M[:, :, q_s:q_e, :]
            m_new = torch.max(S, dim=-1, keepdim=True).values
            m_cur = torch.maximum(m_old, m_new)

            exp_s = torch.exp(S - m_cur)
            exp_old = torch.exp(m_old - m_cur)

            L[:, :, q_s:q_e, :] = exp_old * L[:, :, q_s:q_e, :] + exp_s.sum(dim=-1, keepdim=True)
            O[:, :, q_s:q_e, :] = exp_old * O[:, :, q_s:q_e, :] + torch.matmul(exp_s, V_q)
            M[:, :, q_s:q_e, :] = m_cur

            bits = PRECISION_BITS[prec]
            total_bits += bits * (q_e - q_s) * (k_e - k_s)
            total_elements += (q_e - q_s) * (k_e - k_s)

    O = O / L
    avg_bits = total_bits / max(total_elements, 1)
    return O, avg_bits


def test_all_fp16_matches_reference():
    """All-FP16 policy should closely match reference."""
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    O_ref = reference_attention(Q.double(), K.double(), V.double(), causal=cfg["causal"])
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, lambda s: "fp16", causal=cfg["causal"])
    assert avg_bits == 16.0
    rmse = (O_mp.double() - O_ref).pow(2).mean().sqrt().item()
    assert rmse < 1e-3, f"FP16 RMSE too high: {rmse}"


def test_all_int4_has_high_compression():
    """All-INT4 policy should achieve 4 bits average."""
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, lambda s: "int4", causal=cfg["causal"])
    assert avg_bits == 4.0
    assert not torch.isnan(O_mp).any()


def test_mixed_policy_between_extremes():
    """A mixed policy should have avg_bits between 4 and 16."""
    def mixed_policy(stats):
        if stats["score_max"] > 3.0:
            return "fp16"
        if stats["entropy"] > 4.0:
            return "int4"
        return "fp8_e4m3"

    cfg = ALL_WORKLOADS[0]  # 512, no outliers, non-causal
    Q, K, V = generate_workload(cfg, seed=42)
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, mixed_policy, causal=cfg["causal"])
    assert 4.0 <= avg_bits <= 16.0, f"avg_bits={avg_bits} not in valid range"
    assert not torch.isnan(O_mp).any()
