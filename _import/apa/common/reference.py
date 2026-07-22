import torch
import math


def reference_attention(Q, K, V, causal=False, block_size=128):
    """Tiled FP64 reference attention using online softmax.

    Computes exact attention without materializing the full N×N score matrix.
    Uses the same online softmax algorithm as FlashAttention.

    Args:
        Q: (B, H, N, d) float64
        K: (B, H, N, d) float64
        V: (B, H, N, d) float64
        causal: apply causal mask
        block_size: tile size for tiled computation

    Returns:
        O: (B, H, N, d) float64
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = block_size
    Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, 1, dtype=Q.dtype, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=Q.dtype, device=Q.device)

    for i in range(Tr):
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Q_block = Q[:, :, q_start:q_end, :]

        o_block = torch.zeros_like(Q_block)
        m_block = torch.full(
            (B, H, q_end - q_start, 1), float("-inf"),
            dtype=Q.dtype, device=Q.device,
        )
        l_block = torch.zeros(
            B, H, q_end - q_start, 1, dtype=Q.dtype, device=Q.device,
        )

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            if causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = q_idx < k_idx
                S_block.masked_fill_(causal_mask, float("-inf"))

            m_block_old = m_block.clone()
            m_new = torch.max(S_block, dim=-1, keepdim=True).values
            m_block = torch.maximum(m_block, m_new)

            exp_scores = torch.exp(S_block - m_block)
            exp_old = torch.exp(m_block_old - m_block)

            l_block = exp_old * l_block + exp_scores.sum(dim=-1, keepdim=True)
            o_block = exp_old * o_block + torch.matmul(exp_scores, V_block)

        o_block = o_block / l_block
        O[:, :, q_start:q_end, :] = o_block
        L[:, :, q_start:q_end, :] = l_block
        M[:, :, q_start:q_end, :] = m_block

    return O


def mixed_precision_attention(Q, K, V, policy_fn, causal=False, block_size=128):
    """Attention with per-block precision selection.

    Args:
        Q, K, V: (B, H, N, d) float32 tensors
        policy_fn: callable(block_stats_dict) -> precision string
        causal: apply causal mask
        block_size: tile size

    Returns:
        (O, avg_bits): output tensor and average bits per element used
    """
    from common.quantization import quantize_dequantize, PRECISION_BITS
    from common.block_stats import compute_block_stats

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
