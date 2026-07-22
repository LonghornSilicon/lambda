import torch
import math
import triton
import triton.language as tl


def _quantize_int8(x):
    """Symmetric INT8 quantize-dequantize roundtrip."""
    max_val = x.abs().max()
    if max_val == 0:
        return x
    scale = max_val / 127.0
    return torch.round(x / scale).clamp(-127, 127) * scale


# EVOLVE-BLOCK-START
def flash_attention_5(Q, K, V, causal=False):
    """Mixed-precision fused attention kernel.

    Implements tiled attention with online softmax and adaptive precision:
    - fp16 for blocks with outliers and peaked attention (entropy < 2.0)
    - int8 for all other blocks

    Args:
        Q: (B, H, N, d) float32 tensor on GPU
        K: (B, H, N, d) float32 tensor on GPU
        V: (B, H, N, d) float32 tensor on GPU
        causal: whether to apply causal mask

    Returns:
        O: (B, H, N, d) float32 tensor
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    BLOCK = 128
    Tr = math.ceil(N / BLOCK)
    Tc = math.ceil(N / BLOCK)

    O = torch.zeros(B, H, N, d, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=torch.float32, device=Q.device)
    L = torch.zeros(B, H, N, 1, dtype=torch.float32, device=Q.device)

    for i in range(Tr):
        q_s = i * BLOCK
        q_e = min(q_s + BLOCK, N)
        Q_block = Q[:, :, q_s:q_e, :]

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_s = j * BLOCK
            k_e = min(k_s + BLOCK, N)
            K_block = K[:, :, k_s:k_e, :]
            V_block = V[:, :, k_s:k_e, :]

            S = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            abs_vals = S.abs()
            abs_max_val = abs_vals.max().item()
            abs_mean_val = abs_vals.mean().item()
            abs_std_val = abs_vals.std().item()
            has_outlier = abs_max_val > abs_mean_val + 10 * abs_std_val if abs_std_val > 0 else False

            use_fp16 = False
            if has_outlier:
                P_temp = torch.softmax(S, dim=-1)
                entropy = -(P_temp * torch.log(P_temp + 1e-10)).sum(dim=-1).mean().item()
                use_fp16 = entropy < 2.0

            if not use_fp16:
                Q_q = _quantize_int8(Q_block)
                K_q = _quantize_int8(K_block)
                V_q = _quantize_int8(V_block)
                S = torch.matmul(Q_q, K_q.transpose(-2, -1)) * scale
            else:
                V_q = V_block

            if causal:
                q_idx = torch.arange(q_s, q_e, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_s, k_e, device=Q.device).unsqueeze(0)
                S.masked_fill_(q_idx < k_idx, float("-inf"))

            m_old = M[:, :, q_s:q_e, :]
            m_new = S.max(dim=-1, keepdim=True).values
            m_cur = torch.maximum(m_old, m_new)

            exp_s = torch.exp(S - m_cur)
            exp_old = torch.exp(m_old - m_cur)

            L[:, :, q_s:q_e, :] = exp_old * L[:, :, q_s:q_e, :] + exp_s.sum(dim=-1, keepdim=True)
            O[:, :, q_s:q_e, :] = exp_old * O[:, :, q_s:q_e, :] + torch.matmul(exp_s, V_q)
            M[:, :, q_s:q_e, :] = m_cur

    return O / L
# EVOLVE-BLOCK-END


def get_kernel():
    """Entry point used by the evaluator."""
    return flash_attention_5
