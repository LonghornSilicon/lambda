import torch
import math


def compute_block_stats(Q, K, V, block_size=128):
    """Extract per-block statistics for precision policy decisions.

    Computes statistics block-by-block without materializing the full N×N matrix.
    Averages across batch and head dimensions for each (block_row, block_col) position.

    Args:
        Q, K, V: (B, H, N, d) tensors
        block_size: tile size

    Returns:
        List of lists: stats_grid[i][j] is a dict of statistics for block (i, j)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    stats_grid = []

    for i in range(Tr):
        row_stats = []
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Q_block = Q[:, :, q_start:q_end, :]

        q_norm = Q_block.float().norm() / math.sqrt(Q_block.numel())

        for j in range(Tc):
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            k_norm = K_block.float().norm() / math.sqrt(K_block.numel())
            v_norm = V_block.float().norm() / math.sqrt(V_block.numel())

            S_block = torch.matmul(Q_block.float(), K_block.float().transpose(-2, -1)) * scale

            score_max = S_block.max().item()
            score_mean = S_block.mean().item()
            score_var = S_block.var().item()

            abs_max = S_block.abs().max().item()
            abs_mean = S_block.abs().mean().item()
            abs_std = S_block.abs().std().item()
            has_outlier = bool(abs_max > abs_mean + 10 * abs_std) if abs_std > 0 else False

            P_block = torch.softmax(S_block, dim=-1)
            log_P = torch.log(P_block + 1e-10)
            entropy_per_row = -(P_block * log_P).sum(dim=-1)
            entropy = entropy_per_row.mean().item()

            q_center = (q_start + q_end) / 2.0
            k_center = (k_start + k_end) / 2.0
            causal_dist = int(abs(q_center - k_center))

            row_stats.append({
                "q_norm": q_norm.item(),
                "k_norm": k_norm.item(),
                "v_norm": v_norm.item(),
                "score_max": score_max,
                "score_mean": score_mean,
                "score_var": score_var,
                "entropy": entropy,
                "causal_dist": causal_dist,
                "block_row": i,
                "block_col": j,
                "has_outlier": has_outlier,
                "seq_len": N,
                "head_dim": d,
                "num_heads": H,
            })

        stats_grid.append(row_stats)

    return stats_grid
