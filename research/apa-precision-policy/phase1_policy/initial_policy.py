# EVOLVE-BLOCK-START
def precision_policy(block_stats):
    """Map block-level statistics to a precision level.

    Args:
        block_stats: dict with keys:
            q_norm       (float): L2 norm of query block
            k_norm       (float): L2 norm of key block
            v_norm       (float): L2 norm of value block
            score_max    (float): max pre-softmax attention score in block
            score_mean   (float): mean pre-softmax attention score
            score_var    (float): variance of pre-softmax scores
            entropy      (float): softmax entropy (higher = more uniform)
            causal_dist  (int):   |query_pos - key_pos| average for block
            block_row    (int):   query block index
            block_col    (int):   key block index
            has_outlier  (bool):  True if max value > 6 * mean abs value
            seq_len      (int):   total sequence length
            head_dim     (int):   head dimension
            num_heads    (int):   number of attention heads

    Returns:
        One of: "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "fp4", "int4"
    """
    return "fp16"
# EVOLVE-BLOCK-END


def get_policy():
    """Entry point used by the evaluator to retrieve the current policy function."""
    return precision_policy
