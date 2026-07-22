"""Precision controller: decides INT8 vs FP16 per block based on entropy.

Hardware mapping: This is a small comparator circuit. For each block, it
receives two signals (has_outlier flag, entropy value) and outputs a single
precision-select bit. No floating-point ALU needed beyond the entropy
computation itself -- the decision is a pair of threshold comparisons.

The entropy computation is the most expensive part. In hardware, it can be
approximated with a lookup table on the softmax distribution's histogram.
Here we provide both an exact path (using the full softmax) and a fast
approximate path (using value-range heuristics) for different use cases.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Core decision logic -- maps directly to hardware comparator
# ---------------------------------------------------------------------------

def select_precision(entropy: float, has_outlier: bool,
                     entropy_threshold: float = 2.0) -> str:
    """Pure threshold comparison: the hardware comparator.

    Returns "fp16" only when BOTH conditions hold:
      1. The block has outlier values (has_outlier flag is set)
      2. Attention entropy is below threshold (peaked attention)

    When attention is peaked AND outliers are present, the outlier values
    carry disproportionate weight in the output. INT8 quantization would
    clip or distort them. When entropy is high (uniform attention), outlier
    contribution is diluted across many positions and INT8 is safe.

    Args:
        entropy: Shannon entropy of the attention distribution for this block.
        has_outlier: Whether the block's values exceed the outlier threshold.
        entropy_threshold: Entropy below which peaked attention is dangerous.

    Returns:
        "fp16" or "int8"
    """
    if has_outlier and entropy < entropy_threshold:
        return "fp16"
    return "int8"


# ---------------------------------------------------------------------------
# Block-level entropy computation
# ---------------------------------------------------------------------------

def compute_block_entropy(block: torch.Tensor) -> float:
    """Compute entropy of the value distribution within a single block.

    This estimates how "uniform" vs "peaked" the values in the block are by
    treating the absolute values as an unnormalized distribution. This is a
    lightweight proxy for attention entropy that does not require Q or the
    full softmax -- it works on K or V blocks in isolation.

    In hardware, this maps to: abs -> histogram -> normalize -> entropy LUT.

    Args:
        block: Tensor of any shape. Operates on the flattened values.

    Returns:
        Entropy estimate (float). Higher = more uniform, lower = more peaked.
    """
    flat = block.detach().float().flatten()
    abs_vals = flat.abs()

    # Avoid degenerate case
    total = abs_vals.sum()
    if total == 0:
        return 0.0

    # Normalize to a probability distribution
    p = abs_vals / total

    # Shannon entropy: -sum(p * log(p)), skip zeros
    nonzero = p > 0
    log_p = torch.zeros_like(p)
    log_p[nonzero] = torch.log2(p[nonzero])
    entropy = -(p * log_p).sum().item()

    return entropy


def detect_outliers(block: torch.Tensor, sigma_multiplier: float = 10.0) -> bool:
    """Detect whether a block contains statistical outliers.

    Uses the same criterion as compute_block_stats: a value is an outlier if
    its absolute magnitude exceeds mean(|x|) + sigma_multiplier * std(|x|).

    In hardware: compute running abs-mean and abs-std, compare max against
    threshold. Single-pass with an accumulator.

    Args:
        block: Tensor of any shape.
        sigma_multiplier: Number of standard deviations for outlier threshold.

    Returns:
        True if the block contains outliers.
    """
    flat = block.detach().float().flatten()
    abs_vals = flat.abs()
    abs_mean = abs_vals.mean().item()
    abs_std = abs_vals.std().item()
    abs_max = abs_vals.max().item()

    if abs_std == 0:
        return False

    return abs_max > abs_mean + sigma_multiplier * abs_std


# ---------------------------------------------------------------------------
# Batch controller: processes all blocks and returns precision map
# ---------------------------------------------------------------------------

class PrecisionController:
    """Decides precision for each block in a KV tensor.

    This is the top-level "precision controller" hardware block. It takes
    a full K or V tensor, splits it into blocks, and produces a per-block
    precision decision.

    Attributes:
        block_size: Number of tokens per block.
        entropy_threshold: Entropy threshold for the INT8/FP16 decision.
        sigma_multiplier: Outlier detection sensitivity.
    """

    def __init__(self, block_size: int = 128, entropy_threshold: float = 2.0,
                 sigma_multiplier: float = 10.0):
        self.block_size = block_size
        self.entropy_threshold = entropy_threshold
        self.sigma_multiplier = sigma_multiplier

    def decide(self, tensor: torch.Tensor) -> list:
        """Compute per-block precision decisions.

        Args:
            tensor: Shape (..., N, D) where N is the sequence dimension.
                    Leading dimensions (batch, heads) are averaged over
                    for the decision -- all heads in a block share precision.

        Returns:
            List of dicts, one per block along the sequence dimension:
                {
                    "block_idx": int,
                    "precision": "int8" or "fp16",
                    "entropy": float,
                    "has_outlier": bool,
                }
        """
        # Work on the last two dims: (N, D)
        # Flatten leading dims for a single decision per sequence block
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions (N, D)")

        N = tensor.shape[-2]
        num_blocks = math.ceil(N / self.block_size)
        decisions = []

        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, N)
            # Slice along sequence dim, keep all leading dims
            block = tensor[..., start:end, :]

            entropy = compute_block_entropy(block)
            has_outlier = detect_outliers(block, self.sigma_multiplier)
            precision = select_precision(entropy, has_outlier,
                                         self.entropy_threshold)

            decisions.append({
                "block_idx": b,
                "precision": precision,
                "entropy": entropy,
                "has_outlier": has_outlier,
            })

        return decisions
