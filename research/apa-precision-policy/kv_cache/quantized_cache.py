"""Quantized KV cache with entropy-based mixed-precision storage.

Hardware mapping:
  - Quantizer block: takes FP16/FP32 input, produces INT8 (value + scale) or
    passes through FP16. Single multiplier + round + clamp pipeline.
  - Cache SRAM: stores packed blocks. Each block has a header (precision tag,
    scale factor, zero point) followed by the data payload.
  - Dequantizer block: reads header, applies inverse scale to produce FP16/FP32
    output. Single multiplier pipeline.

All three are simple, streaming, and have fixed latency per block.
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from kv_cache.precision_controller import PrecisionController


# ---------------------------------------------------------------------------
# Block storage: what lives in the cache SRAM
# ---------------------------------------------------------------------------

@dataclass
class CachedBlock:
    """A single cached block with precision metadata.

    In hardware, this is the SRAM entry format:
      [1-bit precision tag] [16-bit scale] [N*D payload bits]
    """
    data: torch.Tensor          # INT8 tensor or FP16 tensor
    scale: Optional[float]      # Scale factor for INT8 dequantization (None for FP16)
    precision: str              # "int8" or "fp16"
    seq_start: int              # Start token index in the sequence
    seq_end: int                # End token index (exclusive)


# ---------------------------------------------------------------------------
# Quantizer: FP32 -> INT8 with per-block symmetric quantization
# ---------------------------------------------------------------------------

def quantize_block_int8(block: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Symmetric INT8 quantization of a single block.

    Hardware: single-pass max-abs scan, then multiply-round-clamp pipeline.

    Args:
        block: Float tensor of shape (..., N, D).

    Returns:
        (quantized_int8, scale) where quantized_int8 is a torch.int8 tensor
        and scale is the float scale factor for dequantization.
    """
    max_abs = block.detach().float().abs().max().item()
    if max_abs == 0.0:
        return torch.zeros_like(block, dtype=torch.int8), 1.0

    scale = max_abs / 127.0
    quantized = torch.round(block.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def quantize_block_fp16(block: torch.Tensor) -> torch.Tensor:
    """Store block in FP16 (no quantization, just cast).

    Hardware: passthrough with precision conversion.

    Args:
        block: Float tensor.

    Returns:
        FP16 tensor.
    """
    return block.detach().to(torch.float16)


# ---------------------------------------------------------------------------
# Dequantizer: INT8 -> FP32, FP16 -> FP32
# ---------------------------------------------------------------------------

def dequantize_block(cached: CachedBlock, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize a cached block back to full precision.

    Hardware: read precision tag, select pipeline (multiply by scale, or
    FP16-to-FP32 upcast). Fixed latency.

    Args:
        cached: The CachedBlock to dequantize.
        dtype: Target output dtype (float32 or float16).

    Returns:
        Dequantized tensor in the requested dtype.
    """
    if cached.precision == "int8":
        return (cached.data.float() * cached.scale).to(dtype)
    else:
        # FP16 -> target dtype
        return cached.data.to(dtype)


# ---------------------------------------------------------------------------
# Layer cache: all blocks for one KV pair at one layer
# ---------------------------------------------------------------------------

@dataclass
class LayerCache:
    """Cache for K and V at a single transformer layer."""
    k_blocks: List[CachedBlock] = field(default_factory=list)
    v_blocks: List[CachedBlock] = field(default_factory=list)
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Top-level quantized KV cache
# ---------------------------------------------------------------------------

class QuantizedKVCache:
    """Mixed-precision KV cache with entropy-based precision selection.

    Hardware-level view:
      Input K, V tensors arrive at the quantizer.
      The precision controller inspects each block and sets a precision tag.
      The quantizer compresses each block (INT8 + scale, or FP16 passthrough).
      Blocks are written to cache SRAM with their headers.
      On read, the dequantizer reconstructs FP32 for downstream attention.

    Args:
        block_size: Tokens per block. Must be a power of 2 for hardware.
        entropy_threshold: Entropy threshold for INT8/FP16 decision.
        sigma_multiplier: Outlier detection sensitivity.
        output_dtype: Dtype for dequantized output (float32 or float16).
    """

    def __init__(self, block_size: int = 128, entropy_threshold: float = 2.0,
                 sigma_multiplier: float = 10.0,
                 output_dtype: torch.dtype = torch.float32):
        self.block_size = block_size
        self.entropy_threshold = entropy_threshold
        self.output_dtype = output_dtype
        self.controller = PrecisionController(
            block_size=block_size,
            entropy_threshold=entropy_threshold,
            sigma_multiplier=sigma_multiplier,
        )
        # layer_idx -> LayerCache
        self._layers: Dict[int, LayerCache] = {}

    def _get_layer(self, layer_idx: int) -> LayerCache:
        if layer_idx not in self._layers:
            self._layers[layer_idx] = LayerCache()
        return self._layers[layer_idx]

    def _quantize_and_store(self, tensor: torch.Tensor,
                            decisions: list) -> List[CachedBlock]:
        """Quantize tensor blocks according to precision decisions.

        Args:
            tensor: Shape (B, H, N, D).
            decisions: Per-block precision decisions from the controller.

        Returns:
            List of CachedBlock, one per sequence block.
        """
        blocks = []
        N = tensor.shape[-2]

        for decision in decisions:
            b_idx = decision["block_idx"]
            start = b_idx * self.block_size
            end = min(start + self.block_size, N)
            block_data = tensor[..., start:end, :]
            precision = decision["precision"]

            if precision == "int8":
                quantized, scale = quantize_block_int8(block_data)
                blocks.append(CachedBlock(
                    data=quantized,
                    scale=scale,
                    precision="int8",
                    seq_start=start,
                    seq_end=end,
                ))
            else:
                fp16_data = quantize_block_fp16(block_data)
                blocks.append(CachedBlock(
                    data=fp16_data,
                    scale=None,
                    precision="fp16",
                    seq_start=start,
                    seq_end=end,
                ))

        return blocks

    def update(self, K: torch.Tensor, V: torch.Tensor,
               layer_idx: int = 0) -> None:
        """Quantize and store new K, V tensors into the cache.

        Replaces any existing cache for this layer. For incremental append
        (autoregressive decoding), call append() instead.

        Args:
            K: Key tensor, shape (B, H, N, D).
            V: Value tensor, shape (B, H, N, D).
            layer_idx: Transformer layer index.
        """
        if K.dim() != 4 or V.dim() != 4:
            raise ValueError("K and V must be 4D tensors (B, H, N, D)")

        # Precision controller runs on K (keys determine attention pattern)
        k_decisions = self.controller.decide(K)
        v_decisions = self.controller.decide(V)

        layer = self._get_layer(layer_idx)
        layer.k_blocks = self._quantize_and_store(K, k_decisions)
        layer.v_blocks = self._quantize_and_store(V, v_decisions)
        layer.total_tokens = K.shape[-2]

    def append(self, K_new: torch.Tensor, V_new: torch.Tensor,
               layer_idx: int = 0) -> None:
        """Append new tokens to the existing cache for a layer.

        Used during autoregressive decoding. The new tokens are quantized
        and appended to the existing block list.

        Args:
            K_new: New key tensor, shape (B, H, N_new, D).
            V_new: New value tensor, shape (B, H, N_new, D).
            layer_idx: Transformer layer index.
        """
        if K_new.dim() != 4 or V_new.dim() != 4:
            raise ValueError("K and V must be 4D tensors (B, H, N, D)")

        layer = self._get_layer(layer_idx)
        offset = layer.total_tokens

        k_decisions = self.controller.decide(K_new)
        v_decisions = self.controller.decide(V_new)

        new_k_blocks = self._quantize_and_store(K_new, k_decisions)
        new_v_blocks = self._quantize_and_store(V_new, v_decisions)

        # Adjust sequence indices by offset
        for block in new_k_blocks:
            block.seq_start += offset
            block.seq_end += offset
        for block in new_v_blocks:
            block.seq_start += offset
            block.seq_end += offset

        layer.k_blocks.extend(new_k_blocks)
        layer.v_blocks.extend(new_v_blocks)
        layer.total_tokens += K_new.shape[-2]

    def get(self, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return the full K, V tensors for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            (K, V) tensors in output_dtype, shape (B, H, N_total, D).

        Raises:
            KeyError: If no cache exists for this layer.
        """
        if layer_idx not in self._layers:
            raise KeyError(f"No cache for layer {layer_idx}")

        layer = self._layers[layer_idx]

        if not layer.k_blocks:
            raise KeyError(f"Empty cache for layer {layer_idx}")

        # Dequantize all blocks and concatenate along sequence dim
        k_parts = [dequantize_block(b, self.output_dtype) for b in layer.k_blocks]
        v_parts = [dequantize_block(b, self.output_dtype) for b in layer.v_blocks]

        K = torch.cat(k_parts, dim=-2)
        V = torch.cat(v_parts, dim=-2)

        return K, V

    def memory_stats(self) -> Dict:
        """Report memory usage and compression statistics.

        Returns:
            Dict with:
                total_bytes: Actual bytes stored in the cache.
                fp16_equivalent_bytes: Bytes if everything were FP16.
                compression_ratio: fp16_equivalent / actual.
                bits_per_element: Average bits per stored element.
                num_int8_blocks: Count of INT8 blocks.
                num_fp16_blocks: Count of FP16 blocks.
                per_layer: Dict of per-layer stats.
        """
        total_bytes = 0
        fp16_equiv_bytes = 0
        num_int8 = 0
        num_fp16 = 0
        per_layer = {}

        for layer_idx, layer in self._layers.items():
            layer_bytes = 0
            layer_fp16_equiv = 0
            layer_int8 = 0
            layer_fp16 = 0

            for block in layer.k_blocks + layer.v_blocks:
                num_elements = block.data.numel()

                if block.precision == "int8":
                    # INT8: 1 byte per element + 4 bytes for scale factor
                    block_bytes = num_elements * 1 + 4
                    layer_int8 += 1
                else:
                    # FP16: 2 bytes per element
                    block_bytes = num_elements * 2
                    layer_fp16 += 1

                # FP16 equivalent: 2 bytes per element
                equiv_bytes = num_elements * 2

                layer_bytes += block_bytes
                layer_fp16_equiv += equiv_bytes

            total_bytes += layer_bytes
            fp16_equiv_bytes += layer_fp16_equiv
            num_int8 += layer_int8
            num_fp16 += layer_fp16

            per_layer[layer_idx] = {
                "bytes": layer_bytes,
                "fp16_equivalent_bytes": layer_fp16_equiv,
                "compression_ratio": layer_fp16_equiv / max(layer_bytes, 1),
                "int8_blocks": layer_int8,
                "fp16_blocks": layer_fp16,
            }

        total_elements = fp16_equiv_bytes // 2 if fp16_equiv_bytes > 0 else 0
        bits_per_element = (total_bytes * 8) / max(total_elements, 1)

        return {
            "total_bytes": total_bytes,
            "fp16_equivalent_bytes": fp16_equiv_bytes,
            "compression_ratio": fp16_equiv_bytes / max(total_bytes, 1),
            "bits_per_element": bits_per_element,
            "num_int8_blocks": num_int8,
            "num_fp16_blocks": num_fp16,
            "per_layer": per_layer,
        }

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cached data.

        Args:
            layer_idx: If provided, clear only this layer. Otherwise clear all.
        """
        if layer_idx is not None:
            self._layers.pop(layer_idx, None)
        else:
            self._layers.clear()

    def num_layers(self) -> int:
        """Return the number of layers with cached data."""
        return len(self._layers)

    def seq_len(self, layer_idx: int = 0) -> int:
        """Return the total cached sequence length for a layer."""
        if layer_idx not in self._layers:
            return 0
        return self._layers[layer_idx].total_tokens
