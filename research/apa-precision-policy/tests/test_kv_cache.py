"""Tests for the entropy-based quantized KV cache.

Covers:
  1. Precision controller correctness (threshold logic, entropy, outlier detection)
  2. INT8 quantizer / dequantizer round-trip accuracy
  3. Full cache update/get cycle
  4. Memory savings (compression ratio)
  5. Append (autoregressive) mode
  6. Multi-layer support
  7. Edge cases (zeros, single token, large outliers)
"""

import sys
import math
sys.path.insert(0, "/home/shadeform/flash-attention-5")

import torch

from kv_cache.precision_controller import (
    select_precision,
    compute_block_entropy,
    detect_outliers,
    PrecisionController,
)
from kv_cache.quantized_cache import (
    QuantizedKVCache,
    quantize_block_int8,
    quantize_block_fp16,
    dequantize_block,
    CachedBlock,
)


# -----------------------------------------------------------------------
# Precision controller: threshold comparator
# -----------------------------------------------------------------------

class TestSelectPrecision:
    """Test the core comparator logic."""

    def test_int8_when_no_outlier(self):
        # No outlier -> always INT8 regardless of entropy
        assert select_precision(entropy=0.5, has_outlier=False) == "int8"
        assert select_precision(entropy=5.0, has_outlier=False) == "int8"

    def test_int8_when_high_entropy_with_outlier(self):
        # Outlier but high entropy -> INT8 (outlier effect diluted)
        assert select_precision(entropy=3.0, has_outlier=True) == "int8"
        assert select_precision(entropy=4.3, has_outlier=True) == "int8"

    def test_fp16_when_outlier_and_low_entropy(self):
        # Outlier AND low entropy -> FP16 (dangerous to quantize)
        assert select_precision(entropy=1.0, has_outlier=True) == "fp16"
        assert select_precision(entropy=1.99, has_outlier=True) == "fp16"

    def test_boundary_at_threshold(self):
        # Exactly at threshold -> INT8 (strict less-than comparison)
        assert select_precision(entropy=2.0, has_outlier=True) == "int8"

    def test_custom_threshold(self):
        # Custom threshold
        assert select_precision(entropy=2.5, has_outlier=True,
                                entropy_threshold=3.0) == "fp16"
        assert select_precision(entropy=3.0, has_outlier=True,
                                entropy_threshold=3.0) == "int8"


# -----------------------------------------------------------------------
# Entropy computation
# -----------------------------------------------------------------------

class TestBlockEntropy:

    def test_uniform_block_has_high_entropy(self):
        torch.manual_seed(42)
        block = torch.randn(128, 64)
        entropy = compute_block_entropy(block)
        # Uniform-ish distribution should have high entropy
        assert entropy > 5.0

    def test_peaked_block_has_low_entropy(self):
        # One huge value, rest near zero
        block = torch.zeros(128, 64)
        block[0, 0] = 100.0
        entropy = compute_block_entropy(block)
        assert entropy < 1.0

    def test_zero_block_has_zero_entropy(self):
        block = torch.zeros(32, 64)
        entropy = compute_block_entropy(block)
        assert entropy == 0.0

    def test_entropy_is_nonnegative(self):
        torch.manual_seed(123)
        for _ in range(10):
            block = torch.randn(64, 32)
            entropy = compute_block_entropy(block)
            assert entropy >= 0.0


# -----------------------------------------------------------------------
# Outlier detection
# -----------------------------------------------------------------------

class TestDetectOutliers:

    def test_normal_data_no_outlier(self):
        torch.manual_seed(42)
        block = torch.randn(128, 64)
        # Standard normal data should very rarely trigger 10-sigma
        assert detect_outliers(block) is False

    def test_injected_outlier_detected(self):
        torch.manual_seed(42)
        block = torch.randn(128, 64)
        block[0, 0] = 500.0  # Massive outlier
        assert detect_outliers(block) is True

    def test_zero_block_no_outlier(self):
        block = torch.zeros(64, 32)
        assert detect_outliers(block) is False

    def test_constant_block_no_outlier(self):
        block = torch.ones(64, 32) * 5.0
        assert detect_outliers(block) is False


# -----------------------------------------------------------------------
# PrecisionController (batch decisions)
# -----------------------------------------------------------------------

class TestPrecisionController:

    def test_returns_one_decision_per_block(self):
        torch.manual_seed(42)
        controller = PrecisionController(block_size=128)
        tensor = torch.randn(2, 4, 256, 64)  # 2 blocks of 128
        decisions = controller.decide(tensor)
        assert len(decisions) == 2
        for d in decisions:
            assert "block_idx" in d
            assert "precision" in d
            assert "entropy" in d
            assert "has_outlier" in d

    def test_normal_data_all_int8(self):
        torch.manual_seed(42)
        controller = PrecisionController(block_size=128)
        tensor = torch.randn(1, 1, 512, 64)
        decisions = controller.decide(tensor)
        for d in decisions:
            assert d["precision"] == "int8"

    def test_outlier_with_peaked_gets_fp16(self):
        # Construct a block that has an outlier and low entropy (peaked)
        controller = PrecisionController(block_size=128, entropy_threshold=2.0)
        # Nearly zero block with one giant value -> low entropy + outlier
        tensor = torch.zeros(1, 1, 128, 64)
        tensor[0, 0, 0, 0] = 1000.0
        decisions = controller.decide(tensor)
        assert len(decisions) == 1
        assert decisions[0]["precision"] == "fp16"
        assert decisions[0]["has_outlier"] is True

    def test_partial_last_block(self):
        controller = PrecisionController(block_size=128)
        tensor = torch.randn(1, 1, 200, 64)  # 128 + 72 tokens
        decisions = controller.decide(tensor)
        assert len(decisions) == 2


# -----------------------------------------------------------------------
# Quantizer / Dequantizer
# -----------------------------------------------------------------------

class TestQuantizer:

    def test_int8_roundtrip_accuracy(self):
        torch.manual_seed(42)
        block = torch.randn(2, 4, 128, 64)
        quantized, scale = quantize_block_int8(block)
        assert quantized.dtype == torch.int8
        assert scale > 0

        # Reconstruct
        reconstructed = quantized.float() * scale
        # Mean absolute error should be small (within one quantization step)
        mae = (block - reconstructed).abs().mean().item()
        assert mae < 0.02, f"INT8 MAE too large: {mae}"
        # Max absolute error bounded by scale factor
        max_error = (block - reconstructed).abs().max().item()
        assert max_error < scale * 1.5, f"Max error {max_error} exceeds 1.5 * scale {scale}"

    def test_int8_zero_block(self):
        block = torch.zeros(1, 1, 64, 32)
        quantized, scale = quantize_block_int8(block)
        assert quantized.abs().max() == 0
        assert scale > 0  # Scale should be nonzero (set to 1.0 fallback)

    def test_fp16_passthrough(self):
        torch.manual_seed(42)
        block = torch.randn(2, 4, 128, 64)
        fp16_block = quantize_block_fp16(block)
        assert fp16_block.dtype == torch.float16
        # Should be very close (just precision loss from float32->float16)
        mae = (block - fp16_block.float()).abs().mean().item()
        assert mae < 1e-3, f"FP16 passthrough MAE too large: {mae}"

    def test_dequantize_int8(self):
        torch.manual_seed(42)
        block = torch.randn(1, 1, 64, 32)
        quantized, scale = quantize_block_int8(block)

        cached = CachedBlock(
            data=quantized, scale=scale, precision="int8",
            seq_start=0, seq_end=64
        )
        result = dequantize_block(cached, dtype=torch.float32)
        assert result.dtype == torch.float32
        assert result.shape == block.shape

    def test_dequantize_fp16(self):
        torch.manual_seed(42)
        block = torch.randn(1, 1, 64, 32)
        fp16_data = quantize_block_fp16(block)

        cached = CachedBlock(
            data=fp16_data, scale=None, precision="fp16",
            seq_start=0, seq_end=64
        )
        result = dequantize_block(cached, dtype=torch.float32)
        assert result.dtype == torch.float32
        assert torch.allclose(block, result, atol=1e-3)


# -----------------------------------------------------------------------
# Full cache: update + get cycle
# -----------------------------------------------------------------------

class TestQuantizedKVCache:

    def test_basic_update_and_get(self):
        torch.manual_seed(42)
        B, H, N, D = 2, 4, 256, 64
        K = torch.randn(B, H, N, D)
        V = torch.randn(B, H, N, D)

        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)
        cache.update(K, V, layer_idx=0)

        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == K.shape
        assert V_out.shape == V.shape
        assert K_out.dtype == torch.float32
        assert V_out.dtype == torch.float32

    def test_reconstruction_accuracy(self):
        """INT8 quantization should preserve values within a few percent."""
        torch.manual_seed(42)
        B, H, N, D = 1, 2, 256, 64
        K = torch.randn(B, H, N, D)
        V = torch.randn(B, H, N, D)

        cache = QuantizedKVCache(block_size=128)
        cache.update(K, V, layer_idx=0)
        K_out, V_out = cache.get(layer_idx=0)

        # Mean absolute error should be small
        k_mae = (K - K_out).abs().mean().item()
        v_mae = (V - V_out).abs().mean().item()
        assert k_mae < 0.02, f"K MAE too large: {k_mae}"
        assert v_mae < 0.02, f"V MAE too large: {v_mae}"

    def test_memory_stats_compression(self):
        """Normal data should be mostly INT8 -> ~2x compression vs FP16."""
        torch.manual_seed(42)
        B, H, N, D = 1, 4, 512, 64
        K = torch.randn(B, H, N, D)
        V = torch.randn(B, H, N, D)

        cache = QuantizedKVCache(block_size=128)
        cache.update(K, V, layer_idx=0)
        stats = cache.memory_stats()

        assert stats["compression_ratio"] > 1.5, (
            f"Expected significant compression, got {stats['compression_ratio']:.2f}x"
        )
        assert stats["bits_per_element"] < 12, (
            f"Expected < 12 bits/element, got {stats['bits_per_element']:.1f}"
        )
        assert stats["num_int8_blocks"] > 0

    def test_memory_stats_all_fields(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)
        K = torch.randn(1, 1, 128, 64)
        V = torch.randn(1, 1, 128, 64)
        cache.update(K, V, layer_idx=0)
        stats = cache.memory_stats()

        assert "total_bytes" in stats
        assert "fp16_equivalent_bytes" in stats
        assert "compression_ratio" in stats
        assert "bits_per_element" in stats
        assert "num_int8_blocks" in stats
        assert "num_fp16_blocks" in stats
        assert "per_layer" in stats
        assert 0 in stats["per_layer"]

    def test_outlier_blocks_use_fp16(self):
        """Blocks with outliers and peaked values should be stored as FP16."""
        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)

        # Create a tensor where one block has a massive outlier and near-zero rest
        K = torch.zeros(1, 1, 128, 64)
        V = torch.zeros(1, 1, 128, 64)
        K[0, 0, 0, 0] = 1000.0
        V[0, 0, 0, 0] = 1000.0

        cache.update(K, V, layer_idx=0)
        stats = cache.memory_stats()
        # Should have at least some FP16 blocks
        assert stats["num_fp16_blocks"] > 0

    def test_seq_len_tracking(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)
        K = torch.randn(1, 1, 300, 64)
        V = torch.randn(1, 1, 300, 64)
        cache.update(K, V, layer_idx=0)
        assert cache.seq_len(layer_idx=0) == 300

    def test_get_missing_layer_raises(self):
        cache = QuantizedKVCache()
        try:
            cache.get(layer_idx=99)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_clear_single_layer(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)
        K = torch.randn(1, 1, 128, 64)
        V = torch.randn(1, 1, 128, 64)
        cache.update(K, V, layer_idx=0)
        cache.update(K, V, layer_idx=1)
        assert cache.num_layers() == 2

        cache.clear(layer_idx=0)
        assert cache.num_layers() == 1

    def test_clear_all(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)
        K = torch.randn(1, 1, 128, 64)
        V = torch.randn(1, 1, 128, 64)
        cache.update(K, V, layer_idx=0)
        cache.update(K, V, layer_idx=1)
        cache.clear()
        assert cache.num_layers() == 0


# -----------------------------------------------------------------------
# Append mode (autoregressive decoding)
# -----------------------------------------------------------------------

class TestAppendMode:

    def test_append_extends_sequence(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        K1 = torch.randn(1, 2, 256, 64)
        V1 = torch.randn(1, 2, 256, 64)
        cache.update(K1, V1, layer_idx=0)
        assert cache.seq_len(0) == 256

        K2 = torch.randn(1, 2, 128, 64)
        V2 = torch.randn(1, 2, 128, 64)
        cache.append(K2, V2, layer_idx=0)
        assert cache.seq_len(0) == 384

        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == (1, 2, 384, 64)
        assert V_out.shape == (1, 2, 384, 64)

    def test_append_preserves_existing_data(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        K1 = torch.randn(1, 1, 128, 64)
        V1 = torch.randn(1, 1, 128, 64)
        cache.update(K1, V1, layer_idx=0)
        K_before, V_before = cache.get(layer_idx=0)

        K2 = torch.randn(1, 1, 64, 64)
        V2 = torch.randn(1, 1, 64, 64)
        cache.append(K2, V2, layer_idx=0)
        K_after, V_after = cache.get(layer_idx=0)

        # First 128 tokens should be identical
        assert torch.equal(K_before, K_after[..., :128, :])
        assert torch.equal(V_before, V_after[..., :128, :])

    def test_single_token_append(self):
        """Simulate autoregressive single-token decode steps."""
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        K_init = torch.randn(1, 1, 128, 64)
        V_init = torch.randn(1, 1, 128, 64)
        cache.update(K_init, V_init, layer_idx=0)

        # Append 5 single tokens
        for _ in range(5):
            k_tok = torch.randn(1, 1, 1, 64)
            v_tok = torch.randn(1, 1, 1, 64)
            cache.append(k_tok, v_tok, layer_idx=0)

        assert cache.seq_len(0) == 133
        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape[-2] == 133


# -----------------------------------------------------------------------
# Multi-layer support
# -----------------------------------------------------------------------

class TestMultiLayer:

    def test_independent_layers(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        for layer in range(4):
            K = torch.randn(1, 2, 256, 64) * (layer + 1)
            V = torch.randn(1, 2, 256, 64) * (layer + 1)
            cache.update(K, V, layer_idx=layer)

        assert cache.num_layers() == 4

        for layer in range(4):
            K_out, V_out = cache.get(layer_idx=layer)
            assert K_out.shape == (1, 2, 256, 64)

    def test_per_layer_stats(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        for layer in range(3):
            K = torch.randn(1, 1, 128, 64)
            V = torch.randn(1, 1, 128, 64)
            cache.update(K, V, layer_idx=layer)

        stats = cache.memory_stats()
        assert len(stats["per_layer"]) == 3


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:

    def test_very_small_block_size(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=16)
        K = torch.randn(1, 1, 64, 32)
        V = torch.randn(1, 1, 64, 32)
        cache.update(K, V, layer_idx=0)
        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == K.shape

    def test_sequence_not_divisible_by_block(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)
        K = torch.randn(1, 1, 200, 64)
        V = torch.randn(1, 1, 200, 64)
        cache.update(K, V, layer_idx=0)
        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == (1, 1, 200, 64)

    def test_fp16_output_dtype(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128, output_dtype=torch.float16)
        K = torch.randn(1, 1, 128, 64)
        V = torch.randn(1, 1, 128, 64)
        cache.update(K, V, layer_idx=0)
        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.dtype == torch.float16
        assert V_out.dtype == torch.float16

    def test_update_replaces_existing(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128)

        K1 = torch.randn(1, 1, 128, 64)
        V1 = torch.randn(1, 1, 128, 64)
        cache.update(K1, V1, layer_idx=0)

        K2 = torch.randn(1, 1, 256, 64)
        V2 = torch.randn(1, 1, 256, 64)
        cache.update(K2, V2, layer_idx=0)

        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == (1, 1, 256, 64)
        assert cache.seq_len(0) == 256

    def test_rejects_non_4d_input(self):
        cache = QuantizedKVCache()
        try:
            cache.update(torch.randn(128, 64), torch.randn(128, 64), layer_idx=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_large_batch_and_heads(self):
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=64)
        B, H, N, D = 4, 8, 128, 32
        K = torch.randn(B, H, N, D)
        V = torch.randn(B, H, N, D)
        cache.update(K, V, layer_idx=0)
        K_out, V_out = cache.get(layer_idx=0)
        assert K_out.shape == (B, H, N, D)

        # Check accuracy
        k_mae = (K - K_out).abs().mean().item()
        assert k_mae < 0.02


# -----------------------------------------------------------------------
# Integration: verify the full pipeline matches the evolved policy
# -----------------------------------------------------------------------

class TestPolicyConsistency:
    """Verify the cache matches the Phase 1 evolved policy:
    'INT8 everywhere, FP16 only for blocks with outliers AND entropy < 2.0'
    """

    def test_all_normal_data_is_int8(self):
        """Normal randn data should be 100% INT8."""
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)
        K = torch.randn(1, 4, 512, 64)
        V = torch.randn(1, 4, 512, 64)
        cache.update(K, V, layer_idx=0)

        stats = cache.memory_stats()
        assert stats["num_fp16_blocks"] == 0, "Normal data should be all INT8"
        assert stats["compression_ratio"] > 1.9, (
            f"All-INT8 should give ~2x compression, got {stats['compression_ratio']:.2f}"
        )

    def test_peaked_outlier_block_is_fp16(self):
        """A block with outlier + low entropy should be FP16."""
        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)
        # Near-zero block with a single large value: low entropy + outlier
        K = torch.zeros(1, 1, 128, 64)
        V = torch.zeros(1, 1, 128, 64)
        K[0, 0, 0, 0] = 500.0
        V[0, 0, 0, 0] = 500.0
        cache.update(K, V, layer_idx=0)

        stats = cache.memory_stats()
        assert stats["num_fp16_blocks"] > 0, "Peaked outlier block should be FP16"

    def test_mixed_blocks(self):
        """Mix of normal and outlier blocks: should see both precisions."""
        torch.manual_seed(42)
        cache = QuantizedKVCache(block_size=128, entropy_threshold=2.0)

        # Block 0: normal -> INT8
        # Block 1: peaked outlier -> FP16
        K = torch.randn(1, 1, 256, 64) * 0.01  # Small values
        V = torch.randn(1, 1, 256, 64) * 0.01

        # Make first block uniformly distributed (high entropy, no outlier)
        K[0, 0, :128, :] = torch.randn(128, 64)
        V[0, 0, :128, :] = torch.randn(128, 64)

        # Make second block peaked with outlier
        K[0, 0, 128:, :] = 0.0
        V[0, 0, 128:, :] = 0.0
        K[0, 0, 128, 0] = 1000.0
        V[0, 0, 128, 0] = 1000.0

        cache.update(K, V, layer_idx=0)
        stats = cache.memory_stats()

        # Should have both INT8 and FP16 blocks
        assert stats["num_int8_blocks"] > 0, "Normal block should be INT8"
        assert stats["num_fp16_blocks"] > 0, "Peaked outlier block should be FP16"
