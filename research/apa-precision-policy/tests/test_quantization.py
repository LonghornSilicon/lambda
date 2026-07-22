import torch
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.quantization import quantize_dequantize, PRECISION_BITS


def test_fp16_is_lossless_for_fp16_input():
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp16")
    assert torch.allclose(x, x_rt, atol=1e-3)


def test_int8_roundtrip_preserves_sign():
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "int8")
    scale = x.abs().max() / 127
    large_mask = x.abs() > scale
    assert (x[large_mask].sign() == x_rt[large_mask].sign()).all()


def test_int4_has_at_most_15_unique_values():
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "int4")
    unique_ratios = x_rt.unique().numel()
    assert unique_ratios <= 15


def test_fp8_e4m3_clamps_outliers():
    x = torch.tensor([0.1, 1.0, 100.0, 1000.0, 10000.0], dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp8_e4m3")
    assert x_rt.max() <= 500.0


def test_fp4_extreme_compression():
    x = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp4")
    assert (x_rt[1:] >= x_rt[:-1]).all(), "FP4 should preserve ordering"


def test_precision_bits_map():
    assert PRECISION_BITS["fp16"] == 16
    assert PRECISION_BITS["fp8_e4m3"] == 8
    assert PRECISION_BITS["fp8_e5m2"] == 8
    assert PRECISION_BITS["int8"] == 8
    assert PRECISION_BITS["fp4"] == 4
    assert PRECISION_BITS["int4"] == 4


def test_quantize_dequantize_rejects_unknown():
    x = torch.randn(16, 16, dtype=torch.float32)
    try:
        quantize_dequantize(x, "bfloat3")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
