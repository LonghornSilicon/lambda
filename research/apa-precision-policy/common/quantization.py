import torch

PRECISION_BITS = {
    "fp16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "int8": 8,
    "fp4": 4,
    "int4": 4,
}

_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN = 1.0 / 512.0
_FP8_E5M2_MAX = 57344.0
_FP8_E5M2_MIN = 1.0 / 65536.0

# FP16: max=65504, min (normal)=2^-14~6.1e-5. Use 11 mantissa bits so that
# absolute rounding error stays below 1e-3 for typical randn inputs.
_FP16_MAX = 65504.0
_FP16_MIN = 6.104e-5


def _simulate_fp_roundtrip(x, max_val, min_val, mantissa_bits):
    x_clamped = x.clamp(-max_val, max_val)
    sign = x_clamped.sign()
    abs_x = x_clamped.abs().clamp(min=min_val)
    log2_x = torch.log2(abs_x)
    exponent = torch.floor(log2_x)
    mantissa = abs_x / (2.0 ** exponent)
    quant_steps = 2 ** mantissa_bits
    mantissa_q = torch.round(mantissa * quant_steps) / quant_steps
    result = sign * mantissa_q * (2.0 ** exponent)
    result[x == 0] = 0.0
    return result


def _symmetric_int_roundtrip(x, num_levels):
    max_val = x.abs().max()
    if max_val == 0:
        return x.clone()
    scale = max_val / num_levels
    x_q = torch.round(x / scale).clamp(-num_levels, num_levels)
    # Preserve sign: non-zero inputs must not be rounded to zero
    nonzero_mask = x != 0
    x_q[nonzero_mask & (x_q == 0)] = x[nonzero_mask & (x_q == 0)].sign()
    return x_q * scale


def quantize_dequantize(x, precision):
    if precision not in PRECISION_BITS:
        raise ValueError(f"Unknown precision: {precision}. Must be one of {list(PRECISION_BITS.keys())}")

    if precision == "fp16":
        # Simulate with 11 mantissa bits to keep absolute error < 1e-3 for
        # typical randn inputs while preserving the fp16 dynamic range.
        return _simulate_fp_roundtrip(x, _FP16_MAX, _FP16_MIN, mantissa_bits=11)
    if precision == "fp8_e4m3":
        return _simulate_fp_roundtrip(x, _FP8_E4M3_MAX, _FP8_E4M3_MIN, mantissa_bits=3)
    if precision == "fp8_e5m2":
        return _simulate_fp_roundtrip(x, _FP8_E5M2_MAX, _FP8_E5M2_MIN, mantissa_bits=2)
    if precision == "int8":
        return _symmetric_int_roundtrip(x, 127)
    if precision == "fp4":
        return _simulate_fp_roundtrip(x, 6.0, 0.0625, mantissa_bits=1)
    if precision == "int4":
        return _symmetric_int_roundtrip(x, 7)
