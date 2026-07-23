"""Bit-accurate golden for the VecU RMSNorm slice.

Models the ACTUAL hardware algorithm the RTL implements (rtl/rmsnorm.sv), NOT
`x/np.sqrt(mean(x**2)+eps)*g`:

    y_i = x_i * rsqrt(mean(x^2) + eps) * g_i

  - sum-of-squares over the D-element vector in an fp32 accumulator (each x_i
    widened to fp32, squared with fp32_mul, summed with fp32_add) — like the MatE
    accumulator, so the error is the rsqrt LUT, not fp16 sum drift;
  - mean = ss * (1/D)  (1/D a folded fp32 constant), v = mean + eps;
  - rsqrt(v) via a 64-entry LUT with linear interpolation over the reduced
    mantissa f in [1,2): decompose v = 2^E * f, look up rsqrt(f), then apply the
    exponent as 2^(-floor(E/2)) and, when E is odd, an extra 1/sqrt(2) factor
    (rsqrt(v) = rsqrt(f) * 2^(-E/2)). LUT entry
        R[j] = round_fp16(1/sqrt(1 + j/64)),  j = 0..63.
    The fp16 rounding of the rsqrt table IS the LUT approximation modeled here.
  - y_i = round_fp16( x_i * scale * g_i ), products in fp32, one fp16 rounding.

Two comparison bars (see __main__ / the RTL TB):
  (a) the RTL is BIT-EXACT to this LUT-golden;
  (b) this LUT-golden is within a measured tolerance of exact fp64 RMSNorm — the
      rsqrt LUT's actual error (reported below).

Pure Python (reuses the fp16/fp32 bit helpers of vecu_softmax_ref) so the RTL
testbench generator needs no numpy.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vecu_softmax_ref import (  # noqa: E402
    fp16_to_fp32, fp32_to_fp16, fp32_add, fp32_mul,
    f16_from_real, real_from_f16, f32_from_real,
)

# ---------------------------------------------------------------------------
# RMSNorm parameters + the rsqrt LUT
# ---------------------------------------------------------------------------
D        = 16                 # vector length
EPS      = 1.0 / 65536.0      # 2^-16, exactly representable in fp32/fp16
INV_D32  = f32_from_real(1.0 / D)
EPS32    = f32_from_real(EPS)
INV_SQRT2_32 = f32_from_real(1.0 / math.sqrt(2.0))

NLUT = 64
# R[j] = 1/sqrt(1 + j/64) for f in [1,2); R_TOP anchors the last interval (f->2).
RSQRT_LUT = [f16_from_real(1.0 / math.sqrt(1.0 + j / NLUT)) for j in range(NLUT)]
RSQRT_TOP = f16_from_real(1.0 / math.sqrt(2.0))


def _frac17_to_fp32(frac17: int) -> int:
    """Exact fp32 of frac17 / 2^17, frac17 in [0, 2^17-1]."""
    if frac17 == 0:
        return 0
    p = frac17.bit_length() - 1                 # MSB position (0..16)
    mant = (frac17 << (23 - p)) & 0x7FFFFF
    exp = (p - 17) + 127
    return ((exp & 0xFF) << 23) | mant


def _pow2_fp32(n: int) -> int:
    """2^n as an fp32 pattern (n a modest signed integer)."""
    e = 127 + n
    if e >= 255:
        return 0x7F800000                        # +inf (clamped; not hit by test data)
    if e <= 0:
        return 0                                  # underflow -> +0
    return (e & 0xFF) << 23


def rsqrt_fp32(v32: int) -> int:
    """rsqrt(v) for v a positive fp32 (normal) -> fp32."""
    e = (v32 >> 23) & 0xFF
    m = v32 & 0x7FFFFF
    if e == 0:                                    # subnormal/zero -> clamp to rsqrt(eps)
        v32 = EPS32
        e = (v32 >> 23) & 0xFF
        m = v32 & 0x7FFFFF
    E = e - 127                                   # unbiased exponent (v = 2^E * 1.m)
    idx = m >> 17                                 # top 6 mantissa bits -> LUT entry
    frac17 = m & 0x1FFFF                          # low 17 bits -> interp fraction
    lo = fp16_to_fp32(RSQRT_LUT[idx])
    hi = fp16_to_fp32(RSQRT_LUT[idx + 1] if idx < NLUT - 1 else RSQRT_TOP)
    frac32 = _frac17_to_fp32(frac17)
    r = fp32_add(lo, fp32_mul(frac32, fp32_add(hi, lo ^ 0x80000000)))  # lo + frac*(hi-lo)
    # apply exponent: rsqrt(v) = rsqrt(f) * 2^(-floor(E/2)) [* 1/sqrt2 if E odd]
    k = E >> 1                                     # arithmetic floor(E/2)
    r = fp32_mul(r, _pow2_fp32(-k))
    if E & 1:
        r = fp32_mul(r, INV_SQRT2_32)
    return r


def rmsnorm(x_f16, g_f16):
    """x_f16, g_f16: lists of D fp16 codes -> list of D fp16 output codes."""
    ss = 0                                         # fp32 +0.0
    for x in x_f16:
        x32 = fp16_to_fp32(x)
        ss = fp32_add(ss, fp32_mul(x32, x32))
    mean = fp32_mul(ss, INV_D32)
    v = fp32_add(mean, EPS32)
    scale = rsqrt_fp32(v)
    out = []
    for x, g in zip(x_f16, g_f16):
        t = fp32_mul(fp16_to_fp32(x), scale)
        t = fp32_mul(t, fp16_to_fp32(g))
        out.append(fp32_to_fp16(t))
    return out


# ---------------------------------------------------------------------------
# SV LUT emitter (embedded rsqrt case ROM in rtl/rmsnorm.sv is generated here)
# ---------------------------------------------------------------------------
def emit_sv_lut() -> str:
    lines = ["    // rsqrt_lut[j] = round_fp16(1/sqrt(1 + j/64)), j = 0..63"]
    for j in range(NLUT):
        lines.append(f"    rsqrt_lut[{j}] = 16'h{RSQRT_LUT[j]:04X};")
    lines.append(f"    // RSQRT_TOP (f -> 2) = 16'h{RSQRT_TOP:04X}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test: bar (b) — LUT-golden vs exact fp64 RMSNorm; report the LUT error
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    if len(sys.argv) > 1 and sys.argv[1] == "--emit-sv":
        print(emit_sv_lut())
        sys.exit(0)

    def exact_rmsnorm(x_f16, g_f16):
        xs = [real_from_f16(x) for x in x_f16]
        gs = [real_from_f16(g) for g in g_f16]
        mean = sum(v * v for v in xs) / len(xs)
        scale = 1.0 / math.sqrt(mean + EPS)
        return [v * scale * g for v, g in zip(xs, gs)]

    rng = random.Random(20260723)
    worst_rel = 0.0
    worst_case = None
    for _ in range(20000):
        rng_lo, rng_hi = rng.choice([(-1, 1), (-4, 4), (-0.1, 0.1), (-8, 8), (-2, 2)])
        x = [f16_from_real(rng.uniform(rng_lo, rng_hi)) for _ in range(D)]
        g = [f16_from_real(rng.uniform(0.5, 1.5)) for _ in range(D)]
        got = [real_from_f16(v) for v in rmsnorm(x, g)]
        ref = exact_rmsnorm(x, g)
        scale = max(1e-6, max(abs(v) for v in ref))
        for a, b in zip(got, ref):
            re = abs(a - b) / scale
            if re > worst_rel:
                worst_rel = re; worst_case = (rng_lo, rng_hi)
    print(f"[bar b] RMSNorm LUT-golden vs exact fp64: max rel-err (to vector peak) "
          f"= {worst_rel:.4e}  (worst range={worst_case})")
    # rsqrt LUT quantization alone (over [1,2))
    worst_r = 0.0
    for j in range(NLUT):
        f = 1.0 + j / NLUT
        worst_r = max(worst_r, abs(real_from_f16(RSQRT_LUT[j]) - 1.0 / math.sqrt(f)))
    print(f"[rsqrt LUT] max |fp16(rsqrt) - exact| over [1,2) grid = {worst_r:.4e}")
