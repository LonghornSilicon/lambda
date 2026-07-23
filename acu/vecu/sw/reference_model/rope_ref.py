"""Bit-accurate golden for the VecU RoPE (rotary position embedding) slice.

Models the ACTUAL hardware algorithm the RTL implements (rtl/rope.sv), NOT an
exact float rotation:

  - A sin/cos LUT (the codebook ROM's RoPE table, architecture/dataflow_walkthrough
    §VecU) indexed by (position, channel-pair). For head dim H the vector has
    H/2 channel pairs i = 0..H/2-1 with angular frequency
        theta_i = BASE^(-2i/H)          (BASE = 10000, the Llama/Qwen RoPE base)
    and the LUT for a given position `pos` holds
        COS[pos][i] = round_fp16(cos(pos * theta_i))
        SIN[pos][i] = round_fp16(sin(pos * theta_i))
    The fp16 rounding of cos/sin IS the LUT approximation this golden models.
  - The rotation of each (even, odd) channel pair:
        x'_2i   = x_2i * cos - x_2i+1 * sin
        x'_2i+1 = x_2i * sin + x_2i+1 * cos
    Each product is an fp32 multiply and each pair-sum an fp32 add/sub, rounded
    ONCE to fp16 on emit (the same fp32 IEEE datapath as mate_pv_fp16 / softmax),
    so the block's error is the cos/sin LUT quantization, not fp16 product drift.

Two comparison bars (see __main__ / the RTL TB):
  (a) the RTL is BIT-EXACT to this LUT-golden;
  (b) this LUT-golden is within a measured tolerance of an exact fp64 rotation —
      the cos/sin LUT's actual error (reported below).

Pure Python (reuses the fp16/fp32 bit helpers of vecu_softmax_ref) so the RTL
testbench generator needs no numpy.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vecu_softmax_ref import (  # noqa: E402
    fp16_to_fp32, fp32_to_fp16, fp32_add, fp32_sub, fp32_mul,
    f16_from_real, real_from_f16,
)

# ---------------------------------------------------------------------------
# RoPE geometry + the sin/cos LUT (the codebook ROM RoPE table)
# ---------------------------------------------------------------------------
HEAD_DIM = 8            # channels per Q/K vector (even); D_PAIRS = HEAD_DIM/2 pairs
MAX_POS  = 16           # LUT positions 0..MAX_POS-1
BASE     = 10000.0      # RoPE frequency base (Llama/Qwen)
D_PAIRS  = HEAD_DIM // 2


def theta(i: int) -> float:
    return BASE ** (-2.0 * i / HEAD_DIM)


# COS_LUT[pos][i], SIN_LUT[pos][i] as fp16 codes
COS_LUT = [[f16_from_real(math.cos(pos * theta(i))) for i in range(D_PAIRS)]
           for pos in range(MAX_POS)]
SIN_LUT = [[f16_from_real(math.sin(pos * theta(i))) for i in range(D_PAIRS)]
           for pos in range(MAX_POS)]


def rope(vec_f16, pos: int):
    """vec_f16: list of HEAD_DIM fp16 codes; pos in [0, MAX_POS). -> HEAD_DIM fp16."""
    out = [0] * HEAD_DIM
    for i in range(D_PAIRS):
        x0 = fp16_to_fp32(vec_f16[2 * i])
        x1 = fp16_to_fp32(vec_f16[2 * i + 1])
        c = fp16_to_fp32(COS_LUT[pos][i])
        s = fp16_to_fp32(SIN_LUT[pos][i])
        # x0' = x0*c - x1*s ; x1' = x0*s + x1*c  (products fp32, one fp16 rounding)
        y0 = fp32_sub(fp32_mul(x0, c), fp32_mul(x1, s))
        y1 = fp32_add(fp32_mul(x0, s), fp32_mul(x1, c))
        out[2 * i] = fp32_to_fp16(y0)
        out[2 * i + 1] = fp32_to_fp16(y1)
    return out


# ---------------------------------------------------------------------------
# SV LUT emitters (embedded case ROM in rtl/rope.sv is generated from these)
# ---------------------------------------------------------------------------
def emit_sv_lut() -> str:
    lines = []
    for name, tbl in (("rope_cos", COS_LUT), ("rope_sin", SIN_LUT)):
        lines.append(f"    // {name}: index = pos*{D_PAIRS} + pair")
        for pos in range(MAX_POS):
            for i in range(D_PAIRS):
                idx = pos * D_PAIRS + i
                lines.append(f"    {name}[{idx}] = 16'h{tbl[pos][i]:04X};")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test: bar (b) — LUT-golden vs exact fp64 rotation; report the LUT error
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    if len(sys.argv) > 1 and sys.argv[1] == "--emit-sv":
        print(emit_sv_lut())
        sys.exit(0)

    def exact_rope(vec_f16, pos):
        out = []
        for i in range(D_PAIRS):
            x0 = real_from_f16(vec_f16[2 * i])
            x1 = real_from_f16(vec_f16[2 * i + 1])
            ang = pos * theta(i)
            c = math.cos(ang); s = math.sin(ang)
            out.append(x0 * c - x1 * s)
            out.append(x0 * s + x1 * c)
        return out

    rng = random.Random(20260723)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_case = None
    for _ in range(20000):
        pos = rng.randrange(MAX_POS)
        vec = [f16_from_real(rng.uniform(-4.0, 4.0)) for _ in range(HEAD_DIM)]
        got = [real_from_f16(x) for x in rope(vec, pos)]
        ref = exact_rope(vec, pos)
        scale = max(1e-6, max(abs(v) for v in ref))
        for a, b in zip(got, ref):
            ae = abs(a - b)
            re = ae / scale
            if re > worst_rel:
                worst_rel = re; worst_abs = ae; worst_case = pos
    print(f"[bar b] RoPE LUT-golden vs exact fp64 rotation: max rel-err "
          f"(to vector peak) = {worst_rel:.4e}  (abs {worst_abs:.4e}, worst pos={worst_case})")
    # LUT quantization alone
    worst_cs = 0.0
    for pos in range(MAX_POS):
        for i in range(D_PAIRS):
            ang = pos * theta(i)
            worst_cs = max(worst_cs, abs(real_from_f16(COS_LUT[pos][i]) - math.cos(ang)))
            worst_cs = max(worst_cs, abs(real_from_f16(SIN_LUT[pos][i]) - math.sin(ang)))
    print(f"[cos/sin LUT] max |fp16(cos/sin) - exact| = {worst_cs:.4e}")
