"""Bit-accurate golden for the VecU decode online-softmax slice.

Models the ACTUAL hardware algorithm the RTL implements (rtl/vecu_softmax.sv), NOT
`np.softmax`:

  - A 64-entry exp(x) LUT for x ∈ [-16, 0] with linear interpolation (Stage 9 of
    architecture/dataflow_walkthrough.md; §VecU of paper/lambda.tex). Entry
    E[i] = round_fp16(exp(-0.25·i)), i = 0..63 (E[0] = 1.0 anchors the running
    max exactly); the last interval interpolates toward exp(-16). x below -16
    clamps to 0.
  - The online-softmax recurrence with the exp(m_old - m_new) rescale (Milakov &
    Gimelshein 2018; the FlashAttention running-max/running-sum core): per row,
    keep m_i (running max, fp16) and ℓ_i (running sum-of-exp, fp32 accumulator).
    On each score s_j:  m_new = max(m, s_j);  ℓ = ℓ·exp(m-m_new) + exp(s_j-m_new).
  - Emit the L attention weights p_j = exp(s_j - m_final) · (1/ℓ_final), with the
    reciprocal 1/ℓ in fp16 (ℓ rounded to fp16) and the product rounded to fp16.

Numerics reuse the fp16/fp32 IEEE primitives of the MatE datapath (fp32_add,
fp32_mul, fp32_to_fp16, fp16 widen), plus a correctly-rounded fp16 reciprocal.
The exp/interp run in fp32 and ℓ accumulates in fp32 (like the MatE accumulator)
so the reported error reflects the LUT approximation, not fp16 sum drift.

Two comparison bars (see __main__ / the RTL TB):
  (a) the RTL is BIT-EXACT to this LUT-golden;
  (b) this LUT-golden is within a measured tolerance of exact fp64 softmax — the
      LUT's actual error (reported below), separating "RTL matches spec" from
      "the LUT is accurate enough".

Pure Python (struct only) so the RTL testbench generator needs no numpy.
"""
from __future__ import annotations

import math
import struct

# ---------------------------------------------------------------------------
# fp16 / fp32 bit helpers
# ---------------------------------------------------------------------------
def f32_from_real(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]

def real_from_f32(b: int) -> float:
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]

def f16_from_real(x: float) -> int:
    return fp32_to_fp16(f32_from_real(x))

def real_from_f16(h: int) -> float:
    return real_from_f32(fp16_to_fp32(h))

# ---------------------------------------------------------------------------
# fp16 -> fp32 widen (exact)
# ---------------------------------------------------------------------------
def fp16_to_fp32(h: int) -> int:
    s = (h >> 15) & 1; e = (h >> 10) & 0x1F; m = h & 0x3FF
    if e == 0x1F:
        return (s << 31) | (0xFF << 23) | (m << 13)          # inf / nan
    if e == 0:
        if m == 0:
            return s << 31                                   # signed zero
        # subnormal fp16 -> normal fp32
        e2 = -14
        while not (m & 0x400):
            m <<= 1; e2 -= 1
        m &= 0x3FF
        return (s << 31) | (((e2 + 127) & 0xFF) << 23) | (m << 13)
    return (s << 31) | (((e - 15 + 127) & 0xFF) << 23) | (m << 13)

# ---------------------------------------------------------------------------
# fp32 add (correctly-rounded RTNE) — same algorithm as the MatE fp32 adder
# ---------------------------------------------------------------------------
def fp32_add(a: int, b: int) -> int:
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    a_nan = ea == 0xFF and ma != 0; a_inf = ea == 0xFF and ma == 0
    b_nan = eb == 0xFF and mb != 0; b_inf = eb == 0xFF and mb == 0
    if a_nan or b_nan:
        return 0x7FC00000
    if a_inf and b_inf:
        return a if sa == sb else 0x7FC00000
    if a_inf:
        return a
    if b_inf:
        return b
    siga = ((1 << 23) | ma) if ea != 0 else ma
    sigb = ((1 << 23) | mb) if eb != 0 else mb
    eea = ea if ea != 0 else 1
    eeb = eb if eb != 0 else 1
    if eea > eeb or (eea == eeb and siga >= sigb):
        E = eea; d = eea - eeb; big = siga << 3; small0 = sigb << 3; sbig = sa; ssmall = sb
    else:
        E = eeb; d = eeb - eea; big = sigb << 3; small0 = siga << 3; sbig = sb; ssmall = sa
    if d == 0:
        small_sh = small0
    elif d > 27:
        small_sh = 1 if small0 else 0
    else:
        small_sh = small0 >> d
        if small0 & ((1 << d) - 1):
            small_sh |= 1
    sres = sbig
    summ = (big + small_sh) if sbig == ssmall else (big - small_sh)
    if summ == 0:
        return 0
    if summ & (1 << 27):
        dr = summ & 1; summ >>= 1; summ |= dr; E += 1
    for _ in range(27):
        if (summ & (1 << 26)) or E <= 1:
            break
        summ <<= 1; E -= 1
    kept = (summ >> 3) & 0xFFFFFF
    guard = (summ >> 2) & 1; roundb = (summ >> 1) & 1; sticky = summ & 1
    kept += guard & (roundb | sticky | (kept & 1))
    if kept & (1 << 24):
        kept >>= 1; E += 1
    if E >= 255:
        return (sres << 31) | (0xFF << 23)
    EF = E if (kept & (1 << 23)) else 0
    return (sres << 31) | ((EF & 0xFF) << 23) | (kept & 0x7FFFFF)

def fp32_sub(a: int, b: int) -> int:
    return fp32_add(a, b ^ 0x80000000)

# ---------------------------------------------------------------------------
# fp32 multiply (correctly-rounded RTNE)
# ---------------------------------------------------------------------------
def fp32_mul(a: int, b: int) -> int:
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    sy = sa ^ sb
    a_nan = ea == 0xFF and ma != 0; a_inf = ea == 0xFF and ma == 0
    b_nan = eb == 0xFF and mb != 0; b_inf = eb == 0xFF and mb == 0
    if a_nan or b_nan:
        return 0x7FC00000
    if a_inf or b_inf:
        if (a_inf and (eb == 0 and mb == 0)) or (b_inf and (ea == 0 and ma == 0)):
            return 0x7FC00000
        return (sy << 31) | (0xFF << 23)
    if (ea == 0 and ma == 0) or (eb == 0 and mb == 0):
        return sy << 31
    if ea == 0:
        sig_a = ma; Ea = -149
        while not (sig_a >> 23):
            sig_a <<= 1; Ea -= 1
    else:
        sig_a = (1 << 23) | ma; Ea = ea - 150
    if eb == 0:
        sig_b = mb; Eb = -149
        while not (sig_b >> 23):
            sig_b <<= 1; Eb -= 1
    else:
        sig_b = (1 << 23) | mb; Eb = eb - 150
    P = sig_a * sig_b; Ep = Ea + Eb
    msb = 47 if (P >> 47) else 46
    sh = msb - 23
    sig = P >> sh
    guard = (P >> (sh - 1)) & 1
    sticky = 1 if (P & ((1 << (sh - 1)) - 1)) else 0
    exp = msb + Ep + 127
    sig += guard & (sticky | (sig & 1))
    if sig >> 24:
        sig >>= 1; exp += 1
    if exp >= 255:
        return (sy << 31) | (0xFF << 23)
    if exp <= 0:
        return sy << 31                                      # underflow -> signed 0
    return (sy << 31) | ((exp & 0xFF) << 23) | (sig & 0x7FFFFF)

# ---------------------------------------------------------------------------
# fp32 -> fp16 round-to-nearest-even (same as the MatE narrow)
# ---------------------------------------------------------------------------
def fp32_to_fp16(fb: int) -> int:
    s = (fb >> 31) & 1; e = (fb >> 23) & 0xFF; m = fb & 0x7FFFFF
    if e == 0xFF:
        return (s << 15) | (0x7E00 if m else 0x7C00)
    if e == 0:
        return (s << 15)
    sig = (1 << 23) | m
    he = e - 112
    if he >= 31:
        return (s << 15) | 0x7C00
    drop = (14 - he) if he <= 0 else 13
    if drop > 25:
        drop = 25
    kept = sig >> drop
    guard = (sig >> (drop - 1)) & 1 if drop <= 24 else 0
    sticky = 1 if (drop >= 2 and (sig & ((1 << (drop - 1)) - 1))) else 0
    kept += guard & (sticky | (kept & 1))
    if he <= 0:
        return (s << 15) | (kept & 0x7FFF)
    if kept & (1 << 11):
        he += 1; kept >>= 1
    if he >= 31:
        return (s << 15) | 0x7C00
    return (s << 15) | ((he & 0x1F) << 10) | (kept & 0x3FF)

# ---------------------------------------------------------------------------
# fp16 reciprocal-friendly divide (a/b, RTNE) — used for 1/ℓ
# ---------------------------------------------------------------------------
def _norm11(b: int):
    e = (b >> 10) & 0x1F; m = b & 0x3FF
    if e == 0:
        sig = m; E = -24
        while not (sig & 0x400):
            sig <<= 1; E -= 1
        return sig, E
    return (0x400 | m), e - 25

def fp16_div(a: int, b: int) -> int:
    """a/b, round-to-nearest-even. Domain: a >= 0 finite, b a positive normal."""
    sa = (a >> 15) & 1; ea = (a >> 10) & 0x1F; ma = a & 0x3FF
    sb = (b >> 15) & 1
    if ea == 0 and ma == 0:
        return (sa ^ sb) << 15                               # 0 / b = signed 0
    sib, Eb = _norm11(b)
    sia, Ea = _norm11(a)
    sign = sa ^ sb
    num = sia << 13
    Q = num // sib
    rem = num - Q * sib
    Eq = Ea - Eb - 13
    msb = 13 if (Q >> 13) else 12
    sh = msb - 10
    exp = msb + Eq + 15
    if exp < 1:                                              # subnormal / underflow
        tsh = sh + (1 - exp)
        if tsh >= 14:
            return sign << 15
        sig = Q >> tsh
        guard = (Q >> (tsh - 1)) & 1
        sticky = 1 if ((Q & ((1 << (tsh - 1)) - 1)) or rem) else 0
        sig += guard & (sticky | (sig & 1))
        return (sign << 15) | (sig & 0x3FF)
    sig = Q >> sh
    guard = (Q >> (sh - 1)) & 1
    sticky = 1 if ((Q & ((1 << (sh - 1)) - 1)) or rem) else 0
    sig += guard & (sticky | (sig & 1))
    if sig & 0x800:
        sig >>= 1; exp += 1
    if exp >= 31:
        return (sign << 15) | 0x7C00
    return (sign << 15) | ((exp & 0x1F) << 10) | (sig & 0x3FF)

# ---------------------------------------------------------------------------
# The 64-entry exp LUT + the fp32 index conversion + linear interpolation
# ---------------------------------------------------------------------------
EXP_LUT = [f16_from_real(math.exp(-0.25 * i)) for i in range(64)]  # E[0]=1.0
EXP_BOT = f16_from_real(math.exp(-16.0))                            # last-interval anchor

def _frac_to_fp32(frac10: int) -> int:
    """Exact fp32 of frac10 / 1024, frac10 in [0, 1023]."""
    if frac10 == 0:
        return 0
    p = frac10.bit_length() - 1                              # MSB position (0..9)
    mant = (frac10 << (23 - p)) & 0x7FFFFF                   # drop implicit 1
    exp = (p - 10) + 127
    return ((exp & 0xFF) << 23) | mant

def _neg_to_fixed(neg32: int) -> int:
    """floor(neg * 4096) as an integer, neg32 an fp32 pattern with neg >= 0."""
    e = (neg32 >> 23) & 0xFF; m = neg32 & 0x7FFFFF
    if e == 0:
        return 0                                             # neg ~ 0
    sig = (1 << 23) | m
    shift = 138 - e                                          # neg*4096 = sig >> shift
    if shift <= 0:
        return (sig << (-shift))                             # (only for neg large; clamped by caller)
    if shift >= 48:
        return 0
    return sig >> shift

def exp_lut_fp32(x32: int) -> int:
    """exp(x) for x <= 0 via the 64-entry LUT + linear interp; returns fp32."""
    # neg = -x  (>= 0)
    neg32 = x32 ^ 0x80000000 if (x32 & 0x7FFFFFFF) else 0    # negate; +0/-0 -> 0
    if (neg32 >> 31) & 1:
        neg32 &= 0x7FFFFFFF                                  # x was > 0 (shouldn't occur) -> treat as 0 dist
        neg32 = 0
    fixed = _neg_to_fixed(neg32)
    if fixed >= 65536:                                       # neg >= 16 -> exp ~ 0
        return 0
    i = fixed >> 10
    frac10 = fixed & 0x3FF
    lo32 = fp16_to_fp32(EXP_LUT[i])
    hi32 = fp16_to_fp32(EXP_LUT[i + 1] if i < 63 else EXP_BOT)
    diff32 = fp32_sub(hi32, lo32)
    term32 = fp32_mul(_frac_to_fp32(frac10), diff32)
    return fp32_add(lo32, term32)

# ---------------------------------------------------------------------------
# Online softmax — the block's algorithm
# ---------------------------------------------------------------------------
NEG_INF_F16 = 0xFC00

def online_softmax(scores_f16):
    """scores_f16: list of fp16 patterns -> list of fp16 attention-weight patterns."""
    m16 = NEG_INF_F16
    m_real = real_from_f16(m16)
    l32 = 0                                                  # fp32 +0.0
    # ---- LOAD: running max + running sum-of-exp with the exp(m_old-m_new) rescale ----
    for s in scores_f16:
        sv = real_from_f16(s)
        m_old16 = m16
        if sv > m_real:
            m16 = s; m_real = sv
        # x_rescale = m_old - m_new  (<= 0);  x_cur = s - m_new  (<= 0)
        x_resc = fp32_sub(fp16_to_fp32(m_old16), fp16_to_fp32(m16))
        x_cur = fp32_sub(fp16_to_fp32(s), fp16_to_fp32(m16))
        resc32 = exp_lut_fp32(x_resc)
        e32 = exp_lut_fp32(x_cur)
        l32 = fp32_add(fp32_mul(l32, resc32), e32)
    # ---- EMIT: p_j = exp(s_j - m_final) * (1/ℓ_final) ----
    l16 = fp32_to_fp16(l32)
    inv_l16 = fp16_div(f16_from_real(1.0), l16)
    inv_l32 = fp16_to_fp32(inv_l16)
    out = []
    for s in scores_f16:
        x = fp32_sub(fp16_to_fp32(s), fp16_to_fp32(m16))
        e32 = exp_lut_fp32(x)
        out.append(fp32_to_fp16(fp32_mul(e32, inv_l32)))
    return out


# ---------------------------------------------------------------------------
# Self-test: bar (b) — LUT-golden vs exact fp64 softmax; report the LUT error
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    def exact_softmax(scores_f16):
        sv = [real_from_f16(s) for s in scores_f16]
        mx = max(sv)
        e = [math.exp(v - mx) for v in sv]
        Z = sum(e)
        return [v / Z for v in e]

    rng = random.Random(20260721)
    worst_rel = 0.0
    worst_case = None
    n_inf = 0
    for _ in range(6000):
        L = rng.choice([1, 2, 8, 16, 64, 520, 1000])
        lo, hi = rng.choice([(-4, 4), (-1, 1), (-8, 8), (-16, 0), (-0.5, 0.5), (-2, 2)])
        sb = [f16_from_real(rng.uniform(lo, hi)) for _ in range(L)]
        if rng.random() < 0.3:
            sb[rng.randrange(L)] = f16_from_real(rng.uniform(6, 15))
        w = online_softmax(sb)
        wr = [real_from_f16(x) for x in w]
        if any((x >> 10) & 0x1F == 0x1F for x in w):
            n_inf += 1
        we = exact_softmax(sb)
        peak = max(we)
        for a, b in zip(wr, we):
            rel = abs(a - b) / peak
            if rel > worst_rel:
                worst_rel = rel; worst_case = (L, lo, hi)
    print(f"[bar b] LUT-golden vs exact fp64 softmax: max rel-err (to peak weight) "
          f"= {worst_rel:.4e}  (worst L,range={worst_case}); inf/nan outputs={n_inf}")
    # sanity: weights sum ~ 1
    sb = [f16_from_real(v) for v in (1.0, 0.5, -0.5, 2.0, -3.0, 0.1, 0.2, -1.0)]
    ssum = sum(real_from_f16(x) for x in online_softmax(sb))
    print(f"[sanity] sum of emitted weights (should ~1.0) = {ssum:.5f}")
