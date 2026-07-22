#!/usr/bin/env python3
"""Golden vectors for tb_mate_pv_fp16 — bit-exact to the FP16 P·V contract.

The FP16 P·V tile realises mac_array_ref.matmul_fp16's semantics for M=1 (one
attention row): each fp16×fp16 product is EXACT in fp32, the token-dim reduction
is accumulated in fp32 with round-to-nearest-even at every add, and the result is
rounded to fp16 (RTNE, overflow→inf) exactly once.  Accumulation is SEQUENTIAL in
token order — the natural order of a streaming token-reduction MAC.  This
sequential fp32 model was verified bit-exact to numpy's sequential float32
accumulation over 4000 random rows (K≤520, incl. subnormals).  It agrees with
numpy's BLAS `@` (a different, machine-dependent pairwise fp32 order) on ~99.9%
of output lanes and differs by a few ULP on the rest — well within the FP16
path's documented rel_err<5e-3 tolerance, an inherent reduction-order artifact,
not a correctness gap.  See docs/mate_pv_fp16_rtl.md.

Pure Python (struct only) so it runs on the bare read-only venv (no numpy),
operating on 16-bit fp16 / 32-bit fp32 bit patterns exactly as the RTL does.

Emits (same layout as gen_mate_pv_vectors.py, but values are uint16 fp16 codes):
  line 1 : N ROWS
  per row: line "K", then K token lines "a v0 v1 .. v_{N-1}", then "c0 .. c_{N-1}"
"""
import os
import random
import struct

N = int(os.environ.get("MATE_N", "8"))

# ---------------------------------------------------------------------------
# fp32 helpers (via struct) and the three RTL-mirrored bit routines
# ---------------------------------------------------------------------------
def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def fp16_mul(a, b):
    """16-bit × 16-bit fp16 → 32-bit fp32 product pattern (exact)."""
    sa, ea, ma = (a >> 15) & 1, (a >> 10) & 0x1F, a & 0x3FF
    sb, eb, mb = (b >> 15) & 1, (b >> 10) & 0x1F, b & 0x3FF
    sy = sa ^ sb
    a_nan = ea == 0x1F and ma != 0; a_inf = ea == 0x1F and ma == 0; a_zero = ea == 0 and ma == 0
    b_nan = eb == 0x1F and mb != 0; b_inf = eb == 0x1F and mb == 0; b_zero = eb == 0 and mb == 0
    if a_nan or b_nan:
        return 0x7FC00000
    if a_inf or b_inf:
        if (a_inf and b_zero) or (b_inf and a_zero):
            return 0x7FC00000
        return (sy << 31) | (0xFF << 23)
    if a_zero or b_zero:
        return (sy << 31)
    if ea == 0:
        siga = ma; Ea = -24
        while not (siga & 0x400):
            siga <<= 1; Ea -= 1
    else:
        siga = 0x400 | ma; Ea = ea - 25
    if eb == 0:
        sigb = mb; Eb = -24
        while not (sigb & 0x400):
            sigb <<= 1; Eb -= 1
    else:
        sigb = 0x400 | mb; Eb = eb - 25
    P = siga * sigb; Ep = Ea + Eb
    msb = 21 if (P & (1 << 21)) else 20
    eb32 = msb + Ep + 127
    mant = (P << (23 - msb)) & 0x7FFFFF
    return (sy << 31) | ((eb32 & 0xFF) << 23) | mant

def fp32_add(a, b):
    """32-bit + 32-bit fp32 → 32-bit fp32, correctly-rounded RTNE."""
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

def fp32_to_fp16(fb):
    """32-bit fp32 → 16-bit fp16, RTNE, overflow→inf, underflow→subnormal/0."""
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

def fp16_from_float(x):
    return fp32_to_fp16(f32_bits(x))

def mm_fp16_row(A, V):
    """C[n] = round_fp16( Σ_t A[t]·V[t][n] ), sequential fp32 accumulate."""
    out = []
    for n in range(N):
        acc = 0  # +0.0 fp32
        for t in range(len(A)):
            acc = fp32_add(acc, fp16_mul(A[t], V[t][n]))
        out.append(fp32_to_fp16(acc))
    return out

# ---------------------------------------------------------------------------
# Test rows — random + the required corner cases
# ---------------------------------------------------------------------------
rng = random.Random(20260721)

def rand_f16(lo=-8.0, hi=8.0):
    return fp16_from_float(rng.uniform(lo, hi))

def rand_subnormal():
    # fp16 subnormal: exp=0, mantissa 1..1023, random sign
    return (rng.randint(0, 1) << 15) | rng.randint(1, 0x3FF)

rows = []

# 1) assorted random rows of moderate values (signed) — several K
for K in (1, 2, 4, 16, 64, 200):
    A = [rand_f16() for _ in range(K)]
    V = [[rand_f16() for _ in range(N)] for _ in range(K)]
    rows.append((A, V))

# 2) all-zeros row (both +0 and -0 codes) → result +0.0
rows.append(([0x0000, 0x8000, 0x0000, 0x8000],
             [[0x0000] * N, [0x8000] * N, [0x0000] * N, [0x8000] * N]))

# 3) signed cancellation: pairs (+x·+v) then (-x·+v) → exact 0.0 per lane
K = 8
Ac = []
Vc = []
for t in range(K // 2):
    a = rand_f16(0.5, 4.0)
    vrow = [rand_f16(0.5, 4.0) for _ in range(N)]
    Ac.append(a); Vc.append(vrow[:])
    Ac.append(a ^ 0x8000); Vc.append(vrow[:])   # negate A → cancels
rows.append((Ac, Vc))

# 4) subnormal fp16 inputs — products are tiny fp32 normals; sum rounds to
#    fp16 subnormal / zero. Exercises subnormal decode + subnormal output.
K = 16
rows.append(([rand_subnormal() for _ in range(K)],
             [[rand_subnormal() for _ in range(N)] for _ in range(K)]))

# 5) long row (520 tokens) — exercises the fp32 accumulation over a long reduction
K = 520
rows.append(([rand_f16(-6.0, 6.0) for _ in range(K)],
             [[rand_f16(-6.0, 6.0) for _ in range(N)] for _ in range(K)]))

# 6) even longer row (600 tokens), small magnitudes → many RTNE rounding steps
K = 600
rows.append(([rand_f16(-1.5, 1.5) for _ in range(K)],
             [[rand_f16(-1.5, 1.5) for _ in range(N)] for _ in range(K)]))

# 7) overflow-to-inf: A=1.0, large positive V, many tokens → Σ > 65504 → fp16 inf
K = 8
one = fp16_from_float(1.0)
big = fp16_from_float(30000.0)
rows.append(([one] * K, [[big] * N for _ in range(K)]))
# and the negative direction → -inf
rows.append(([one] * K, [[big ^ 0x8000] * N for _ in range(K)]))

# 8) inf / nan input propagation: an inf V operand → inf out; +0·inf → NaN out.
#    (RTL and golden both emit canonical 0x7C00 / 0xFC00 / 0x7E00.)
INF = 0x7C00
rows.append(([one, one], [[INF] * N, [fp16_from_float(1.0)] * N]))          # inf + finite = inf
rows.append(([0x0000, one], [[INF] * N, [fp16_from_float(2.0)] * N]))       # 0·inf = NaN → propagates

# 9) mixed-magnitude row: large and tiny terms in one reduction (alignment/sticky)
K = 12
Am = []
Vm = []
for t in range(K):
    Am.append(fp16_from_float(1.0))
    if t == 0:
        Vm.append([fp16_from_float(2048.0) for _ in range(N)])
    else:
        Vm.append([rand_subnormal() for _ in range(N)])
rows.append((Am, Vm))

# 10) subnormal-OUTPUT row: products land in the fp16 subnormal band
#     [2^-24, 2^-14). One token, A=2^-14, V spanning small powers so the rounded
#     result exercises the fp32→fp16 subnormal encoding path across lanes.
small_a = fp16_from_float(2.0 ** -14)
sub_vs = [2.0 ** -6, 2.0 ** -7, 1.5 * 2.0 ** -6, 2.0 ** -8,
          3.0 * 2.0 ** -8, 2.0 ** -5, 2.0 ** -9, 2.0 ** -10]
rows.append(([small_a], [[fp16_from_float(sub_vs[n % len(sub_vs)]) for n in range(N)]]))

# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mate_pv_fp16_vectors.txt")
n_inf = n_sub = n_nan = 0
with open(out, "w") as f:
    f.write(f"{N} {len(rows)}\n")
    for (A, V) in rows:
        K = len(A)
        C = mm_fp16_row(A, V)
        for c in C:
            e = (c >> 10) & 0x1F; mant = c & 0x3FF
            if e == 0x1F and mant == 0:
                n_inf += 1
            elif e == 0x1F and mant != 0:
                n_nan += 1
            elif e == 0 and mant != 0:
                n_sub += 1
        f.write(f"{K}\n")
        for t in range(K):
            f.write(f"{A[t]} " + " ".join(str(x) for x in V[t]) + "\n")
        f.write(" ".join(str(x) for x in C) + "\n")
print(f"wrote {out}: N={N}, rows={len(rows)} "
      f"(result lanes: {n_inf} inf, {n_nan} nan, {n_sub} subnormal)")
