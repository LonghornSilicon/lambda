#!/usr/bin/env python3
"""Golden vectors for tb_mate_qkt — bit-exact to the Q·Kᵀ decode-scoring contract.

The Q·Kᵀ engine realises mac_array_ref.matmul_fp16(Q, K.T) for the decode case (one
query): each INT8 query code Q[d] is promoted EXACTLY to fp16, each fp16(Q[d])·fp16
key K[l][d] product is exact in fp32, the head-dim (D) reduction is accumulated in
fp32 with round-to-nearest-even at every add, and the score is rounded to fp16 once.
Accumulation is SEQUENTIAL in channel order — the natural streaming-MAC order.  This
sequential-fp32 model was verified bit-exact to numpy's sequential float32 over 1500+
rows (D≤520); it agrees with numpy BLAS `@` to a few ULP (within rel_err<5e-3), an
inherent reduction-order artifact.  See docs/mate_qkt_rtl.md.

Pure Python (struct only) so it runs on the bare read-only venv (no numpy).

Emits (values: q is signed int8 decimal; keys/scores are uint16 fp16 codes decimal):
  line 1 : L ROWS                          (L = number of keys / score lanes)
  per row: line "D", then D channel lines "q k0 k1 .. k_{L-1}", then "s0 .. s_{L-1}"
"""
import os
import random
import struct

L = int(os.environ.get("MATE_QKT_L", "8"))     # number of cached keys (score lanes)

# ---------------------------------------------------------------------------
def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def int8_to_fp16(q):
    """Promote a signed int8 (|q| <= 128) to its exact fp16 code."""
    if q == 0:
        return 0x0000
    s = 1 if q < 0 else 0
    mag = -q if q < 0 else q
    p = 0
    for k in range(8):
        if (mag >> k) & 1:
            p = k
    be = p + 15
    magsh = mag << (10 - p)
    return (s << 15) | ((be & 0x1F) << 10) | (magsh & 0x3FF)

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

def qkt_row(Q, K, D):
    """score[l] = round_fp16( Σ_d fp16(Q[d])·K[l][d] ), sequential fp32 accumulate."""
    out = []
    for l in range(L):
        acc = 0  # +0.0 fp32
        for d in range(D):
            acc = fp32_add(acc, fp16_mul(int8_to_fp16(Q[d]), K[l][d]))
        out.append(fp32_to_fp16(acc))
    return out

# ---------------------------------------------------------------------------
# Test rows — random + the required corner cases. The head-dim reduction length D
# is budget-bounded by L (D·L ≤ BUDGET) so the wide-lane L=520 case stays fast; the
# long-reduction (D=520) corner is exercised whenever the budget allows it.
# ---------------------------------------------------------------------------
rng = random.Random(20260721)
BUDGET = 12000

def rand_key(lo=-4.0, hi=4.0):
    return fp16_from_float(rng.uniform(lo, hi))

def rand_subnormal():
    return (rng.randint(0, 1) << 15) | rng.randint(1, 0x3FF)

maxD = max(1, BUDGET // L)
def fitD(d):
    return max(1, min(d, maxD))

rows = []  # each: (Q list len D, K[L][D])

# 1) assorted random reductions of varying head-dim D (incl. the D=520 long
#    reduction when the L·D budget allows it)
for D0 in (1, 2, 4, 64, 128, 520):
    D = fitD(D0)
    Q = [rng.randint(-127, 127) for _ in range(D)]
    K = [[rand_key() for _ in range(D)] for _ in range(L)]
    rows.append((Q, K))

# 2) single-channel signed reduction (D=1)
rows.append(([-127], [[fp16_from_float(3.5)] for _ in range(L)]))

# 3) all-zero keys → every score +0.0
D = fitD(8)
rows.append(([rng.randint(-127, 127) for _ in range(D)],
             [[0x0000] * D for _ in range(L)]))

# 4) all-zero query → every score +0.0
D = fitD(8)
rows.append(([0] * D, [[rand_key() for _ in range(D)] for _ in range(L)]))

# 5) PEAKED: key 0 aligned with Q (dominant score), the rest near-orthogonal/small,
#    so one score dominates the row (this is what drives the FP16 gate downstream).
D = fitD(64)
Qp = [100] * D
Kp = [[fp16_from_float(1.5) for _ in range(D)]]                     # key 0: aligned, big
for l in range(1, L):
    Kp.append([fp16_from_float(rng.uniform(-0.02, 0.02)) for _ in range(D)])
rows.append((Qp, Kp))

# 6) subnormal fp16 keys — tiny products; sum rounds to fp16 subnormal / zero
D = fitD(16)
rows.append(([rng.randint(-127, 127) for _ in range(D)],
             [[rand_subnormal() for _ in range(D)] for _ in range(L)]))

# 7) signed cancellation: Q alternates sign on identical key channels → near-zero
D = fitD(8)
Qc = [(100 if (d % 2 == 0) else -100) for d in range(D)]
rows.append((Qc, [[fp16_from_float(2.0) for _ in range(D)] for _ in range(L)]))

# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mate_qkt_vectors.txt")
n_sub = n_inf = 0
with open(out, "w") as f:
    f.write(f"{L} {len(rows)}\n")
    for (Q, K) in rows:
        D = len(Q)
        S = qkt_row(Q, K, D)
        for c in S:
            e = (c >> 10) & 0x1F; mant = c & 0x3FF
            if e == 0x1F and mant == 0:
                n_inf += 1
            elif e == 0 and mant != 0:
                n_sub += 1
        f.write(f"{D}\n")
        for d in range(D):
            f.write(f"{Q[d]} " + " ".join(str(K[l][d]) for l in range(L)) + "\n")
        f.write(" ".join(str(x) for x in S) + "\n")
print(f"wrote {out}: L={L}, rows={len(rows)} (score lanes: {n_inf} inf, {n_sub} subnormal)")
