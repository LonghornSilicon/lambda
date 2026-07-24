#!/usr/bin/env python3
"""Golden vectors for tb_rmsnorm — bit-exact to the RMSNorm LUT golden.

Uses sw/reference_model/rmsnorm_ref.rmsnorm (fp32 sum-of-squares + rsqrt LUT +
per-element gain, one fp16 rounding). Pure Python so the TB needs no numpy.

Emits (values are uint16 fp16 codes / decimal ints):
  line 1 : ROWS
  per row: line "D", then a line of D input codes, a line of D gain codes, then
           a line of D normalized-output codes
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "sw", "reference_model"))
from rmsnorm_ref import rmsnorm, f16_from_real, D  # noqa: E402

rng = random.Random(20260723)

def rf16(lo, hi):
    return f16_from_real(rng.uniform(lo, hi))

def rand_subnormal():
    return (rng.randint(0, 1) << 15) | rng.randint(1, 0x3FF)

rows = []   # each: ([D x codes], [D g codes])

def unit_gain():
    return [f16_from_real(1.0)] * D

# 1) unit gain, unit-scale inputs
rows.append(([rf16(-1.0, 1.0) for _ in range(D)], unit_gain()))

# 2) unit gain, wide inputs (large mean(x^2) -> small scale, E negative)
rows.append(([rf16(-8.0, 8.0) for _ in range(D)], unit_gain()))

# 3) unit gain, tiny inputs (small mean -> scale > 1, eps floor matters)
rows.append(([rf16(-0.05, 0.05) for _ in range(D)], unit_gain()))

# 4) learned gain (0.5..1.5), moderate inputs
rows.append(([rf16(-2.0, 2.0) for _ in range(D)], [rf16(0.5, 1.5) for _ in range(D)]))

# 5) constant input (all equal) — clean rsqrt corner
rows.append(([f16_from_real(0.75)] * D, [rf16(0.8, 1.2) for _ in range(D)]))

# 6) one large outlier among small values (skewed sum-of-squares)
v = [rf16(-0.5, 0.5) for _ in range(D)]; v[3] = f16_from_real(9.0)
rows.append((v, unit_gain()))

# 7) subnormal inputs (near-zero mean; clamps to rsqrt(eps))
rows.append(([rand_subnormal() for _ in range(D)], unit_gain()))

# 8) negative-heavy inputs with varied gain
rows.append(([rf16(-4.0, -0.5) for _ in range(D)], [rf16(0.6, 1.4) for _ in range(D)]))

# 9) random stress — assorted input magnitudes x gain ranges
for _ in range(300):
    lo, hi = rng.choice([(-1, 1), (-4, 4), (-8, 8), (-0.1, 0.1), (-2, 2), (-0.02, 0.02)])
    glo, ghi = rng.choice([(1.0, 1.0), (0.5, 1.5), (0.25, 2.0)])
    x = [rf16(lo, hi) for _ in range(D)]
    g = ([f16_from_real(1.0)] * D) if glo == ghi else [rf16(glo, ghi) for _ in range(D)]
    rows.append((x, g))

# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rmsnorm_vectors.txt")
with open(out, "w") as f:
    f.write(f"{len(rows)}\n")
    for x, g in rows:
        y = rmsnorm(x, g)
        f.write(f"{len(x)}\n")
        f.write(" ".join(str(v) for v in x) + "\n")
        f.write(" ".join(str(v) for v in g) + "\n")
        f.write(" ".join(str(v) for v in y) + "\n")
print(f"wrote {out}: rows={len(rows)}, D={D}")
