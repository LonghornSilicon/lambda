#!/usr/bin/env python3
"""Golden vectors for tb_vecu_softmax — bit-exact to the LUT online-softmax golden.

Uses sw/reference_model/vecu_softmax_ref.online_softmax (the hardware algorithm:
64-entry exp LUT + linear interp + online running-max/running-sum recurrence with
exp(m_old-m_new) rescale). Pure Python so the TB needs no numpy.

Emits (values are uint16 fp16 codes, decimal):
  line 1 : ROWS
  per row: line "L", then a line of L score codes, then a line of L weight codes
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "sw", "reference_model"))
from vecu_softmax_ref import online_softmax, f16_from_real   # noqa: E402

rng = random.Random(20260721)

def rf16(lo, hi):
    return f16_from_real(rng.uniform(lo, hi))

def rand_subnormal():
    return (rng.randint(0, 1) << 15) | rng.randint(1, 0x3FF)

rows = []   # each: list of fp16 score codes

# 1) L=1 (single score) — trivial, weight must be 1.0
rows.append([rf16(-3.0, 3.0)])

# 2) assorted small random rows
for L in (2, 4, 8, 16, 32):
    rows.append([rf16(-4.0, 4.0) for _ in range(L)])

# 3) PEAKED: one dominant score, rest small -> one weight ~1, others ~0
pk = [rf16(-2.0, 2.0) for _ in range(16)]
pk[5] = f16_from_real(12.0)
rows.append(pk)

# 4) near-UNIFORM: all scores equal -> weights all 1/L (exercises the flat case)
rows.append([f16_from_real(0.5)] * 16)

# 5) all-NEGATIVE scores (max is negative; shift-invariance)
rows.append([rf16(-16.0, -1.0) for _ in range(16)])

# 6) increasing scores -> forces the exp(m_old-m_new) rescale on every step
rows.append([f16_from_real(-4.0 + 0.5 * k) for k in range(16)])

# 7) subnormal fp16 scores (tiny magnitudes near 0)
rows.append([rand_subnormal() for _ in range(12)])

# 8) mixed: big spread incl a far-below-max score that clamps to 0
rows.append([f16_from_real(v) for v in (3.0, -20.0, 1.5, -2.0, 0.0, -8.0, 2.9, -1.0)])

# 9) long context L=520 — the decode stress (wide fp32 sum)
rows.append([rf16(-8.0, 8.0) for _ in range(520)])

# 10) long context, near-uniform small range (worst LUT case)
rows.append([rf16(-1.0, 1.0) for _ in range(520)])

# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vecu_softmax_vectors.txt")
n_inf = n_sub = 0
with open(out, "w") as f:
    f.write(f"{len(rows)}\n")
    for scores in rows:
        w = online_softmax(scores)
        for c in w:
            e = (c >> 10) & 0x1F; mant = c & 0x3FF
            if e == 0x1F: n_inf += 1
            elif e == 0 and mant != 0: n_sub += 1
        f.write(f"{len(scores)}\n")
        f.write(" ".join(str(x) for x in scores) + "\n")
        f.write(" ".join(str(x) for x in w) + "\n")
maxL = max(len(r) for r in rows)
print(f"wrote {out}: rows={len(rows)}, maxL={maxL} "
      f"(weight lanes: {n_inf} inf, {n_sub} subnormal)")
