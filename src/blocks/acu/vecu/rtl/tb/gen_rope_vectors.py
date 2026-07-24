#!/usr/bin/env python3
"""Golden vectors for tb_rope — bit-exact to the RoPE LUT golden.

Uses sw/reference_model/rope_ref.rope (sin/cos LUT + fp32 rotation, one fp16
rounding). Pure Python so the TB needs no numpy.

Emits (values are uint16 fp16 codes / decimal ints):
  line 1 : ROWS
  per row: line "POS", then a line of HEAD_DIM input codes, then a line of
           HEAD_DIM rotated-output codes
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "sw", "reference_model"))
from rope_ref import rope, f16_from_real, HEAD_DIM, MAX_POS, D_PAIRS  # noqa: E402

rng = random.Random(20260723)

def rf16(lo, hi):
    return f16_from_real(rng.uniform(lo, hi))

def rand_subnormal():
    return (rng.randint(0, 1) << 15) | rng.randint(1, 0x3FF)

rows = []   # each: (pos, [HEAD_DIM fp16 codes])

# 1) pos = 0 — identity rotation (cos=1, sin=0): output must equal input
rows.append((0, [rf16(-4.0, 4.0) for _ in range(HEAD_DIM)]))

# 2) every position with a fixed moderate vector (sweeps the whole LUT)
for p in range(MAX_POS):
    rows.append((p, [rf16(-3.0, 3.0) for _ in range(HEAD_DIM)]))

# 3) large-magnitude channels at the max position (rounding stress)
rows.append((MAX_POS - 1, [rf16(-30.0, 30.0) for _ in range(HEAD_DIM)]))

# 4) tiny / subnormal channels
rows.append((7, [rand_subnormal() for _ in range(HEAD_DIM)]))

# 5) one-hot-ish (single non-zero channel) at a few positions
for p in (1, 5, 11):
    v = [f16_from_real(0.0)] * HEAD_DIM
    v[0] = f16_from_real(2.5); v[1] = f16_from_real(-1.75)
    rows.append((p, v))

# 6) alternating signs, mid position
rows.append((9, [f16_from_real((-1.0) ** k * (1.0 + 0.1 * k)) for k in range(HEAD_DIM)]))

# 7) random stress — every position x many magnitude ranges
for _ in range(300):
    p = rng.randrange(MAX_POS)
    lo, hi = rng.choice([(-1, 1), (-4, 4), (-16, 16), (-0.25, 0.25), (-40, 40)])
    rows.append((p, [rf16(lo, hi) for _ in range(HEAD_DIM)]))

# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rope_vectors.txt")
with open(out, "w") as f:
    f.write(f"{len(rows)}\n")
    for pos, vec in rows:
        y = rope(vec, pos)
        f.write(f"{pos}\n")
        f.write(" ".join(str(x) for x in vec) + "\n")
        f.write(" ".join(str(x) for x in y) + "\n")
print(f"wrote {out}: rows={len(rows)}, HEAD_DIM={HEAD_DIM}, MAX_POS={MAX_POS}, D_PAIRS={D_PAIRS}")
