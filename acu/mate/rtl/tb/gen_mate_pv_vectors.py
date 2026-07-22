#!/usr/bin/env python3
"""Golden vectors for tb_mate_pv — bit-exact to mac_array_ref.matmul_int8.

matmul_int8 is exactly  C[n] = Σ_t int32(A[t])·int32(V[t][n])  (signed, no saturation),
which is plain Python-int arithmetic here — provably identical to the reference's
`(A.astype(int32) @ B.astype(int32))` for the M=1 rows the P·V tile computes. Pure
Python so it runs on the bare read-only venv (no numpy). Emits:

  line 1 : N ROWS
  per row: line "K", then K token lines "a v0 v1 .. v_{N-1}", then "c0 .. c_{N-1}"

Covers random rows plus the flat-attention corner (all ±127) that motivates INT32.
"""
import os, random

N = int(os.environ.get("MATE_N", "8"))
rng = random.Random(20260721)

def mm_int8_row(A, V):                     # C[n] = Σ_t A[t]·V[t][n], signed int32
    return [sum(A[t] * V[t][n] for t in range(len(A))) for n in range(N)]

rows = []
for K in (1, 4, 16, 64, 200):
    A = [rng.randint(-127, 127) for _ in range(K)]
    V = [[rng.randint(-127, 127) for _ in range(N)] for _ in range(K)]
    rows.append((A, V))
# Flat-attention corner: every code ±127 → largest |acc|; proves INT32 headroom.
rows.append(([127]*520, [[127]*N for _ in range(520)]))
rows.append(([127]*520, [[-127]*N for _ in range(520)]))

INT32 = 2**31
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mate_pv_vectors.txt")
maxabs = 0
with open(out, "w") as f:
    f.write(f"{N} {len(rows)}\n")
    for (A, V) in rows:
        K = len(A)
        C = mm_int8_row(A, V)
        for c in C:
            assert -INT32 <= c < INT32, f"acc {c} overflows int32 — reference contract broken"
            maxabs = max(maxabs, abs(c))
        f.write(f"{K}\n")
        for t in range(K):
            f.write(f"{A[t]} " + " ".join(str(x) for x in V[t]) + "\n")
        f.write(" ".join(str(x) for x in C) + "\n")
print(f"wrote {out}: N={N}, rows={len(rows)}, max|acc|={maxabs} "
      f"(fits int32: {maxabs < INT32}; overflows int24: {maxabs >= 2**23})")
