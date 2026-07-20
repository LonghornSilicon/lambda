#!/usr/bin/env python3
"""Numpy hw-faithful mirror of the WHT-rotated INT3 VALUE codec (channelquant_ref.hpp).

Bit-for-bit parity target for the C++ reference: same fp16 semantics (numpy float16 add ==
real_to_f16(exact double sum)), same round-half-to-even (np.rint == srint_ll), same raw
Walsh-Hadamard butterfly, same 1/D (2^-k) decode scale. (WHT value rotation: Abhiram Bandi
+ Chaithu Talasila.) Reads/writes the same hex formats as test_wht_value_ref.cpp.
"""
import sys
import numpy as np

EPS = 1.0 / 16384.0            # 2^-14, matches channelquant_ref.hpp
QMIN, QMAX = -4, 3             # 3-bit signed


def f16_to_real(h16):          # fp16 bits -> python double
    return np.float64(np.frombuffer(np.uint16(h16).tobytes(), np.float16)[0])

def real_to_f16(x):            # double -> fp16 value (round-half-even)
    return np.float16(x)

def f16bits(v):                # fp16 value -> 16-bit int
    return int(np.float16(v).view(np.uint16))

def f32bits(x):                # double -> fp32 bits (round-half-even)
    return int(np.float32(x).view(np.uint32))


def fwht_raw_f16(row):         # row: np.float16 [D], in-place raw butterfly (add/sub only)
    D = len(row); h = 1
    while h < D:
        for i in range(0, D, 2 * h):
            for j in range(i, i + h):
                a = np.float64(row[j]); b = np.float64(row[j + h])
                row[j] = np.float16(a + b)          # == real_to_f16(a+b)
                row[j + h] = np.float16(a - b)
        h *= 2
    return row


def compress_row(row_f16):     # row_f16: np.float16 [D] -> (scale_f16_val, codes[D])
    r = fwht_raw_f16(row_f16.copy())
    amax = np.float16(np.max(np.abs(r)))            # winner magnitude, fp16
    s = np.float64(amax) / QMAX
    if s < EPS: s = EPS
    s = np.float16(s)                               # scale_from_amax
    codes = []
    for d in range(len(r)):
        q = np.rint(np.float64(r[d]) / np.float64(s))   # srint_ll(x/s), half-even
        q = int(max(QMIN, min(QMAX, q)))
        codes.append(q)
    return s, codes


def decompress_row(scale_f16, codes, D):
    r = np.empty(D, np.float16)
    for d in range(D):
        r[d] = np.float16(np.float64(codes[d]) * np.float64(scale_f16))   # dequant -> fp16
    fwht_raw_f16(r)
    inv_d = 1.0 / float(D)                          # exact (D = 2^k)
    return [np.float32(np.float64(r[d]) * inv_d) for d in range(D)]        # x(1/D) -> fp32


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    lines = open(inp).read().split("\n")
    D, T, _bits = (int(x) for x in lines[0].split())
    vhat_bits = []
    for t in range(T):
        vals = lines[1 + t].split()
        row = np.array([np.frombuffer(np.uint16(int(v, 16)).tobytes(), np.float16)[0] for v in vals],
                       dtype=np.float16)
        s, codes = compress_row(row)
        vh = decompress_row(s, codes, D)
        vhat_bits.append([f32bits(x) for x in vh])
    with open(outp, "w") as f:
        for t in range(T):
            f.write(" ".join(f"{vhat_bits[t][d]:08x}" for d in range(D)) + "\n")
    print(f"wrote {outp}: T={T} D={D}")


if __name__ == "__main__":
    main()
