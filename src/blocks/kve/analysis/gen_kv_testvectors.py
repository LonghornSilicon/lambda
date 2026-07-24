#!/usr/bin/env python3
"""
Generate test vectors for the KV Cache Engine RTL testbench.

Outputs hex files readable by $readmemh:
  - input_vectors.hex          : raw FP16→fixed-point KV vectors
  - expected_compressed_k.hex  : compressed key output (norm + indices + res_norm + signs)
  - expected_compressed_v.hex  : compressed value output (norm + indices)
  - expected_decompressed_k.hex: reconstructed key vectors
  - expected_decompressed_v.hex: reconstructed value vectors
  - metadata.hex               : test metadata (vector count, dim, etc.)
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'sw', 'reference_model'))

from kv_cache_engine_ref import (
    KVCacheEngineInfo, KVCacheEngine, CompressedKey, CompressedValue,
    write_hex, _to_signed, _to_unsigned, _mask
)


def generate_test_vectors(info: KVCacheEngineInfo, seed: int = 12345):
    """Generate a suite of test vectors covering key scenarios."""
    vectors = []
    labels = []

    dim = info.vector_dim

    # --- 1. Zero vector ---
    vectors.append([0] * dim)
    labels.append("zero")

    # --- 2. Max positive ---
    max_pos = (1 << (info.coord_width - 1)) - 1
    vectors.append([max_pos] * dim)
    labels.append("max_positive")

    # --- 3. Max negative ---
    max_neg = -(1 << (info.coord_width - 1))
    vectors.append([_to_signed(max_neg, info.coord_width)] * dim)
    labels.append("max_negative")

    # --- 4. Single spike (one large element, rest zero) ---
    v = [0] * dim
    v[0] = max_pos
    vectors.append(v)
    labels.append("single_spike_pos")

    v = [0] * dim
    v[dim // 2] = _to_signed(max_neg, info.coord_width)
    vectors.append(v)
    labels.append("single_spike_neg")

    # --- 5. Alternating ±1 (fixed-point 1.0 = 1 << coord_frac) ---
    one = 1 << info.coord_frac
    if one > max_pos:
        one = max_pos
    v = []
    for i in range(dim):
        v.append(one if i % 2 == 0 else _to_signed(-one, info.coord_width))
    vectors.append(v)
    labels.append("alternating")

    # --- 6. Ramp (linearly increasing) ---
    step = max(1, max_pos // dim)
    v = [_to_signed(i * step - (dim // 2) * step, info.coord_width)
         for i in range(dim)]
    vectors.append(v)
    labels.append("ramp")

    # --- 7-26. Pseudorandom Gaussian-like vectors (LCG-based) ---
    state = seed
    for t in range(20):
        v = []
        for _ in range(dim):
            state = (state * 6364136223846793005 +
                     1442695040888963407) & _mask(64)
            raw = (state >> 33) & _mask(16)
            scale = max_pos // 8
            val = _to_signed(raw, 16) * scale >> 15
            v.append(_to_signed(val, info.coord_width))
        vectors.append(v)
        labels.append(f"random_{t}")

    # --- 27. Near-boundary vector (values near quantization boundaries) ---
    from kv_cache_engine_ref import _compute_boundaries_fixed
    boundaries = _compute_boundaries_fixed(dim, info.coord_frac,
                                           info.coord_width)
    v = []
    for i in range(dim):
        b_idx = i % len(boundaries)
        offset = 1 if (i // len(boundaries)) % 2 == 0 else -1
        v.append(_to_signed(boundaries[b_idx] + offset, info.coord_width))
    vectors.append(v)
    labels.append("near_boundary")

    # --- 28. Small magnitude (tests norm precision) ---
    v = [_to_signed(i - dim // 2, info.coord_width) for i in range(dim)]
    vectors.append(v)
    labels.append("small_magnitude")

    return vectors, labels


def pack_compressed_key(ck: CompressedKey, info: KVCacheEngineInfo) -> list:
    """Pack a CompressedKey into a flat list of fixed-width values for hex output."""
    packed = []
    packed.append(_to_unsigned(ck.norm, info.norm_width))
    for idx in ck.indices:
        packed.append(idx & _mask(info.pq_bits))
    packed.append(_to_unsigned(ck.residual_norm, info.norm_width))
    for s in ck.signs:
        packed.append(s & 1)
    return packed


def pack_compressed_value(cv: CompressedValue, info: KVCacheEngineInfo) -> list:
    """Pack a CompressedValue into a flat list."""
    packed = []
    packed.append(_to_unsigned(cv.norm, info.norm_width))
    for idx in cv.indices:
        packed.append(idx & _mask(info.pq_bits))
    return packed


def main():
    info = KVCacheEngineInfo()
    engine = KVCacheEngine(info)

    vectors, labels = generate_test_vectors(info)

    outdir = os.path.join(os.path.dirname(__file__),
                          '..', 'rtl', 'tb', 'testvectors')
    os.makedirs(outdir, exist_ok=True)

    all_input = []
    all_compressed_k = []
    all_compressed_v = []
    all_decompressed_k = []
    all_decompressed_v = []

    for i, (vec, label) in enumerate(zip(vectors, labels)):
        ck = engine.compress_key(vec)
        cv = engine.compress_value(vec)
        dk = engine.decompress_key(ck)
        dv = engine.decompress_value(cv)

        for v in vec:
            all_input.append(v)

        all_compressed_k.extend(pack_compressed_key(ck, info))
        all_compressed_v.extend(pack_compressed_value(cv, info))

        for v in dk:
            all_decompressed_k.append(v)
        for v in dv:
            all_decompressed_v.append(v)

    write_hex(os.path.join(outdir, 'input_vectors.hex'),
              all_input, info.coord_width)

    max_ck_width = max(info.norm_width, info.pq_bits, 1)
    write_hex(os.path.join(outdir, 'expected_compressed_k.hex'),
              all_compressed_k, max_ck_width)

    max_cv_width = max(info.norm_width, info.pq_bits)
    write_hex(os.path.join(outdir, 'expected_compressed_v.hex'),
              all_compressed_v, max_cv_width)

    write_hex(os.path.join(outdir, 'expected_decompressed_k.hex'),
              all_decompressed_k, info.coord_width)

    write_hex(os.path.join(outdir, 'expected_decompressed_v.hex'),
              all_decompressed_v, info.coord_width)

    meta = [len(vectors), info.vector_dim, info.coord_width,
            info.pq_bits, info.qjl_bits, info.norm_width]
    write_hex(os.path.join(outdir, 'metadata.hex'), meta, 16)

    print(f"Generated {len(vectors)} test vectors ({', '.join(labels[:5])}, ...)")
    print(f"  Input elements:        {len(all_input)}")
    print(f"  Compressed K elements: {len(all_compressed_k)}")
    print(f"  Compressed V elements: {len(all_compressed_v)}")
    print(f"  Output dir: {outdir}")

    cr_k = info.compression_ratio_k()
    cr_v = info.compression_ratio_v()
    print(f"  Compression ratio K: {cr_k:.2f}x")
    print(f"  Compression ratio V: {cr_v:.2f}x")
    print(f"  Combined (K+V):      "
          f"{2 * info.uncompressed_bits / (info.compressed_key_bits + info.compressed_val_bits):.2f}x")


if __name__ == '__main__':
    main()
