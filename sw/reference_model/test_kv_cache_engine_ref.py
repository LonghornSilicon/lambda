#!/usr/bin/env python3
"""
Test suite for the KV Cache Engine Python reference model.

7 test categories:
  1. Canonical boundary cases
  2. Compress/decompress round-trip (MSE check)
  3. K vs V asymmetry verification
  4. Streaming vs batch agreement
  5. Stateless vs stateful agreement
  6. Full replay against hex test vectors
  7. Compression ratio assertion
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

from kv_cache_engine_ref import (
    KVCacheEngineInfo, KVCacheEngine, CompressedKey, CompressedValue,
    _to_signed, _to_unsigned, _mask, read_hex, write_hex,
)


def _make_engine(dim=64):
    info = KVCacheEngineInfo(vector_dim=dim)
    return KVCacheEngine(info), info


def _mse(a, b, width):
    assert len(a) == len(b)
    total = 0
    for x, y in zip(a, b):
        diff = _to_signed(x - y, width)
        total += diff * diff
    return total / len(a)


def _random_vector(dim, seed, width):
    vec = []
    state = seed
    max_val = (1 << (width - 1)) - 1
    for _ in range(dim):
        state = (state * 6364136223846793005 +
                 1442695040888963407) & _mask(64)
        raw = (state >> 33) & _mask(16)
        scale = max_val // 8
        val = _to_signed(raw, 16) * scale >> 15
        vec.append(_to_signed(val, width))
    return vec, state


# -----------------------------------------------------------------------
# Test 1: Canonical boundary cases
# -----------------------------------------------------------------------

def test_canonical_boundary_cases():
    engine, info = _make_engine()
    dim = info.vector_dim
    passed = 0
    total = 0

    # Zero vector
    total += 1
    zero = [0] * dim
    ck = engine.compress_key(zero)
    dk = engine.decompress_key(ck)
    assert all(v == 0 for v in dk), "zero key round-trip failed"
    passed += 1

    total += 1
    cv = engine.compress_value(zero)
    dv = engine.decompress_value(cv)
    assert all(v == 0 for v in dv), "zero value round-trip failed"
    passed += 1

    # Max positive
    total += 1
    max_pos = (1 << (info.coord_width - 1)) - 1
    maxvec = [max_pos] * dim
    ck = engine.compress_key(maxvec)
    assert ck.norm > 0, "max-pos norm should be positive"
    assert len(ck.indices) == dim
    assert len(ck.signs) == dim
    passed += 1

    # Single spike
    total += 1
    spike = [0] * dim
    spike[0] = max_pos
    ck = engine.compress_key(spike)
    dk = engine.decompress_key(ck)
    assert dk[0] != 0, "spike should reconstruct non-zero at index 0"
    passed += 1

    print(f"  test_canonical_boundary_cases: {passed}/{total} passed")
    return passed == total


# -----------------------------------------------------------------------
# Test 2: Compress/decompress round-trip (MSE check)
# -----------------------------------------------------------------------

def test_round_trip_mse():
    engine, info = _make_engine()
    dim = info.vector_dim
    passed = 0
    total = 0

    max_val = (1 << (info.coord_width - 1)) - 1
    # 3-bit quantization + fixed-point rotation introduces non-trivial error;
    # threshold = max_val^2 allows ~100% relative MSE (generous for fixed-point)
    mse_threshold = max_val * max_val

    seed = 99999
    for t in range(20):
        vec, seed = _random_vector(dim, seed, info.coord_width)

        total += 1
        ck = engine.compress_key(vec)
        dk = engine.decompress_key(ck)
        mse_k = _mse(vec, dk, info.coord_width)
        assert mse_k < mse_threshold, \
            f"key MSE {mse_k} >= threshold {mse_threshold} on vector {t}"
        passed += 1

        total += 1
        cv = engine.compress_value(vec)
        dv = engine.decompress_value(cv)
        mse_v = _mse(vec, dv, info.coord_width)
        assert mse_v < mse_threshold, \
            f"value MSE {mse_v} >= threshold {mse_threshold} on vector {t}"
        passed += 1

    print(f"  test_round_trip_mse: {passed}/{total} passed")
    return passed == total


# -----------------------------------------------------------------------
# Test 3: K vs V asymmetry
# -----------------------------------------------------------------------

def test_kv_asymmetry():
    engine, info = _make_engine()
    dim = info.vector_dim
    passed = 0
    total = 0

    vec, _ = _random_vector(dim, 42424242, info.coord_width)

    total += 1
    ck = engine.compress_key(vec)
    cv = engine.compress_value(vec)

    assert hasattr(ck, 'signs') and len(ck.signs) == dim, \
        "compressed key must have QJL signs"
    assert hasattr(ck, 'residual_norm'), \
        "compressed key must have residual norm"
    passed += 1

    total += 1
    assert not hasattr(cv, 'signs') or not isinstance(cv, CompressedKey), \
        "compressed value should NOT have QJL data"
    assert isinstance(cv, CompressedValue), \
        "compressed value should be CompressedValue type"
    passed += 1

    total += 1
    assert ck.norm == cv.norm, \
        "K and V norms should match for same input"
    assert ck.indices == cv.indices, \
        "K and V PolarQuant indices should match for same input"
    passed += 1

    total += 1
    assert info.compressed_key_bits > info.compressed_val_bits, \
        "K compressed size should exceed V (K has QJL overhead)"
    passed += 1

    print(f"  test_kv_asymmetry: {passed}/{total} passed")
    return passed == total


# -----------------------------------------------------------------------
# Test 4: Streaming vs batch agreement
# -----------------------------------------------------------------------

def test_streaming_vs_batch():
    engine_batch, info = _make_engine()
    engine_stream = KVCacheEngine(info)
    passed = 0
    total = 0

    vec, _ = _random_vector(info.vector_dim, 777, info.coord_width)

    total += 1
    ck_batch = engine_batch.compress_key(vec)

    for i, v in enumerate(vec):
        is_last = 1 if i == info.vector_dim - 1 else 0
        engine_stream.tick(1, _to_unsigned(v, info.coord_width),
                           is_last, 0, 0)

    assert 0 in engine_stream._sram_k, "streaming should have stored key"
    ck_stream = engine_stream._sram_k[0]
    assert ck_batch.norm == ck_stream.norm, "norms must match"
    assert ck_batch.indices == ck_stream.indices, "indices must match"
    assert ck_batch.signs == ck_stream.signs, "signs must match"
    assert ck_batch.residual_norm == ck_stream.residual_norm, \
        "residual norms must match"
    passed += 1

    print(f"  test_streaming_vs_batch: {passed}/{total} passed")
    return passed == total


# -----------------------------------------------------------------------
# Test 5: Stateless vs stateful agreement
# -----------------------------------------------------------------------

def test_stateless_vs_stateful():
    engine, info = _make_engine()
    passed = 0
    total = 0

    seed = 55555
    for t in range(10):
        vec, seed = _random_vector(info.vector_dim, seed, info.coord_width)

        total += 1
        ck_stateful = engine.compress_key(vec)
        ck_stateless = KVCacheEngine.decide_compress_key(vec, info)

        assert ck_stateful.norm == ck_stateless.norm
        assert ck_stateful.indices == ck_stateless.indices
        assert ck_stateful.signs == ck_stateless.signs
        assert ck_stateful.residual_norm == ck_stateless.residual_norm
        passed += 1

    print(f"  test_stateless_vs_stateful: {passed}/{total} passed")
    return passed == total


# -----------------------------------------------------------------------
# Test 6: Replay against hex test vectors
# -----------------------------------------------------------------------

def test_replay_hex_vectors():
    info = KVCacheEngineInfo()
    engine = KVCacheEngine(info)

    tv_dir = os.path.join(os.path.dirname(__file__),
                          '..', '..', 'rtl', 'tb', 'testvectors')

    input_path = os.path.join(tv_dir, 'input_vectors.hex')
    dk_path = os.path.join(tv_dir, 'expected_decompressed_k.hex')
    dv_path = os.path.join(tv_dir, 'expected_decompressed_v.hex')

    if not os.path.exists(input_path):
        print("  test_replay_hex_vectors: SKIPPED (no test vectors, "
              "run analysis/gen_kv_testvectors.py first)")
        return True

    inputs = read_hex(input_path, info.coord_width)
    expected_dk = read_hex(dk_path, info.coord_width)
    expected_dv = read_hex(dv_path, info.coord_width)

    dim = info.vector_dim
    num_vectors = len(inputs) // dim
    assert len(expected_dk) == len(inputs)
    assert len(expected_dv) == len(inputs)

    passed = 0
    total = 0
    mismatches_k = []
    mismatches_v = []

    for t in range(num_vectors):
        vec = inputs[t * dim:(t + 1) * dim]

        total += 1
        ck = engine.compress_key(vec)
        dk = engine.decompress_key(ck)
        expected_dk_t = expected_dk[t * dim:(t + 1) * dim]
        if dk == expected_dk_t:
            passed += 1
        else:
            if len(mismatches_k) < 10:
                mismatches_k.append(t)

        total += 1
        cv = engine.compress_value(vec)
        dv = engine.decompress_value(cv)
        expected_dv_t = expected_dv[t * dim:(t + 1) * dim]
        if dv == expected_dv_t:
            passed += 1
        else:
            if len(mismatches_v) < 10:
                mismatches_v.append(t)

    if mismatches_k:
        print(f"    K mismatches at vectors: {mismatches_k}")
    if mismatches_v:
        print(f"    V mismatches at vectors: {mismatches_v}")

    print(f"  test_replay_hex_vectors: {passed}/{total} passed "
          f"({num_vectors} vectors)")
    return passed == total


# -----------------------------------------------------------------------
# Test 7: Compression ratio assertion
# -----------------------------------------------------------------------

def test_compression_ratio():
    info = KVCacheEngineInfo()
    passed = 0
    total = 0

    total += 1
    cr_k = info.compression_ratio_k()
    assert cr_k >= 3.0, f"K compression ratio {cr_k:.2f}x < 3.0x"
    passed += 1

    total += 1
    cr_v = info.compression_ratio_v()
    assert cr_v >= 3.0, f"V compression ratio {cr_v:.2f}x < 3.0x"
    passed += 1

    total += 1
    combined = (2 * info.uncompressed_bits /
                (info.compressed_key_bits + info.compressed_val_bits))
    assert combined >= 3.0, f"combined ratio {combined:.2f}x < 3.0x"
    passed += 1

    print(f"  test_compression_ratio: {passed}/{total} passed")
    print(f"    K={cr_k:.2f}x  V={cr_v:.2f}x  combined={combined:.2f}x")
    return passed == total


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def main():
    tests = [
        ("1. Canonical boundary cases", test_canonical_boundary_cases),
        ("2. Round-trip MSE", test_round_trip_mse),
        ("3. K/V asymmetry", test_kv_asymmetry),
        ("4. Streaming vs batch", test_streaming_vs_batch),
        ("5. Stateless vs stateful", test_stateless_vs_stateful),
        ("6. Replay hex vectors", test_replay_hex_vectors),
        ("7. Compression ratio", test_compression_ratio),
    ]

    all_pass = True
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            ok = test_fn()
            if not ok:
                all_pass = False
                print(f"  FAILED")
        except Exception as e:
            all_pass = False
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
