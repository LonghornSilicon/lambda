#!/usr/bin/env python3
"""
Reconstruction quality gate for the KV Cache Engine reference model.

Gates on cosine_similarity > 0.9 and relative_mse < 0.1 for both K and V
paths, on N(0, 1/sqrt(d)) Gaussians and on _random_vector inputs.

This is the accuracy gate that test_kv_cache_engine_ref.py's MSE threshold
(max_val^2) does not provide. See issue #1.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from kv_cache_engine_ref import (
    KVCacheEngineInfo, KVCacheEngine,
    _to_signed, _to_unsigned, _mask,
)


COSINE_THRESHOLD = 0.9
RMSE_THRESHOLD = 0.1


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


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == 0 and nb == 0 else 0.0
    return float(a @ b / (na * nb))


def relative_mse(original, reconstructed):
    a = np.array(original, dtype=np.float64)
    b = np.array(reconstructed, dtype=np.float64)
    signal_power = np.mean(a ** 2)
    if signal_power == 0:
        return 0.0
    return float(np.mean((a - b) ** 2) / signal_power)


def test_gaussian_quality():
    """N(0, 1/sqrt(d)) Gaussians — the Lloyd-Max design point."""
    info = KVCacheEngineInfo(vector_dim=64)
    engine = KVCacheEngine(info)
    rng = np.random.default_rng(0)

    n_vectors = 50
    k_cosines, v_cosines = [], []
    k_rmses, v_rmses = [], []

    for _ in range(n_vectors):
        v = rng.standard_normal(64) / np.sqrt(64)
        q = np.clip(np.round(v * (1 << info.coord_frac)),
                    -(1 << (info.coord_width - 1)),
                    (1 << (info.coord_width - 1)) - 1).astype(np.int32).tolist()

        dk = engine.decompress_key(engine.compress_key(q))
        dv = engine.decompress_value(engine.compress_value(q))

        k_cosines.append(cosine_sim(q, dk))
        v_cosines.append(cosine_sim(q, dv))
        k_rmses.append(relative_mse(q, dk))
        v_rmses.append(relative_mse(q, dv))

    med_k_cos = float(np.median(k_cosines))
    med_v_cos = float(np.median(v_cosines))
    med_k_rmse = float(np.median(k_rmses))
    med_v_rmse = float(np.median(v_rmses))

    print(f"  Gaussian (n={n_vectors}):")
    print(f"    K: median cosine={med_k_cos:.3f}  median rMSE={med_k_rmse:.4f}")
    print(f"    V: median cosine={med_v_cos:.3f}  median rMSE={med_v_rmse:.4f}")

    passed = 0
    total = 4
    assert med_k_cos > COSINE_THRESHOLD, \
        f"K cosine {med_k_cos:.3f} <= {COSINE_THRESHOLD}"
    passed += 1
    assert med_v_cos > COSINE_THRESHOLD, \
        f"V cosine {med_v_cos:.3f} <= {COSINE_THRESHOLD}"
    passed += 1
    assert med_k_rmse < RMSE_THRESHOLD, \
        f"K rMSE {med_k_rmse:.4f} >= {RMSE_THRESHOLD}"
    passed += 1
    assert med_v_rmse < RMSE_THRESHOLD, \
        f"V rMSE {med_v_rmse:.4f} >= {RMSE_THRESHOLD}"
    passed += 1

    print(f"  {passed}/{total} passed")
    return True


def test_random_vector_quality():
    """_random_vector inputs — this repo's own test distribution."""
    info = KVCacheEngineInfo(vector_dim=64)
    engine = KVCacheEngine(info)

    n_vectors = 20
    seed = 99999
    k_cosines, v_cosines = [], []
    k_rmses, v_rmses = [], []

    for _ in range(n_vectors):
        vec, seed = _random_vector(info.vector_dim, seed, info.coord_width)

        dk = engine.decompress_key(engine.compress_key(vec))
        dv = engine.decompress_value(engine.compress_value(vec))

        k_cosines.append(cosine_sim(vec, dk))
        v_cosines.append(cosine_sim(vec, dv))
        k_rmses.append(relative_mse(vec, dk))
        v_rmses.append(relative_mse(vec, dv))

    med_k_cos = float(np.median(k_cosines))
    med_v_cos = float(np.median(v_cosines))
    med_k_rmse = float(np.median(k_rmses))
    med_v_rmse = float(np.median(v_rmses))

    print(f"  _random_vector (n={n_vectors}):")
    print(f"    K: median cosine={med_k_cos:.3f}  median rMSE={med_k_rmse:.4f}")
    print(f"    V: median cosine={med_v_cos:.3f}  median rMSE={med_v_rmse:.4f}")

    passed = 0
    total = 4
    assert med_k_cos > COSINE_THRESHOLD, \
        f"K cosine {med_k_cos:.3f} <= {COSINE_THRESHOLD}"
    passed += 1
    assert med_v_cos > COSINE_THRESHOLD, \
        f"V cosine {med_v_cos:.3f} <= {COSINE_THRESHOLD}"
    passed += 1
    assert med_k_rmse < RMSE_THRESHOLD, \
        f"K rMSE {med_k_rmse:.4f} >= {RMSE_THRESHOLD}"
    passed += 1
    assert med_v_rmse < RMSE_THRESHOLD, \
        f"V rMSE {med_v_rmse:.4f} >= {RMSE_THRESHOLD}"
    passed += 1

    print(f"  {passed}/{total} passed")
    return True


def main():
    tests = [
        ("1. Gaussian N(0, 1/sqrt(d))", test_gaussian_quality),
        ("2. _random_vector inputs", test_random_vector_quality),
    ]

    all_pass = True
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            ok = test_fn()
            if not ok:
                all_pass = False
                print("  FAILED")
        except Exception as e:
            all_pass = False
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL QUALITY GATES PASSED")
    else:
        print("QUALITY GATES FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
