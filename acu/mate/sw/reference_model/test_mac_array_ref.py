"""Tests for the Python MAC array reference.

Verifies:
  - INT8 matmul matches a naive numpy reference at int64 precision
  - FP16 matmul matches a naive numpy reference within ±5e-3 relative
  - FP16 round-trip helpers (round, float<->bits) are self-consistent
  - Cost estimate sane: FP16 slower than INT8, energy positive
  - Edge cases (zero, identity, signed) work

Run:    python sw/reference_model/test_mac_array_ref.py
Or:     python -m pytest sw/reference_model/test_mac_array_ref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the module under test.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mac_array_ref import (  # noqa: E402
    MacArray,
    MacArrayInfo,
    round_to_fp16,
    float_to_fp16_bits,
    fp16_bits_to_float,
)


def _rand_int8(shape, rng):
    return rng.integers(-128, 128, size=shape, dtype=np.int8)


def _rand_fp16(shape, rng):
    return rng.uniform(-2.0, 2.0, size=shape).astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# INT8
# ---------------------------------------------------------------------------
def test_int8_matches_naive():
    """INT8 matmul must match an int64 numpy reference exactly."""
    rng = np.random.default_rng(0xC0FFEE)
    mac = MacArray()

    for _ in range(20):
        M, K, N = rng.integers(1, 32, size=3)
        A = _rand_int8((M, K), rng)
        B = _rand_int8((K, N), rng)

        ref = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)
        got = mac.matmul_int8(A, B)
        np.testing.assert_array_equal(got, ref)


def test_int8_edge_cases():
    mac = MacArray()
    # all-zero
    z = np.zeros((8, 8), dtype=np.int8)
    out = mac.matmul_int8(z, z)
    assert (out == 0).all()
    # identity
    I = np.eye(8, dtype=np.int8)
    B = np.arange(-32, 32, dtype=np.int32).reshape(8, 8).astype(np.int8)
    out = mac.matmul_int8(I, B)
    np.testing.assert_array_equal(out, B.astype(np.int32))


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------
def test_fp16_matches_naive_within_tolerance():
    rng = np.random.default_rng(0xBEEF)
    mac = MacArray()

    for _ in range(20):
        M, K, N = rng.integers(1, 16, size=3)
        A = _rand_fp16((M, K), rng)
        B = _rand_fp16((K, N), rng)

        # Reference: double-precision matmul, then rounded to fp16.
        ref = (A.astype(np.float64) @ B.astype(np.float64))
        ref = ref.astype(np.float16).astype(np.float32)
        got = mac.matmul_fp16(A, B)

        denom = np.maximum(np.abs(ref), 1e-3)
        rel_err = np.abs(got - ref) / denom
        # Allow ±5e-3 because order of accumulation can differ by 1 ULP.
        assert rel_err.max() < 5e-3, f"max rel err {rel_err.max()}"


def test_fp16_edge_cases():
    mac = MacArray()
    z = np.zeros((8, 8), dtype=np.float32)
    assert (mac.matmul_fp16(z, z) == 0.0).all()

    I = np.eye(8, dtype=np.float32)
    B = (np.arange(-32, 32, dtype=np.float32).reshape(8, 8) * 0.25).astype(np.float16).astype(np.float32)
    out = mac.matmul_fp16(I, B)
    np.testing.assert_array_equal(out, B)


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------
def test_fp16_roundtrip_helpers_agree():
    values = [0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 65504.0, -65504.0,
              6.10352e-5, 1.0 / 3.0]
    for v in values:
        bits = float_to_fp16_bits(v)
        back = fp16_bits_to_float(bits)
        rt = round_to_fp16(v)
        # NaN handling sidestepped (we don't pass NaN above)
        assert back == rt, f"v={v}: fp16_bits round-trip ({back}) != round_to_fp16 ({rt})"


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------
def test_estimate_sanity():
    mac = MacArray()
    e_int8 = mac.estimate(64, 64, 64, dtype="int8")
    e_fp16 = mac.estimate(64, 64, 64, dtype="fp16")
    assert e_int8.cycles > 0
    assert e_fp16.cycles > e_int8.cycles, "FP16 must be slower than INT8 in v0.1"
    assert e_fp16.energy_pj > e_int8.energy_pj


# ---------------------------------------------------------------------------
# Run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_int8_matches_naive()
    test_int8_edge_cases()
    test_fp16_matches_naive_within_tolerance()
    test_fp16_edge_cases()
    test_fp16_roundtrip_helpers_agree()
    test_estimate_sanity()
    print("MAC array Python: ALL SELF-TESTS PASSED")
