"""Bit-accurate Python reference model of the LonghornSilicon ACU MAC array.

Pairs with the C++ implementation in mac_array_ref.{hpp,cpp}; both must
produce identical output for identical input (verified by the test suite).

See MAC_ARRAY_DESIGN.md for the design decisions baked in here:
  - INT8 path: signed int8 multiply, accumulate in int32, no saturation
  - FP16 path: float storage + arithmetic, rounded to fp16 (round-nearest-
    even) on each matmul output. v0.1 simplification — see design doc.

Two abstraction levels:

  Class-based (mirrors the C++ class):
      mac = MacArray()
      mac.matmul_int8(A, B, C, M, K, N)
      mac.matmul_fp16(A, B, C, M, K, N)
      cost = mac.estimate(M, K, N, dtype="int8")

  Free functions (no state — useful for compilers that want pure functions):
      C = matmul_int8(A, B)        # numpy arrays
      C = matmul_fp16(A, B)
      cost = estimate(M, K, N, dtype="int8")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — matches the C++ MacArrayInfo
# ---------------------------------------------------------------------------
@dataclass
class MacArrayInfo:
    pe_grid_m: int = 8               # chip: 8×8 grid = 64 PEs
    pe_grid_n: int = 8
    int8_throughput: int = 64        # INT8 MACs / cycle (1 per PE, 64 PEs → 128 GOPS)
    fp16_throughput: int = 16        # FP16 MACs / cycle (¼ per PE)
    int8_accumulator_bits: int = 32
    fp16_accumulator_bits: int = 32
    pipeline_depth_cyc: int = 16
    supports_int8: bool = True
    supports_fp16: bool = True


@dataclass
class CostEstimate:
    cycles: int
    energy_pj: float


# ---------------------------------------------------------------------------
# FP16 round-trip helpers
# ---------------------------------------------------------------------------
def round_to_fp16(x: float) -> float:
    """Round a Python float to IEEE-754 binary16 precision and back to float."""
    # numpy provides bit-accurate fp16 conversion. Going through float16
    # round-trips the value at fp16 precision (round-to-nearest-even).
    return float(np.float32(np.float16(x)))


def float_to_fp16_bits(x: float) -> int:
    """Convert a float to its 16-bit IEEE-754 binary16 representation."""
    half = np.float16(x)
    return int(np.frombuffer(half.tobytes(), dtype=np.uint16)[0])


def fp16_bits_to_float(bits: int) -> float:
    """Inverse of float_to_fp16_bits."""
    u = np.array([bits], dtype=np.uint16)
    return float(np.frombuffer(u.tobytes(), dtype=np.float16)[0])


def _round_array_to_fp16(arr: np.ndarray) -> np.ndarray:
    """Round every element of a float32 array to fp16 precision."""
    return arr.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# MAC array class
# ---------------------------------------------------------------------------
class MacArray:
    """Bit-accurate reference for the LonghornSilicon MAC array."""

    def __init__(self, info: MacArrayInfo | None = None) -> None:
        self.info = info or MacArrayInfo()

    # ------------------------------------------------------------------
    # INT8 path — signed int8 × signed int8 → int32
    # ------------------------------------------------------------------
    def matmul_int8(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """C = A @ B in INT8, returning int32. Inputs must be int8.

        int32 is REQUIRED (not headroom) for the P·V tile: it reduces over the token
        dim, so a flat attention row of length L needs 14+ceil(log2 L) bits — INT24
        overflows past ~520 tokens. See analysis/pv_accumulator_width.py and arch.yml
        accumulator_rationale. (Hidden-dim reductions — W4A8 GEMM, Q·Kᵀ — fit INT24.)"""
        if A.dtype != np.int8 or B.dtype != np.int8:
            raise TypeError("matmul_int8 inputs must be np.int8")
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch: A={A.shape}, B={B.shape}")
        # Promote to int32 before multiply so we don't overflow int8.
        return (A.astype(np.int32) @ B.astype(np.int32)).astype(np.int32)

    # ------------------------------------------------------------------
    # FP16 path — float storage with fp16 rounding on outputs
    # ------------------------------------------------------------------
    def matmul_fp16(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """C = A @ B with fp16 semantics on outputs, returning float32.

        Inputs are float32 (already at fp16 precision). The matmul runs
        at float32 precision internally and rounds each output element
        back to fp16. This matches what the hardware will do in v0.1
        (see MAC_ARRAY_DESIGN.md for the open caveats).
        """
        if A.dtype != np.float32 or B.dtype != np.float32:
            raise TypeError("matmul_fp16 inputs must be np.float32")
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch: A={A.shape}, B={B.shape}")
        # Confirm inputs are at fp16 precision; the compiler should be
        # passing in already-rounded values, but the round here is a
        # safety net (idempotent for fp16-precision floats).
        A_fp16 = _round_array_to_fp16(A)
        B_fp16 = _round_array_to_fp16(B)
        # Multiply-accumulate at fp32, round output back to fp16.
        out = (A_fp16 @ B_fp16).astype(np.float32)
        return _round_array_to_fp16(out)

    # ------------------------------------------------------------------
    # Cost estimate — rough placeholders, see MAC_ARRAY_DESIGN.md
    # ------------------------------------------------------------------
    def estimate(self, M: int, K: int, N: int, dtype: str) -> CostEstimate:
        ops = M * K * N
        if dtype == "int8":
            tput = self.info.int8_throughput
            energy = ops * 0.5      # pJ per INT8 MAC
        elif dtype == "fp16":
            tput = self.info.fp16_throughput
            energy = ops * 2.5      # pJ per FP16 MAC
        else:
            raise ValueError(f"unknown dtype: {dtype}")
        cycles = (ops + tput - 1) // tput + self.info.pipeline_depth_cyc
        return CostEstimate(cycles=cycles, energy_pj=energy)


# ---------------------------------------------------------------------------
# Free functions — stateless wrappers
# ---------------------------------------------------------------------------
_singleton: MacArray | None = None


def _get_singleton() -> MacArray:
    global _singleton
    if _singleton is None:
        _singleton = MacArray()
    return _singleton


def matmul_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _get_singleton().matmul_int8(A, B)


def matmul_fp16(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _get_singleton().matmul_fp16(A, B)


def estimate(M: int, K: int, N: int, dtype: str) -> CostEstimate:
    return _get_singleton().estimate(M, K, N, dtype)


__all__ = [
    "MacArray",
    "MacArrayInfo",
    "CostEstimate",
    "matmul_int8",
    "matmul_fp16",
    "estimate",
    "round_to_fp16",
    "float_to_fp16_bits",
    "fp16_bits_to_float",
]
