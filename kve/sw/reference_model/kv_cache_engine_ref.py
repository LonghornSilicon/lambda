"""
KV Cache Engine — Python reference model (bit-exact golden reference).

Implements TurboQuant+ (turbo4): 3-bit PolarQuant + 1-bit QJL.
K cache: PolarQuant + QJL (inner-product preservation).
V cache: PolarQuant only (MSE-optimized).

All arithmetic is fixed-point integer, matching the SystemVerilog RTL
cycle-for-cycle. No floating-point in the critical path.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import struct


# ---------------------------------------------------------------------------
# Fixed-point helpers
# ---------------------------------------------------------------------------

def _mask(width: int) -> int:
    return (1 << width) - 1


def _to_signed(val: int, width: int) -> int:
    val &= _mask(width)
    if val >= (1 << (width - 1)):
        return val - (1 << width)
    return val


def _to_unsigned(val: int, width: int) -> int:
    return val & _mask(width)


def _fixed_mul(a: int, b: int, a_frac: int, b_frac: int,
               out_width: int, out_frac: int) -> int:
    product = a * b
    shift = a_frac + b_frac - out_frac
    if shift > 0:
        product = (product + (1 << (shift - 1))) >> shift
    elif shift < 0:
        product <<= (-shift)
    return _to_signed(product, out_width)


# ---------------------------------------------------------------------------
# Integer square root (non-restoring, matches RTL norm_unit.sv)
# ---------------------------------------------------------------------------

def _isqrt(val: int, result_width: int) -> int:
    if val <= 0:
        return 0
    x = 0
    bit = 1 << (result_width - 1)
    while bit > 0:
        trial = x + bit
        if trial * trial <= val:
            x = trial
        bit >>= 1
    return x


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (in-place, fixed-point, self-inverse up to scale)
# ---------------------------------------------------------------------------

def _wht_inplace(vec: list, width: int) -> list:
    n = len(vec)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = vec[j]
                b = vec[j + h]
                vec[j] = _to_signed(a + b, width + 1)
                vec[j + h] = _to_signed(a - b, width + 1)
        h *= 2
    return vec


# ---------------------------------------------------------------------------
# Deterministic PRNG for rotation signs and QJL matrix (LCG)
# ---------------------------------------------------------------------------

def _lcg_next(state: int) -> Tuple[int, int]:
    state = (state * 6364136223846793005 + 1442695040888963407) & _mask(64)
    return state, state >> 32


def _generate_sign_flips(seed: int, n: int) -> List[int]:
    signs = []
    state = seed
    for _ in range(n):
        state, bits = _lcg_next(state)
        signs.append(1 if (bits & 1) == 0 else -1)
    return signs


def _generate_qjl_matrix(seed: int, n: int) -> List[List[int]]:
    matrix = []
    state = seed + 0xDEADBEEF
    for _ in range(n):
        row = []
        for _ in range(n):
            state, bits = _lcg_next(state)
            row.append(1 if (bits & 1) == 0 else -1)
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# Lloyd-Max 3-bit centroids for N(0, 1/d)
#
# Standard 8-level Lloyd-Max for N(0,1):
#   centroids = ±0.2451, ±0.7560, ±1.3439, ±2.1520
#   boundaries = 0, ±0.5006, ±1.0500, ±1.7480
#
# Scaled by sigma = 1/sqrt(d) for the unit-sphere coordinate distribution.
# Stored as fixed-point with COORD_FRAC fractional bits.
# ---------------------------------------------------------------------------

_LLOYD_MAX_3BIT_CENTROIDS_UNIT = [
    -2.1520, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1520,
]

_LLOYD_MAX_3BIT_BOUNDARIES_UNIT = [
    -1.7480, -1.0500, -0.5006, 0.0,
     0.5006,  1.0500,  1.7480,
]


def _compute_centroids_fixed(dim: int, frac: int, width: int) -> List[int]:
    import math
    sigma = 1.0 / math.sqrt(dim)
    centroids = []
    for c in _LLOYD_MAX_3BIT_CENTROIDS_UNIT:
        val = c * sigma
        fixed = int(round(val * (1 << frac)))
        centroids.append(_to_signed(fixed, width))
    return centroids


def _compute_boundaries_fixed(dim: int, frac: int, width: int) -> List[int]:
    import math
    sigma = 1.0 / math.sqrt(dim)
    boundaries = []
    for b in _LLOYD_MAX_3BIT_BOUNDARIES_UNIT:
        val = b * sigma
        fixed = int(round(val * (1 << frac)))
        boundaries.append(_to_signed(fixed, width))
    return boundaries


# ---------------------------------------------------------------------------
# sqrt(pi/2) as fixed-point constant for QJL dequantization
# ---------------------------------------------------------------------------

def _sqrt_pi_over_2_fixed(frac: int, width: int) -> int:
    import math
    val = math.sqrt(math.pi / 2.0)
    return _to_signed(int(round(val * (1 << frac))), width)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEngineInfo:
    vector_dim: int = 64
    num_centroids: int = 8
    pq_bits: int = 3
    qjl_bits: int = 1
    sram_depth: int = 1024
    norm_width: int = 16
    norm_frac: int = 8
    coord_width: int = 16
    coord_frac: int = 12
    rotation_seed: int = 42

    @property
    def n(self) -> int:
        return self.vector_dim

    @property
    def log2_n(self) -> int:
        v = self.vector_dim
        r = 0
        while v > 1:
            v >>= 1
            r += 1
        return r

    @property
    def compressed_key_bits(self) -> int:
        return (self.norm_width +
                self.vector_dim * self.pq_bits +
                self.norm_width +
                self.vector_dim * self.qjl_bits)

    @property
    def compressed_val_bits(self) -> int:
        return self.norm_width + self.vector_dim * self.pq_bits

    @property
    def uncompressed_bits(self) -> int:
        return self.vector_dim * self.coord_width

    def compression_ratio_k(self) -> float:
        return self.uncompressed_bits / self.compressed_key_bits

    def compression_ratio_v(self) -> float:
        return self.uncompressed_bits / self.compressed_val_bits

    def validate(self) -> None:
        assert self.vector_dim > 0 and (self.vector_dim & (self.vector_dim - 1)) == 0, \
            f"vector_dim must be power of 2, got {self.vector_dim}"
        assert self.num_centroids == (1 << self.pq_bits), \
            f"num_centroids must be 2^pq_bits"
        assert self.pq_bits == 3, \
            f"only pq_bits=3 supported in v0.1 (turbo4)"
        assert self.qjl_bits == 1, \
            f"only qjl_bits=1 supported in v0.1"
        assert self.norm_width >= 8
        assert self.coord_width >= 8


# ---------------------------------------------------------------------------
# Compressed data containers
# ---------------------------------------------------------------------------

@dataclass
class CompressedKey:
    norm: int = 0
    indices: List[int] = field(default_factory=list)
    residual_norm: int = 0
    signs: List[int] = field(default_factory=list)


@dataclass
class CompressedValue:
    norm: int = 0
    indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# KV Cache Engine reference model
# ---------------------------------------------------------------------------

class KVCacheEngine:

    def __init__(self, info: Optional[KVCacheEngineInfo] = None):
        if info is None:
            info = KVCacheEngineInfo()
        info.validate()
        self._info = info

        self._centroids = _compute_centroids_fixed(
            info.vector_dim, info.coord_frac, info.coord_width)
        self._boundaries = _compute_boundaries_fixed(
            info.vector_dim, info.coord_frac, info.coord_width)
        self._sign_flips = _generate_sign_flips(
            info.rotation_seed, info.vector_dim)
        self._qjl_matrix = _generate_qjl_matrix(
            info.rotation_seed, info.vector_dim)
        self._sqrt_pi_2 = _sqrt_pi_over_2_fixed(
            info.coord_frac, info.coord_width)

        self._sram_k: dict = {}
        self._sram_v: dict = {}
        self._occupancy = 0

        self._tiles_compressed = 0
        self._tiles_decompressed = 0

    @property
    def info(self) -> KVCacheEngineInfo:
        return self._info

    @property
    def occupancy(self) -> int:
        return self._occupancy

    @property
    def tiles_compressed(self) -> int:
        return self._tiles_compressed

    @property
    def tiles_decompressed(self) -> int:
        return self._tiles_decompressed

    def reset(self) -> None:
        self._sram_k.clear()
        self._sram_v.clear()
        self._occupancy = 0
        self._tiles_compressed = 0
        self._tiles_decompressed = 0

    # -------------------------------------------------------------------
    # Sub-operations (individually testable, match RTL pipeline stages)
    # -------------------------------------------------------------------

    def compute_norm(self, vec: List[int]) -> int:
        info = self._info
        sum_sq = 0
        for v in vec:
            sv = _to_signed(v, info.coord_width)
            sum_sq += sv * sv

        sq_width = 2 * info.coord_width + info.log2_n
        norm = _isqrt(sum_sq, sq_width // 2)

        shift = info.coord_frac - info.norm_frac
        if shift > 0:
            norm = (norm + (1 << (shift - 1))) >> shift
        elif shift < 0:
            norm <<= (-shift)
        return _to_unsigned(norm, info.norm_width)

    def normalize(self, vec: List[int], norm: int) -> List[int]:
        info = self._info
        if norm == 0:
            return [0] * info.vector_dim

        result = []
        for v in vec:
            sv = _to_signed(v, info.coord_width)
            numerator = sv << info.norm_frac
            normalized = numerator // norm if norm != 0 else 0
            result.append(_to_signed(normalized, info.coord_width))
        return result

    def rotate(self, vec: List[int]) -> List[int]:
        info = self._info
        work = list(vec)

        for i in range(info.vector_dim):
            work[i] = _to_signed(work[i] * self._sign_flips[i],
                                 info.coord_width)

        wht_width = info.coord_width + info.log2_n
        work = [_to_signed(x, wht_width) for x in work]
        work = _wht_inplace(work, wht_width)

        shift = info.log2_n // 2
        result = []
        for x in work:
            shifted = (x + (1 << (shift - 1))) >> shift if shift > 0 else x
            result.append(_to_signed(shifted, info.coord_width))
        return result

    def inverse_rotate(self, vec: List[int]) -> List[int]:
        info = self._info
        wht_width = info.coord_width + info.log2_n
        work = [_to_signed(x, wht_width) for x in vec]

        work = _wht_inplace(work, wht_width)

        shift = info.log2_n - (info.log2_n // 2)
        result = []
        for i, x in enumerate(work):
            shifted = (x + (1 << (shift - 1))) >> shift if shift > 0 else x
            val = _to_signed(shifted * self._sign_flips[i], info.coord_width)
            result.append(val)
        return result

    def quantize_pq(self, rotated: List[int]) -> List[int]:
        indices = []
        for coord in rotated:
            coord_s = _to_signed(coord, self._info.coord_width)
            idx = 0
            for b_idx, boundary in enumerate(self._boundaries):
                if coord_s >= boundary:
                    idx = b_idx + 1
            indices.append(idx)
        return indices

    def dequantize_pq(self, indices: List[int]) -> List[int]:
        return [self._centroids[idx] for idx in indices]

    def qjl_compress(self, residual: List[int]) -> Tuple[List[int], int]:
        info = self._info
        res_norm = self.compute_norm(residual)

        signs = []
        for row in self._qjl_matrix:
            dot = 0
            for j in range(info.vector_dim):
                r_j = _to_signed(residual[j], info.coord_width)
                dot += row[j] * r_j
            signs.append(1 if dot >= 0 else 0)
        return signs, res_norm

    def qjl_decompress(self, signs: List[int], res_norm: int) -> List[int]:
        info = self._info
        result = []

        scale_num = _fixed_mul(self._sqrt_pi_2, res_norm,
                               info.coord_frac, info.norm_frac,
                               info.coord_width + info.norm_width,
                               info.coord_frac)
        scale = scale_num // info.vector_dim if info.vector_dim > 0 else 0

        for j in range(info.vector_dim):
            acc = 0
            for i in range(info.vector_dim):
                sign_val = 1 if signs[i] else -1
                acc += self._qjl_matrix[i][j] * sign_val
            correction = _fixed_mul(acc, scale, 0, info.coord_frac,
                                    info.coord_width, info.coord_frac)
            result.append(_to_signed(correction, info.coord_width))
        return result

    # -------------------------------------------------------------------
    # Batch API
    # -------------------------------------------------------------------

    def compress_key(self, vec: List[int]) -> CompressedKey:
        info = self._info
        assert len(vec) == info.vector_dim

        norm = self.compute_norm(vec)
        normalized = self.normalize(vec, norm)
        rotated = self.rotate(normalized)
        indices = self.quantize_pq(rotated)

        dequantized = self.dequantize_pq(indices)
        residual = []
        for i in range(info.vector_dim):
            r = _to_signed(rotated[i] - dequantized[i], info.coord_width)
            residual.append(r)

        signs, res_norm = self.qjl_compress(residual)

        self._tiles_compressed += 1
        return CompressedKey(norm=norm, indices=indices,
                             residual_norm=res_norm, signs=signs)

    def compress_value(self, vec: List[int]) -> CompressedValue:
        info = self._info
        assert len(vec) == info.vector_dim

        norm = self.compute_norm(vec)
        normalized = self.normalize(vec, norm)
        rotated = self.rotate(normalized)
        indices = self.quantize_pq(rotated)

        self._tiles_compressed += 1
        return CompressedValue(norm=norm, indices=indices)

    def decompress_key(self, ck: CompressedKey) -> List[int]:
        info = self._info

        dequantized = self.dequantize_pq(ck.indices)
        qjl_correction = self.qjl_decompress(ck.signs, ck.residual_norm)
        corrected = []
        for i in range(info.vector_dim):
            val = _to_signed(dequantized[i] + qjl_correction[i],
                             info.coord_width)
            corrected.append(val)

        unrotated = self.inverse_rotate(corrected)

        result = []
        for v in unrotated:
            sv = _to_signed(v, info.coord_width)
            rescaled = _fixed_mul(sv, ck.norm,
                                  info.coord_frac, info.norm_frac,
                                  info.coord_width, info.coord_frac)
            result.append(_to_signed(rescaled, info.coord_width))

        self._tiles_decompressed += 1
        return result

    def decompress_value(self, cv: CompressedValue) -> List[int]:
        info = self._info

        dequantized = self.dequantize_pq(cv.indices)
        unrotated = self.inverse_rotate(dequantized)

        result = []
        for v in unrotated:
            sv = _to_signed(v, info.coord_width)
            rescaled = _fixed_mul(sv, cv.norm,
                                  info.coord_frac, info.norm_frac,
                                  info.coord_width, info.coord_frac)
            result.append(_to_signed(rescaled, info.coord_width))

        self._tiles_decompressed += 1
        return result

    # -------------------------------------------------------------------
    # Store / load API
    # -------------------------------------------------------------------

    def store_kv(self, addr: int, key: List[int], val: List[int]) -> None:
        info = self._info
        assert len(key) == info.vector_dim
        assert len(val) == info.vector_dim
        assert addr < info.sram_depth

        ck = self.compress_key(key)
        cv = self.compress_value(val)
        self._sram_k[addr] = ck
        self._sram_v[addr] = cv
        if addr >= self._occupancy:
            self._occupancy = addr + 1

    def load_k(self, addr: int) -> List[int]:
        assert addr in self._sram_k, f"address {addr} not populated"
        return self.decompress_key(self._sram_k[addr])

    def load_v(self, addr: int) -> List[int]:
        assert addr in self._sram_v, f"address {addr} not populated"
        return self.decompress_value(self._sram_v[addr])

    # -------------------------------------------------------------------
    # Stateless API
    # -------------------------------------------------------------------

    @staticmethod
    def decide_compress_key(vec: List[int],
                            info: Optional[KVCacheEngineInfo] = None
                            ) -> CompressedKey:
        engine = KVCacheEngine(info)
        return engine.compress_key(vec)

    @staticmethod
    def decide_compress_value(vec: List[int],
                              info: Optional[KVCacheEngineInfo] = None
                              ) -> CompressedValue:
        engine = KVCacheEngine(info)
        return engine.compress_value(vec)

    # -------------------------------------------------------------------
    # Streaming API (cycle-accurate simulation)
    # -------------------------------------------------------------------

    def tick(self, s_valid: int, s_data: int, s_last: int,
             s_user: int, addr: int) -> Tuple[int, int, int]:
        """
        Process one element per cycle.

        Args:
            s_valid: AXI-Stream tvalid
            s_data:  AXI-Stream tdata (one coordinate, COORD_WIDTH bits)
            s_last:  AXI-Stream tlast (last element of vector)
            s_user:  AXI-Stream tuser (0=key, 1=value)
            addr:    Write address for store

        Returns:
            (d_valid, d_data, d_last): Decompressed output stream
        """
        info = self._info

        if not hasattr(self, '_tick_buf'):
            self._tick_buf = []
            self._tick_user = 0
            self._tick_addr = 0

        if s_valid:
            self._tick_buf.append(_to_signed(s_data, info.coord_width))
            self._tick_user = s_user
            self._tick_addr = addr

            if s_last and len(self._tick_buf) == info.vector_dim:
                vec = self._tick_buf
                self._tick_buf = []

                if self._tick_user == 0:
                    ck = self.compress_key(vec)
                    self._sram_k[self._tick_addr] = ck
                else:
                    cv = self.compress_value(vec)
                    self._sram_v[self._tick_addr] = cv

                if self._tick_addr >= self._occupancy:
                    self._occupancy = self._tick_addr + 1

                return (1, 0, 1)

        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Hex file I/O (for RTL testbench vector exchange)
# ---------------------------------------------------------------------------

def write_hex(path: str, values: List[int], width: int) -> None:
    hex_chars = (width + 3) // 4
    with open(path, 'w') as f:
        for v in values:
            f.write(f"{_to_unsigned(v, width):0{hex_chars}x}\n")


def read_hex(path: str, width: int) -> List[int]:
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):
                values.append(_to_signed(int(line, 16), width))
    return values
