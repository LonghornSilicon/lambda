// mac_array_ref.cpp — implementation of the bit-accurate C++ reference for
// the LonghornSilicon ACU MAC array.
//
// See MAC_ARRAY_DESIGN.md for the design rationale behind the numeric
// choices (accumulator widths, FP16 rounding policy, cost-model placeholders).

#include "mac_array_ref.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace {

// ---------------------------------------------------------------------------
// fp32 ↔ fp16 conversion. IEEE-754 binary16 with round-to-nearest-even.
// We do this manually (rather than relying on _Float16 / __fp16) to keep the
// build portable across compilers.
// ---------------------------------------------------------------------------

inline std::uint16_t float_to_half(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    const std::uint32_t sign = (bits >> 31) & 0x1u;
    const std::int32_t  exp  = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127;
    const std::uint32_t mant = bits & 0x7FFFFFu;

    // NaN
    if (exp == 128 && mant != 0) {
        return static_cast<std::uint16_t>((sign << 15) | (0x1Fu << 10) | 0x200u);
    }
    // ±Inf
    if (exp == 128) {
        return static_cast<std::uint16_t>((sign << 15) | (0x1Fu << 10));
    }
    // Overflow → ±Inf
    if (exp > 15) {
        return static_cast<std::uint16_t>((sign << 15) | (0x1Fu << 10));
    }
    // Normal range: -14 ≤ exp ≤ 15
    if (exp >= -14) {
        const std::uint32_t hexp  = static_cast<std::uint32_t>(exp + 15);
        std::uint32_t hmant       = mant >> 13;
        const std::uint32_t round = mant & 0x1FFFu;
        // Round-to-nearest-even
        if (round > 0x1000u || (round == 0x1000u && (hmant & 1u))) {
            ++hmant;
            if (hmant == 0x400u) {
                hmant = 0;
                if (hexp + 1 >= 0x1Fu) {
                    return static_cast<std::uint16_t>((sign << 15) | (0x1Fu << 10));
                }
                return static_cast<std::uint16_t>((sign << 15) | ((hexp + 1) << 10));
            }
        }
        return static_cast<std::uint16_t>((sign << 15) | (hexp << 10) | hmant);
    }
    // Subnormal range or underflow
    if (exp >= -24) {
        const std::uint32_t shift = static_cast<std::uint32_t>(-14 - exp);
        const std::uint32_t full  = mant | 0x800000u;     // restore implicit 1
        std::uint32_t hmant       = full >> (13 + shift);
        const std::uint32_t round_mask = (1u << (13 + shift)) - 1u;
        const std::uint32_t round      = full & round_mask;
        const std::uint32_t half_pt    = 1u << (12 + shift);
        if (round > half_pt || (round == half_pt && (hmant & 1u))) {
            ++hmant;
        }
        return static_cast<std::uint16_t>((sign << 15) | hmant);
    }
    // Underflow to zero
    return static_cast<std::uint16_t>(sign << 15);
}

inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 0x1u;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;

    std::uint32_t bits;
    if (exp == 0 && mant == 0) {
        bits = sign << 31;
    } else if (exp == 0) {
        // Subnormal: normalize.
        std::uint32_t m = mant;
        std::int32_t  e = -14;
        while ((m & 0x400u) == 0) {
            m <<= 1;
            --e;
        }
        m &= 0x3FFu;
        bits = (sign << 31) | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
    } else if (exp == 0x1Fu) {
        // Inf / NaN
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | (static_cast<std::uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

} // anonymous namespace

namespace lhsi::mac {

// ---------------------------------------------------------------------------
// FP16 helpers (also exposed in the public API)
// ---------------------------------------------------------------------------
float round_to_fp16(float x) {
    return half_to_float(float_to_half(x));
}

std::uint16_t float_to_fp16_bits(float x) {
    return float_to_half(x);
}

float fp16_bits_to_float(std::uint16_t bits) {
    return half_to_float(bits);
}

// ---------------------------------------------------------------------------
// matmul_int8 — signed int8 × signed int8 → int32 accumulator
// Row-major storage throughout. No saturation; the int32 accumulator is
// wide enough to hold up to ~16 M-element dot products without overflow.
// ---------------------------------------------------------------------------
void MacArray::matmul_int8(const std::int8_t* A, const std::int8_t* B,
                           std::int32_t* C,
                           std::uint32_t M, std::uint32_t K, std::uint32_t N) const {
    for (std::uint32_t i = 0; i < M; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            std::int32_t acc = 0;
            for (std::uint32_t k = 0; k < K; ++k) {
                acc += static_cast<std::int32_t>(A[i * K + k]) *
                       static_cast<std::int32_t>(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// matmul_fp16 — float storage, with fp16 rounding on each output element.
// Internally we accumulate in float (fp32). On output we round-to-nearest-
// even back into fp16 precision so the compiler sees the same numerical
// behavior the hardware will eventually produce.
// (See MAC_ARRAY_DESIGN.md §"FP16 path (open — v0.1 simplification)".)
// ---------------------------------------------------------------------------
void MacArray::matmul_fp16(const float* A, const float* B,
                           float* C,
                           std::uint32_t M, std::uint32_t K, std::uint32_t N) const {
    for (std::uint32_t i = 0; i < M; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::uint32_t k = 0; k < K; ++k) {
                // Inputs are assumed already at fp16 precision; the
                // multiplier-accumulator runs at fp32 and rounds the
                // result back to fp16 at the output.
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = round_to_fp16(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// estimate — rough cycle / energy cost model
// (See MAC_ARRAY_DESIGN.md §"Cost model" — placeholders for v0.1.)
// ---------------------------------------------------------------------------
CostEstimate MacArray::estimate(std::uint32_t M, std::uint32_t K, std::uint32_t N,
                                MacArray::DType dtype) const {
    const std::uint64_t ops = static_cast<std::uint64_t>(M) *
                              static_cast<std::uint64_t>(N) *
                              static_cast<std::uint64_t>(K);
    CostEstimate out{};
    if (dtype == DType::Int8) {
        const std::uint64_t tput = info_.int8_throughput;
        out.cycles    = (ops + tput - 1) / tput + info_.pipeline_depth_cyc;
        out.energy_pj = static_cast<double>(ops) * 0.5;    // ~0.5 pJ / INT8 MAC
    } else {
        const std::uint64_t tput = info_.fp16_throughput;
        out.cycles    = (ops + tput - 1) / tput + info_.pipeline_depth_cyc;
        out.energy_pj = static_cast<double>(ops) * 2.5;    // ~2.5 pJ / FP16 MAC
    }
    return out;
}

} // namespace lhsi::mac

// ===========================================================================
// extern "C" wrappers — singleton MacArray instance internally.
// ===========================================================================
namespace {
const lhsi::mac::MacArray& singleton() {
    static lhsi::mac::MacArray instance;
    return instance;
}
} // namespace

extern "C" {

void lhsi_mac_info(lhsi_mac_info_t* out) {
    const auto& info = singleton().info();
    out->pe_grid_m              = info.pe_grid_m;
    out->pe_grid_n              = info.pe_grid_n;
    out->int8_throughput        = info.int8_throughput;
    out->fp16_throughput        = info.fp16_throughput;
    out->int8_accumulator_bits  = info.int8_accumulator_bits;
    out->fp16_accumulator_bits  = info.fp16_accumulator_bits;
    out->pipeline_depth_cyc     = info.pipeline_depth_cyc;
    out->supports_int8          = info.supports_int8 ? 1 : 0;
    out->supports_fp16          = info.supports_fp16 ? 1 : 0;
}

void lhsi_mac_matmul_int8(const int8_t* A, const int8_t* B, int32_t* C,
                          uint32_t M, uint32_t K, uint32_t N) {
    singleton().matmul_int8(A, B, C, M, K, N);
}

void lhsi_mac_matmul_fp16(const float* A, const float* B, float* C,
                          uint32_t M, uint32_t K, uint32_t N) {
    singleton().matmul_fp16(A, B, C, M, K, N);
}

void lhsi_mac_estimate(uint32_t M, uint32_t K, uint32_t N,
                       lhsi_mac_dtype_t dtype,
                       lhsi_mac_cost_t* out) {
    const auto d = (dtype == LHSI_MAC_DTYPE_INT8)
        ? lhsi::mac::MacArray::DType::Int8
        : lhsi::mac::MacArray::DType::Fp16;
    const auto e = singleton().estimate(M, K, N, d);
    out->cycles    = e.cycles;
    out->energy_pj = e.energy_pj;
}

float lhsi_mac_round_to_fp16(float x) {
    return lhsi::mac::round_to_fp16(x);
}

uint16_t lhsi_mac_float_to_fp16_bits(float x) {
    return lhsi::mac::float_to_fp16_bits(x);
}

float lhsi_mac_fp16_bits_to_float(uint16_t bits) {
    return lhsi::mac::fp16_bits_to_float(bits);
}

} // extern "C"
