// mac_array_ref.hpp — bit-accurate C++ reference model of the LonghornSilicon
// ACU MAC array (multiply-accumulate compute engine).
//
// Pairs with precision_controller_ref.hpp: the precision controller decides
// INT8 vs FP16 per tile; the MAC array executes the chosen matmul. The
// compiler emits matmul_int8 or matmul_fp16 per tile based on the gate's
// decision.
//
// See MAC_ARRAY_DESIGN.md for the design decisions baked in here.
//
// Two interfaces are exposed (same pattern as the precision controller):
//   1. C++ class `lhsi::mac::MacArray`  — native C++ codegen backends
//   2. extern "C" API (`lhsi_mac_*`)    — plain-C runtimes and FFI
//
// Both interfaces share state. Bit semantics:
//   - INT8 path: signed int8 multiply, accumulate in int32 (no saturation)
//   - FP16 path: float storage + arithmetic, with explicit fp16 round-trip
//     between matmul ops. See MAC_ARRAY_DESIGN.md §"FP16 path (open)" for
//     v0.1 caveats.

#ifndef LHSI_MAC_ARRAY_REF_HPP
#define LHSI_MAC_ARRAY_REF_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus

namespace lhsi::mac {

// ---------------------------------------------------------------------------
// Configuration. Mirrors MacArrayInfo in the chip's INFO_* registers.
// ---------------------------------------------------------------------------
struct MacArrayInfo {
    std::uint32_t pe_grid_m            = 8;      // PE grid M dimension (chip: 8×8)
    std::uint32_t pe_grid_n            = 8;      // PE grid N dimension (chip: 8×8)
    std::uint32_t int8_throughput      = 64;     // INT8 MACs per cycle (1/PE, 64 PEs → 128 GOPS)
    std::uint32_t fp16_throughput      = 16;     // FP16 MACs per cycle (¼/PE)
    std::uint32_t int8_accumulator_bits = 32;    // int32 internal accumulator
    std::uint32_t fp16_accumulator_bits = 32;    // fp32 internal accumulator
    std::uint32_t pipeline_depth_cyc   = 16;     // fixed-startup latency
    bool          supports_int8        = true;
    bool          supports_fp16        = true;
};

// ---------------------------------------------------------------------------
// Cost estimate, returned by MacArray::estimate(...).
// ---------------------------------------------------------------------------
struct CostEstimate {
    std::uint64_t cycles;      // wall-clock cycles on the MAC array
    double        energy_pj;   // picojoules total (rough; for scheduling)
};

// ---------------------------------------------------------------------------
// MAC array reference.
// ---------------------------------------------------------------------------
class MacArray {
public:
    MacArray() = default;
    explicit MacArray(MacArrayInfo info) : info_(info) {}

    // C = A @ B where A is M×K, B is K×N, C is M×N. All row-major.
    //   matmul_int8: int8 × int8 → int32 accumulator → int32 output
    //   matmul_fp16: fp16 storage; float arithmetic with fp16 rounding on
    //                each output (see MAC_ARRAY_DESIGN.md for caveats)
    void matmul_int8(const std::int8_t* A, const std::int8_t* B,
                     std::int32_t* C,
                     std::uint32_t M, std::uint32_t K, std::uint32_t N) const;

    // FP16 surface uses `float` for the API and internally rounds to fp16
    // (round-to-nearest-even) on the output. This is the v0.1 simplification.
    void matmul_fp16(const float* A, const float* B,
                     float* C,
                     std::uint32_t M, std::uint32_t K, std::uint32_t N) const;

    // Cost estimate (v0.1: rough — see MAC_ARRAY_DESIGN.md §"Cost model").
    enum class DType { Int8, Fp16 };
    CostEstimate estimate(std::uint32_t M, std::uint32_t K, std::uint32_t N,
                          DType dtype) const;

    const MacArrayInfo& info() const noexcept { return info_; }

private:
    MacArrayInfo info_{};
};

// ---------------------------------------------------------------------------
// Helpers (also useful standalone for compilers that need to round values
// to fp16 at API boundaries).
// ---------------------------------------------------------------------------
//
// Round a float to IEEE-754 binary16 precision and back to float.
// Round-to-nearest-even. NaN / Inf are preserved.
float round_to_fp16(float x);

// Pack/unpack between float and the 16-bit fp16 wire format. Useful for
// reading/writing fp16 buffers in the host without depending on _Float16.
std::uint16_t float_to_fp16_bits(float x);
float         fp16_bits_to_float(std::uint16_t bits);

} // namespace lhsi::mac

#endif // __cplusplus

// ===========================================================================
// extern "C" API — plain-C runtimes.
// ===========================================================================
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t pe_grid_m;
    uint32_t pe_grid_n;
    uint32_t int8_throughput;
    uint32_t fp16_throughput;
    uint32_t int8_accumulator_bits;
    uint32_t fp16_accumulator_bits;
    uint32_t pipeline_depth_cyc;
    int      supports_int8;
    int      supports_fp16;
} lhsi_mac_info_t;

typedef struct {
    uint64_t cycles;
    double   energy_pj;
} lhsi_mac_cost_t;

typedef enum {
    LHSI_MAC_DTYPE_INT8 = 0,
    LHSI_MAC_DTYPE_FP16 = 1,
} lhsi_mac_dtype_t;

void lhsi_mac_info(lhsi_mac_info_t* out);

void lhsi_mac_matmul_int8(const int8_t* A, const int8_t* B,
                          int32_t* C,
                          uint32_t M, uint32_t K, uint32_t N);

void lhsi_mac_matmul_fp16(const float* A, const float* B,
                          float* C,
                          uint32_t M, uint32_t K, uint32_t N);

void lhsi_mac_estimate(uint32_t M, uint32_t K, uint32_t N,
                       lhsi_mac_dtype_t dtype,
                       lhsi_mac_cost_t* out);

// FP16 utility helpers
float    lhsi_mac_round_to_fp16(float x);
uint16_t lhsi_mac_float_to_fp16_bits(float x);
float    lhsi_mac_fp16_bits_to_float(uint16_t bits);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LHSI_MAC_ARRAY_REF_HPP
