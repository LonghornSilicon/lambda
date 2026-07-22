// precision_controller_ref.hpp — bit-accurate C++ reference model of the
// LonghornSilicon ACU precision controller.
//
// This is a direct port of `precision_controller_ref.py`. Both models are
// verified bit-exact against the SystemVerilog RTL TB across all 143 replay
// tiles in rtl/tb/testvectors/.
//
// Two interfaces are exposed:
//   1. C++ class `lhsi::PrecisionController` — for native-C++ codegen
//      backends (MLIR, TVM C++ runtime, custom compilers).
//   2. extern "C" API (`lhsi_pc_*`) — for plain-C runtimes, FFI bindings,
//      and ABI-stable linking from any language.
//
// Both interfaces share the same underlying state machine; using the C API
// internally instantiates the C++ class. Bit semantics:
//   - All arithmetic is unsigned fixed-width modular arithmetic
//   - SCORE_WIDTH-bit two's complement scores on input
//   - SUM_WIDTH = SCORE_WIDTH + log2(N) bits for the running sum
//   - CMP_WIDTH = SUM_WIDTH + 4 bits for the comparison
//   - THRESHOLD is hardcoded 10 (implemented as `(sum<<3) + (sum<<1)`)
//
// Build: see Makefile in this directory. Requires C++17 or later.

#ifndef LHSI_PRECISION_CONTROLLER_REF_HPP
#define LHSI_PRECISION_CONTROLLER_REF_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus

#include <vector>
#include <stdexcept>

namespace lhsi {

// ---------------------------------------------------------------------------
// Configuration. Mirrors PrecisionControllerInfo in the Python ref.
// ---------------------------------------------------------------------------
struct PrecisionControllerInfo {
    std::uint32_t block_m     = 64;
    std::uint32_t block_n     = 64;
    std::uint32_t score_width = 8;
    std::uint32_t threshold   = 10;

    constexpr std::uint32_t n()          const { return block_m * block_n; }
    std::uint32_t           log2_n()     const; // computed via bitscan
    constexpr std::uint32_t sum_width()  const { return score_width + log2_n_const_(); }
    constexpr std::uint32_t cmp_width()  const { return sum_width() + 4; }

    // Validate at construction.
    void validate() const;

private:
    // constexpr-friendly log2 of an integer power of two, for use in
    // sum_width()/cmp_width(). Throws at runtime if N is not a power of two.
    constexpr std::uint32_t log2_n_const_() const {
        std::uint32_t v = n();
        std::uint32_t l = 0;
        while (v >>= 1) ++l;
        return l;
    }
};

// ---------------------------------------------------------------------------
// Bit-accurate streaming model.
// ---------------------------------------------------------------------------
class PrecisionController {
public:
    explicit PrecisionController(PrecisionControllerInfo info = {});

    // Low-level — one clock tick. Mirrors the SV ports.
    void tick(bool s_valid, std::int32_t s_data, bool s_last);

    // Output read-back (mirrors d_valid, d_fp16 SV ports + the chip's status registers).
    bool d_valid()                       const noexcept { return d_valid_; }
    bool d_fp16()                        const noexcept { return d_fp16_; }
    std::uint32_t tiles_processed()      const noexcept { return tiles_processed_; }
    const std::vector<bool>& decision_history() const noexcept { return decision_history_; }

    // Reset accumulators (asserts rst_n=0 for one cycle).
    void reset();

    // High-level — stream a complete tile and return its decision.
    bool process_tile(const std::int32_t* scores, std::size_t n);
    bool process_tile(const std::vector<std::int32_t>& scores);

    // Stateless one-shot — pure function, no class state needed.
    static bool decide(const std::int32_t* scores, std::size_t n,
                       PrecisionControllerInfo info = {});

    // Read the configuration (mirrors the chip's INFO_* registers).
    const PrecisionControllerInfo& info() const noexcept { return info_; }

private:
    PrecisionControllerInfo info_;
    std::uint32_t max_acc_;
    std::uint64_t sum_acc_;
    bool          d_valid_;
    bool          d_fp16_;
    std::uint32_t tiles_processed_;
    std::vector<bool> decision_history_;
};

} // namespace lhsi

#endif // __cplusplus

// ===========================================================================
// extern "C" API — for plain-C codegen runtimes and language FFIs.
// Same semantics as the C++ class. Each handle owns one PrecisionController.
// ===========================================================================
#ifdef __cplusplus
extern "C" {
#endif

typedef struct lhsi_pc_handle lhsi_pc_handle_t;

typedef struct {
    uint32_t block_m;
    uint32_t block_n;
    uint32_t n;            // = block_m * block_n
    uint32_t score_width;  // bits per score
    uint32_t threshold;
    uint32_t log2_n;
    uint32_t sum_width;
    uint32_t cmp_width;
} lhsi_pc_info_t;

// Create / destroy
lhsi_pc_handle_t* lhsi_pc_create(void);
void              lhsi_pc_destroy(lhsi_pc_handle_t* h);

// State control
void              lhsi_pc_reset (lhsi_pc_handle_t* h);

// Read-only configuration
void              lhsi_pc_info  (const lhsi_pc_handle_t* h, lhsi_pc_info_t* out);

// Low-level streaming
void              lhsi_pc_tick  (lhsi_pc_handle_t* h,
                                 int s_valid, int32_t s_data, int s_last);
int               lhsi_pc_d_valid(const lhsi_pc_handle_t* h); // 0 or 1
int               lhsi_pc_d_fp16 (const lhsi_pc_handle_t* h); // 0 or 1
uint32_t          lhsi_pc_tiles  (const lhsi_pc_handle_t* h);

// High-level batched
int               lhsi_pc_process_tile(lhsi_pc_handle_t* h,
                                       const int32_t* scores, size_t n); // returns d_fp16

// Stateless one-shot (pure function — no handle needed)
int               lhsi_pc_decide(const int32_t* scores, size_t n);       // returns d_fp16

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LHSI_PRECISION_CONTROLLER_REF_HPP
