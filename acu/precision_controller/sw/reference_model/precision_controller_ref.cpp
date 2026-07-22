// precision_controller_ref.cpp — implementation of the bit-accurate
// C++ reference model. Mirrors precision_controller_ref.py exactly.
//
// All arithmetic is unsigned fixed-width modular arithmetic. The mask
// helpers and the abs-of-two's-complement circuit are direct translations
// of the Python helpers.

#include "precision_controller_ref.hpp"

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

inline std::uint64_t mask_n(std::uint32_t width) {
    // (1 << width) - 1, but safe for width == 64.
    if (width >= 64) return ~static_cast<std::uint64_t>(0);
    return (static_cast<std::uint64_t>(1) << width) - 1;
}

// Replicates the SV `s_data[W-1] ? (~s_data + 1) : s_data` ABS circuit,
// constrained to `width` bits. abs(-2^(W-1)) returns 2^(W-1) under
// modular arithmetic — same as the hardware.
inline std::uint32_t abs_two_s_complement(std::int32_t value, std::uint32_t width) {
    const std::uint64_t m = mask_n(width);
    if (value < 0) {
        return static_cast<std::uint32_t>((static_cast<std::uint64_t>((~value) + 1)) & m);
    }
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(value) & m);
}

inline std::uint32_t runtime_log2_n(std::uint32_t n) {
    std::uint32_t v = n;
    std::uint32_t l = 0;
    while (v >>= 1) ++l;
    return l;
}

} // anonymous namespace

namespace lhsi {

// ---------------------------------------------------------------------------
// PrecisionControllerInfo helpers
// ---------------------------------------------------------------------------
std::uint32_t PrecisionControllerInfo::log2_n() const {
    return runtime_log2_n(n());
}

void PrecisionControllerInfo::validate() const {
    if (threshold != 10) {
        throw std::invalid_argument(
            "PrecisionControllerInfo: only THRESHOLD=10 is supported in "
            "this reference; other thresholds require a parameterized RTL build.");
    }
    const std::uint32_t v = n();
    if (v == 0 || (v & (v - 1)) != 0) {
        throw std::invalid_argument(
            "PrecisionControllerInfo: N = BLOCK_M * BLOCK_N must be a power of two.");
    }
}

// ---------------------------------------------------------------------------
// PrecisionController
// ---------------------------------------------------------------------------
PrecisionController::PrecisionController(PrecisionControllerInfo info)
    : info_(info)
{
    info_.validate();
    reset();
}

void PrecisionController::reset() {
    max_acc_         = 0;
    sum_acc_         = 0;
    d_valid_         = false;
    d_fp16_          = false;
    tiles_processed_ = 0;
    decision_history_.clear();
}

void PrecisionController::tick(bool s_valid, std::int32_t s_data, bool s_last) {
    const std::uint32_t score_w  = info_.score_width;
    const std::uint64_t score_m  = mask_n(score_w);
    const std::uint64_t sum_m    = mask_n(info_.sum_width());
    const std::uint32_t log2_n   = info_.log2_n();
    const std::uint64_t cmp_m    = mask_n(info_.cmp_width());

    // d_valid pulses one cycle per decision; default to deasserted.
    d_valid_ = false;

    if (!s_valid) return;

    // Mask s_data into a SCORE_WIDTH two's-complement view, then sign-extend.
    const std::uint64_t masked = static_cast<std::uint64_t>(s_data) & score_m;
    std::int32_t signed_v;
    if (masked & (static_cast<std::uint64_t>(1) << (score_w - 1))) {
        signed_v = static_cast<std::int32_t>(masked) -
                   (static_cast<std::int32_t>(1) << score_w);
    } else {
        signed_v = static_cast<std::int32_t>(masked);
    }
    const std::uint32_t abs_score = abs_two_s_complement(signed_v, score_w);

    // Combinational next-state. The new score participates in this cycle's
    // decision when s_last asserts (matches the SV).
    const std::uint32_t max_next =
        (abs_score > max_acc_) ? abs_score : max_acc_;
    const std::uint64_t sum_next =
        (sum_acc_ + static_cast<std::uint64_t>(abs_score)) & sum_m;

    if (s_last) {
        // LHS = max_next * N   (= shift left by log2(N))
        // RHS = sum_next * 10  (= (sum<<3) + (sum<<1))
        const std::uint64_t lhs =
            (static_cast<std::uint64_t>(max_next) << log2_n) & cmp_m;
        const std::uint64_t rhs =
            ((sum_next << 3) + (sum_next << 1)) & cmp_m;

        d_fp16_  = (lhs > rhs);
        d_valid_ = true;
        decision_history_.push_back(d_fp16_);

        // Accumulators reset on s_last (last-write-wins, same as SV).
        max_acc_ = 0;
        sum_acc_ = 0;
        ++tiles_processed_;
    } else {
        max_acc_ = max_next;
        sum_acc_ = sum_next;
    }
}

bool PrecisionController::process_tile(const std::int32_t* scores, std::size_t n) {
    if (n != info_.n()) {
        throw std::invalid_argument("process_tile: scores length must equal N");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const bool last = (i == n - 1);
        tick(true, scores[i], last);
    }
    return d_fp16_;
}

bool PrecisionController::process_tile(const std::vector<std::int32_t>& scores) {
    return process_tile(scores.data(), scores.size());
}

bool PrecisionController::decide(const std::int32_t* scores, std::size_t n,
                                 PrecisionControllerInfo info) {
    info.validate();
    if (n != info.n()) {
        throw std::invalid_argument("decide: scores length must equal N");
    }
    const std::uint32_t score_w = info.score_width;
    const std::uint64_t score_m = mask_n(score_w);
    const std::uint64_t sum_m   = mask_n(info.sum_width());
    const std::uint32_t log2n   = info.log2_n();
    const std::uint64_t cmp_m   = mask_n(info.cmp_width());

    std::uint32_t max_v = 0;
    std::uint64_t sum_v = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t masked =
            static_cast<std::uint64_t>(scores[i]) & score_m;
        std::int32_t signed_v;
        if (masked & (static_cast<std::uint64_t>(1) << (score_w - 1))) {
            signed_v = static_cast<std::int32_t>(masked) -
                       (static_cast<std::int32_t>(1) << score_w);
        } else {
            signed_v = static_cast<std::int32_t>(masked);
        }
        const std::uint32_t a = abs_two_s_complement(signed_v, score_w);
        if (a > max_v) max_v = a;
        sum_v = (sum_v + a) & sum_m;
    }

    const std::uint64_t lhs =
        (static_cast<std::uint64_t>(max_v) << log2n) & cmp_m;
    const std::uint64_t rhs =
        ((sum_v << 3) + (sum_v << 1)) & cmp_m;
    return lhs > rhs;
}

} // namespace lhsi

// ===========================================================================
// extern "C" wrappers — own one lhsi::PrecisionController per handle.
// ===========================================================================
struct lhsi_pc_handle {
    lhsi::PrecisionController pc;
};

extern "C" {

lhsi_pc_handle_t* lhsi_pc_create(void) {
    return new lhsi_pc_handle{ lhsi::PrecisionController() };
}

void lhsi_pc_destroy(lhsi_pc_handle_t* h) {
    delete h;
}

void lhsi_pc_reset(lhsi_pc_handle_t* h) {
    h->pc.reset();
}

void lhsi_pc_info(const lhsi_pc_handle_t* h, lhsi_pc_info_t* out) {
    const auto& info = h->pc.info();
    out->block_m     = info.block_m;
    out->block_n     = info.block_n;
    out->n           = info.n();
    out->score_width = info.score_width;
    out->threshold   = info.threshold;
    out->log2_n      = info.log2_n();
    out->sum_width   = info.sum_width();
    out->cmp_width   = info.cmp_width();
}

void lhsi_pc_tick(lhsi_pc_handle_t* h, int s_valid, int32_t s_data, int s_last) {
    h->pc.tick(s_valid != 0, s_data, s_last != 0);
}

int lhsi_pc_d_valid(const lhsi_pc_handle_t* h) {
    return h->pc.d_valid() ? 1 : 0;
}

int lhsi_pc_d_fp16(const lhsi_pc_handle_t* h) {
    return h->pc.d_fp16() ? 1 : 0;
}

uint32_t lhsi_pc_tiles(const lhsi_pc_handle_t* h) {
    return h->pc.tiles_processed();
}

int lhsi_pc_process_tile(lhsi_pc_handle_t* h, const int32_t* scores, size_t n) {
    return h->pc.process_tile(scores, n) ? 1 : 0;
}

int lhsi_pc_decide(const int32_t* scores, size_t n) {
    return lhsi::PrecisionController::decide(scores, n) ? 1 : 0;
}

} // extern "C"
