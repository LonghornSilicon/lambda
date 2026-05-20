#include "kv_cache_engine_ref.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace lhsi {

// -----------------------------------------------------------------------
// Lloyd-Max 3-bit centroids for N(0,1) — scaled by 1/sqrt(dim) at init
// -----------------------------------------------------------------------

static const double LLOYD_MAX_3BIT_CENTROIDS[] = {
    -2.1520, -1.3439, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3439,  2.1520,
};

static const double LLOYD_MAX_3BIT_BOUNDARIES[] = {
    -1.7480, -1.0500, -0.5006, 0.0,
     0.5006,  1.0500,  1.7480,
};

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

KVCacheEngine::KVCacheEngine(KVCacheEngineInfo info) : info_(info) {
    info_.validate();

    double sigma = 1.0 / std::sqrt(static_cast<double>(info_.vector_dim));

    centroids_.resize(8);
    for (int i = 0; i < 8; i++) {
        double val = LLOYD_MAX_3BIT_CENTROIDS[i] * sigma;
        int32_t fixed = static_cast<int32_t>(
            std::round(val * (1 << info_.coord_frac)));
        centroids_[i] = static_cast<int16_t>(
            to_signed(fixed, info_.coord_width));
    }

    boundaries_.resize(7);
    for (int i = 0; i < 7; i++) {
        double val = LLOYD_MAX_3BIT_BOUNDARIES[i] * sigma;
        int32_t fixed = static_cast<int32_t>(
            std::round(val * (1 << info_.coord_frac)));
        boundaries_[i] = static_cast<int16_t>(
            to_signed(fixed, info_.coord_width));
    }

    sign_flips_.resize(info_.vector_dim);
    LCG rng(info_.rotation_seed);
    for (uint32_t i = 0; i < info_.vector_dim; i++) {
        uint32_t bits = rng.next();
        sign_flips_[i] = (bits & 1) == 0 ? 1 : -1;
    }

    qjl_matrix_.resize(info_.vector_dim);
    LCG rng2(info_.rotation_seed + 0xDEADBEEF);
    for (uint32_t i = 0; i < info_.vector_dim; i++) {
        qjl_matrix_[i].resize(info_.vector_dim);
        for (uint32_t j = 0; j < info_.vector_dim; j++) {
            uint32_t bits = rng2.next();
            qjl_matrix_[i][j] = (bits & 1) == 0 ? 1 : -1;
        }
    }

    double sqrt_pi_2 = std::sqrt(M_PI / 2.0);
    sqrt_pi_2_ = to_signed(
        static_cast<int32_t>(std::round(sqrt_pi_2 * (1 << info_.coord_frac))),
        info_.coord_width);
}

void KVCacheEngine::reset() {
    sram_k_.clear();
    sram_v_.clear();
    occupancy_ = 0;
    tiles_compressed_ = 0;
    tiles_decompressed_ = 0;
}

// -----------------------------------------------------------------------
// WHT (in-place, matching Python _wht_inplace)
// -----------------------------------------------------------------------

void KVCacheEngine::wht_inplace(int32_t* vec, size_t n, int width) const {
    size_t h = 1;
    while (h < n) {
        for (size_t i = 0; i < n; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                int32_t a = vec[j];
                int32_t b = vec[j + h];
                vec[j]     = to_signed(static_cast<int64_t>(a) + b, width + 1);
                vec[j + h] = to_signed(static_cast<int64_t>(a) - b, width + 1);
            }
        }
        h *= 2;
    }
}

// -----------------------------------------------------------------------
// Sub-operations
// -----------------------------------------------------------------------

uint32_t KVCacheEngine::compute_norm(const int16_t* vec, size_t dim) const {
    int64_t sum_sq = 0;
    for (size_t i = 0; i < dim; i++) {
        int32_t sv = to_signed(vec[i], info_.coord_width);
        sum_sq += static_cast<int64_t>(sv) * sv;
    }

    int sq_width = 2 * static_cast<int>(info_.coord_width) +
                   static_cast<int>(info_.log2_n());
    uint32_t norm = isqrt(static_cast<uint64_t>(sum_sq), sq_width / 2);

    int shift = static_cast<int>(info_.coord_frac) -
                static_cast<int>(info_.norm_frac);
    if (shift > 0)
        norm = (norm + (1u << (shift - 1))) >> shift;
    else if (shift < 0)
        norm <<= (-shift);
    return to_unsigned(norm, info_.norm_width);
}

void KVCacheEngine::normalize(const int16_t* vec, uint32_t norm,
                               int16_t* out, size_t dim) const {
    if (norm == 0) {
        std::memset(out, 0, dim * sizeof(int16_t));
        return;
    }
    for (size_t i = 0; i < dim; i++) {
        int32_t sv = to_signed(vec[i], info_.coord_width);
        int64_t numerator = static_cast<int64_t>(sv) << info_.norm_frac;
        int32_t normalized = static_cast<int32_t>(
            floor_div(numerator, static_cast<int64_t>(norm)));
        out[i] = static_cast<int16_t>(
            to_signed(normalized, info_.coord_width));
    }
}

void KVCacheEngine::rotate(const int16_t* vec, int16_t* out,
                            size_t dim) const {
    int log2n = static_cast<int>(info_.log2_n());
    int wht_width = static_cast<int>(info_.coord_width) + log2n;

    std::vector<int32_t> work(dim);
    for (size_t i = 0; i < dim; i++) {
        work[i] = to_signed(
            static_cast<int32_t>(to_signed(vec[i], info_.coord_width)) *
            sign_flips_[i], info_.coord_width);
    }

    for (size_t i = 0; i < dim; i++)
        work[i] = to_signed(work[i], wht_width);

    wht_inplace(work.data(), dim, wht_width);

    int shift = log2n / 2;
    for (size_t i = 0; i < dim; i++) {
        int32_t shifted = shift > 0 ?
            (work[i] + (1 << (shift - 1))) >> shift : work[i];
        out[i] = static_cast<int16_t>(
            to_signed(shifted, info_.coord_width));
    }
}

void KVCacheEngine::inverse_rotate(const int16_t* vec, int16_t* out,
                                    size_t dim) const {
    int log2n = static_cast<int>(info_.log2_n());
    int wht_width = static_cast<int>(info_.coord_width) + log2n;

    std::vector<int32_t> work(dim);
    for (size_t i = 0; i < dim; i++)
        work[i] = to_signed(vec[i], wht_width);

    wht_inplace(work.data(), dim, wht_width);

    int shift = log2n - (log2n / 2);
    for (size_t i = 0; i < dim; i++) {
        int32_t shifted = shift > 0 ?
            (work[i] + (1 << (shift - 1))) >> shift : work[i];
        int32_t val = to_signed(
            static_cast<int64_t>(shifted) * sign_flips_[i],
            info_.coord_width);
        out[i] = static_cast<int16_t>(val);
    }
}

void KVCacheEngine::quantize_pq(const int16_t* rotated, uint8_t* indices,
                                 size_t dim) const {
    for (size_t i = 0; i < dim; i++) {
        int16_t coord_s = to_signed(rotated[i], info_.coord_width);
        uint8_t idx = 0;
        for (size_t b = 0; b < boundaries_.size(); b++) {
            if (coord_s >= boundaries_[b])
                idx = static_cast<uint8_t>(b + 1);
        }
        indices[i] = idx;
    }
}

void KVCacheEngine::dequantize_pq(const uint8_t* indices, int16_t* out,
                                   size_t dim) const {
    for (size_t i = 0; i < dim; i++)
        out[i] = centroids_[indices[i]];
}

void KVCacheEngine::qjl_compress(const int16_t* residual, uint8_t* signs,
                                  uint32_t* res_norm, size_t dim) const {
    *res_norm = compute_norm(residual, dim);

    for (size_t i = 0; i < dim; i++) {
        int64_t dot = 0;
        for (size_t j = 0; j < dim; j++) {
            int32_t r_j = to_signed(residual[j], info_.coord_width);
            dot += static_cast<int64_t>(qjl_matrix_[i][j]) * r_j;
        }
        signs[i] = dot >= 0 ? 1 : 0;
    }
}

void KVCacheEngine::qjl_decompress(const uint8_t* signs, uint32_t res_norm,
                                    int16_t* correction, size_t dim) const {
    int32_t scale_num = fixed_mul(
        sqrt_pi_2_, static_cast<int32_t>(res_norm),
        info_.coord_frac, info_.norm_frac,
        info_.coord_width + info_.norm_width, info_.coord_frac);
    int32_t scale = info_.vector_dim > 0 ?
        static_cast<int32_t>(floor_div(
            static_cast<int64_t>(scale_num),
            static_cast<int64_t>(info_.vector_dim))) : 0;

    for (size_t j = 0; j < dim; j++) {
        int64_t acc = 0;
        for (size_t i = 0; i < dim; i++) {
            int32_t sign_val = signs[i] ? 1 : -1;
            acc += static_cast<int64_t>(qjl_matrix_[i][j]) * sign_val;
        }
        int32_t c = fixed_mul(static_cast<int32_t>(acc), scale,
                              0, info_.coord_frac,
                              info_.coord_width, info_.coord_frac);
        correction[j] = static_cast<int16_t>(
            to_signed(c, info_.coord_width));
    }
}

// -----------------------------------------------------------------------
// Batch API
// -----------------------------------------------------------------------

CompressedKey KVCacheEngine::compress_key(const int16_t* vec, size_t dim) {
    assert(dim == info_.vector_dim);
    CompressedKey ck;

    ck.norm = compute_norm(vec, dim);

    std::vector<int16_t> normalized(dim);
    normalize(vec, ck.norm, normalized.data(), dim);

    std::vector<int16_t> rotated(dim);
    rotate(normalized.data(), rotated.data(), dim);

    ck.indices.resize(dim);
    quantize_pq(rotated.data(), ck.indices.data(), dim);

    std::vector<int16_t> dequantized(dim);
    dequantize_pq(ck.indices.data(), dequantized.data(), dim);

    std::vector<int16_t> residual(dim);
    for (size_t i = 0; i < dim; i++) {
        residual[i] = static_cast<int16_t>(to_signed(
            static_cast<int64_t>(rotated[i]) - dequantized[i],
            info_.coord_width));
    }

    ck.signs.resize(dim);
    qjl_compress(residual.data(), ck.signs.data(), &ck.residual_norm, dim);

    tiles_compressed_++;
    return ck;
}

CompressedValue KVCacheEngine::compress_value(const int16_t* vec, size_t dim) {
    assert(dim == info_.vector_dim);
    CompressedValue cv;

    cv.norm = compute_norm(vec, dim);

    std::vector<int16_t> normalized(dim);
    normalize(vec, cv.norm, normalized.data(), dim);

    std::vector<int16_t> rotated(dim);
    rotate(normalized.data(), rotated.data(), dim);

    cv.indices.resize(dim);
    quantize_pq(rotated.data(), cv.indices.data(), dim);

    tiles_compressed_++;
    return cv;
}

void KVCacheEngine::decompress_key(const CompressedKey& ck,
                                    int16_t* out) const {
    size_t dim = info_.vector_dim;

    std::vector<int16_t> dequantized(dim);
    dequantize_pq(ck.indices.data(), dequantized.data(), dim);

    std::vector<int16_t> qjl_correction(dim);
    qjl_decompress(ck.signs.data(), ck.residual_norm,
                   qjl_correction.data(), dim);

    std::vector<int16_t> corrected(dim);
    for (size_t i = 0; i < dim; i++) {
        corrected[i] = static_cast<int16_t>(to_signed(
            static_cast<int64_t>(dequantized[i]) + qjl_correction[i],
            info_.coord_width));
    }

    std::vector<int16_t> unrotated(dim);
    inverse_rotate(corrected.data(), unrotated.data(), dim);

    for (size_t i = 0; i < dim; i++) {
        int32_t sv = to_signed(unrotated[i], info_.coord_width);
        int32_t rescaled = fixed_mul(sv, static_cast<int32_t>(ck.norm),
                                     info_.coord_frac, info_.norm_frac,
                                     info_.coord_width, info_.coord_frac);
        out[i] = static_cast<int16_t>(
            to_signed(rescaled, info_.coord_width));
    }
}

void KVCacheEngine::decompress_value(const CompressedValue& cv,
                                      int16_t* out) const {
    size_t dim = info_.vector_dim;

    std::vector<int16_t> dequantized(dim);
    dequantize_pq(cv.indices.data(), dequantized.data(), dim);

    std::vector<int16_t> unrotated(dim);
    inverse_rotate(dequantized.data(), unrotated.data(), dim);

    for (size_t i = 0; i < dim; i++) {
        int32_t sv = to_signed(unrotated[i], info_.coord_width);
        int32_t rescaled = fixed_mul(sv, static_cast<int32_t>(cv.norm),
                                     info_.coord_frac, info_.norm_frac,
                                     info_.coord_width, info_.coord_frac);
        out[i] = static_cast<int16_t>(
            to_signed(rescaled, info_.coord_width));
    }
}

// Convenience overloads

CompressedKey KVCacheEngine::compress_key(const std::vector<int16_t>& vec) {
    return compress_key(vec.data(), vec.size());
}

CompressedValue KVCacheEngine::compress_value(const std::vector<int16_t>& vec) {
    return compress_value(vec.data(), vec.size());
}

std::vector<int16_t> KVCacheEngine::decompress_key(
    const CompressedKey& ck) const {
    std::vector<int16_t> out(info_.vector_dim);
    decompress_key(ck, out.data());
    return out;
}

std::vector<int16_t> KVCacheEngine::decompress_value(
    const CompressedValue& cv) const {
    std::vector<int16_t> out(info_.vector_dim);
    decompress_value(cv, out.data());
    return out;
}

// Store/load

void KVCacheEngine::store_kv(uint32_t addr, const int16_t* key,
                              const int16_t* val, size_t dim) {
    assert(addr < info_.sram_depth);
    sram_k_[addr] = compress_key(key, dim);
    sram_v_[addr] = compress_value(val, dim);
    if (addr >= occupancy_)
        occupancy_ = addr + 1;
}

void KVCacheEngine::load_k(uint32_t addr, int16_t* out, size_t /*dim*/) const {
    auto it = sram_k_.find(addr);
    assert(it != sram_k_.end());
    decompress_key(it->second, out);
    const_cast<KVCacheEngine*>(this)->tiles_decompressed_++;
}

void KVCacheEngine::load_v(uint32_t addr, int16_t* out, size_t /*dim*/) const {
    auto it = sram_v_.find(addr);
    assert(it != sram_v_.end());
    decompress_value(it->second, out);
    const_cast<KVCacheEngine*>(this)->tiles_decompressed_++;
}

// Stateless

CompressedKey KVCacheEngine::decide_compress_key(
    const int16_t* vec, size_t dim, const KVCacheEngineInfo& info) {
    KVCacheEngine engine(info);
    return engine.compress_key(vec, dim);
}

} // namespace lhsi

// -----------------------------------------------------------------------
// extern "C" implementation
// -----------------------------------------------------------------------

struct lhsi_kv_handle {
    lhsi::KVCacheEngine engine;
    explicit lhsi_kv_handle(lhsi::KVCacheEngineInfo info)
        : engine(info) {}
};

static lhsi::KVCacheEngineInfo from_c_info(const lhsi_kv_info_t* ci) {
    lhsi::KVCacheEngineInfo info;
    if (ci) {
        info.vector_dim    = ci->vector_dim;
        info.num_centroids = ci->num_centroids;
        info.pq_bits       = ci->pq_bits;
        info.qjl_bits      = ci->qjl_bits;
        info.sram_depth    = ci->sram_depth;
        info.norm_width    = ci->norm_width;
        info.norm_frac     = ci->norm_frac;
        info.coord_width   = ci->coord_width;
        info.coord_frac    = ci->coord_frac;
        info.rotation_seed = ci->rotation_seed;
    }
    return info;
}

static lhsi_kv_info_t to_c_info(const lhsi::KVCacheEngineInfo& info) {
    lhsi_kv_info_t ci;
    ci.vector_dim          = info.vector_dim;
    ci.num_centroids       = info.num_centroids;
    ci.pq_bits             = info.pq_bits;
    ci.qjl_bits            = info.qjl_bits;
    ci.sram_depth          = info.sram_depth;
    ci.norm_width          = info.norm_width;
    ci.norm_frac           = info.norm_frac;
    ci.coord_width         = info.coord_width;
    ci.coord_frac          = info.coord_frac;
    ci.rotation_seed       = info.rotation_seed;
    ci.compressed_key_bits = info.compressed_key_bits();
    ci.compressed_val_bits = info.compressed_val_bits();
    ci.uncompressed_bits   = info.uncompressed_bits();
    return ci;
}

lhsi_kv_handle_t* lhsi_kv_create(const lhsi_kv_info_t* info) {
    return new lhsi_kv_handle(from_c_info(info));
}

lhsi_kv_handle_t* lhsi_kv_create_default(void) {
    return new lhsi_kv_handle(lhsi::KVCacheEngineInfo{});
}

void lhsi_kv_destroy(lhsi_kv_handle_t* h) { delete h; }

void lhsi_kv_reset(lhsi_kv_handle_t* h) { h->engine.reset(); }

lhsi_kv_info_t lhsi_kv_info(const lhsi_kv_handle_t* h) {
    return to_c_info(h->engine.info());
}

void lhsi_kv_compress_key(lhsi_kv_handle_t* h,
                          const int16_t* vec, size_t dim,
                          uint32_t* norm, uint8_t* indices,
                          uint32_t* res_norm, uint8_t* signs) {
    auto ck = h->engine.compress_key(vec, dim);
    *norm = ck.norm;
    std::memcpy(indices, ck.indices.data(), ck.indices.size());
    *res_norm = ck.residual_norm;
    std::memcpy(signs, ck.signs.data(), ck.signs.size());
}

void lhsi_kv_compress_value(lhsi_kv_handle_t* h,
                            const int16_t* vec, size_t dim,
                            uint32_t* norm, uint8_t* indices) {
    auto cv = h->engine.compress_value(vec, dim);
    *norm = cv.norm;
    std::memcpy(indices, cv.indices.data(), cv.indices.size());
}

void lhsi_kv_decompress_key(const lhsi_kv_handle_t* h,
                            uint32_t norm, const uint8_t* indices,
                            uint32_t res_norm, const uint8_t* signs,
                            size_t dim, int16_t* out) {
    lhsi::CompressedKey ck;
    ck.norm = norm;
    ck.indices.assign(indices, indices + dim);
    ck.residual_norm = res_norm;
    ck.signs.assign(signs, signs + dim);
    h->engine.decompress_key(ck, out);
}

void lhsi_kv_decompress_value(const lhsi_kv_handle_t* h,
                              uint32_t norm, const uint8_t* indices,
                              size_t dim, int16_t* out) {
    lhsi::CompressedValue cv;
    cv.norm = norm;
    cv.indices.assign(indices, indices + dim);
    h->engine.decompress_value(cv, out);
}

void lhsi_kv_store(lhsi_kv_handle_t* h, uint32_t addr,
                   const int16_t* key, const int16_t* val, size_t dim) {
    h->engine.store_kv(addr, key, val, dim);
}

void lhsi_kv_load_k(const lhsi_kv_handle_t* h, uint32_t addr,
                    int16_t* out, size_t dim) {
    const_cast<lhsi_kv_handle_t*>(h)->engine.load_k(addr, out, dim);
}

void lhsi_kv_load_v(const lhsi_kv_handle_t* h, uint32_t addr,
                    int16_t* out, size_t dim) {
    const_cast<lhsi_kv_handle_t*>(h)->engine.load_v(addr, out, dim);
}

uint32_t lhsi_kv_occupancy(const lhsi_kv_handle_t* h) {
    return h->engine.occupancy();
}
