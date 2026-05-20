#ifndef KV_CACHE_ENGINE_REF_HPP
#define KV_CACHE_ENGINE_REF_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <map>
#include <cassert>

namespace lhsi {

// -----------------------------------------------------------------------
// Fixed-point helpers
// -----------------------------------------------------------------------

inline uint64_t mask(int width) { return (1ULL << width) - 1; }

inline int32_t to_signed(int64_t val, int width) {
    val &= mask(width);
    if (val >= (1LL << (width - 1)))
        val -= (1LL << width);
    return static_cast<int32_t>(val);
}

inline uint32_t to_unsigned(int64_t val, int width) {
    return static_cast<uint32_t>(val & mask(width));
}

inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((a ^ b) < 0))
        q--;
    return q;
}

inline int32_t fixed_mul(int32_t a, int32_t b, int a_frac, int b_frac,
                         int out_width, int out_frac) {
    int64_t product = static_cast<int64_t>(a) * b;
    int shift = a_frac + b_frac - out_frac;
    if (shift > 0)
        product = (product + (1LL << (shift - 1))) >> shift;
    else if (shift < 0)
        product <<= (-shift);
    return to_signed(product, out_width);
}

inline uint32_t isqrt(uint64_t val, int result_width) {
    if (val == 0) return 0;
    uint64_t x = 0;
    uint64_t bit = 1ULL << (result_width - 1);
    while (bit > 0) {
        uint64_t trial = x + bit;
        if (trial * trial <= val)
            x = trial;
        bit >>= 1;
    }
    return static_cast<uint32_t>(x);
}

// -----------------------------------------------------------------------
// PRNG (LCG matching Python reference)
// -----------------------------------------------------------------------

struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}
    uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 32);
    }
};

// -----------------------------------------------------------------------
// Compressed data containers
// -----------------------------------------------------------------------

struct CompressedKey {
    uint32_t norm;
    std::vector<uint8_t> indices;
    uint32_t residual_norm;
    std::vector<uint8_t> signs;
};

struct CompressedValue {
    uint32_t norm;
    std::vector<uint8_t> indices;
};

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------

struct KVCacheEngineInfo {
    uint32_t vector_dim     = 64;
    uint32_t num_centroids  = 8;
    uint32_t pq_bits        = 3;
    uint32_t qjl_bits       = 1;
    uint32_t sram_depth     = 1024;
    uint32_t norm_width     = 16;
    uint32_t norm_frac      = 8;
    uint32_t coord_width    = 16;
    uint32_t coord_frac     = 12;
    uint32_t rotation_seed  = 42;

    uint32_t n() const { return vector_dim; }

    uint32_t log2_n() const {
        uint32_t v = vector_dim, r = 0;
        while (v > 1) { v >>= 1; r++; }
        return r;
    }

    uint32_t compressed_key_bits() const {
        return norm_width + vector_dim * pq_bits +
               norm_width + vector_dim * qjl_bits;
    }

    uint32_t compressed_val_bits() const {
        return norm_width + vector_dim * pq_bits;
    }

    uint32_t uncompressed_bits() const {
        return vector_dim * coord_width;
    }

    double compression_ratio_k() const {
        return static_cast<double>(uncompressed_bits()) / compressed_key_bits();
    }

    double compression_ratio_v() const {
        return static_cast<double>(uncompressed_bits()) / compressed_val_bits();
    }

    void validate() const {
        assert(vector_dim > 0 && (vector_dim & (vector_dim - 1)) == 0);
        assert(num_centroids == (1u << pq_bits));
        assert(pq_bits == 3);
        assert(qjl_bits == 1);
        assert(norm_width >= 8);
        assert(coord_width >= 8);
    }
};

// -----------------------------------------------------------------------
// KV Cache Engine reference model
// -----------------------------------------------------------------------

class KVCacheEngine {
public:
    explicit KVCacheEngine(KVCacheEngineInfo info = {});

    const KVCacheEngineInfo& info() const noexcept { return info_; }

    void reset();

    // Sub-operations (match RTL pipeline stages)
    uint32_t compute_norm(const int16_t* vec, size_t dim) const;
    void normalize(const int16_t* vec, uint32_t norm,
                   int16_t* out, size_t dim) const;
    void rotate(const int16_t* vec, int16_t* out, size_t dim) const;
    void inverse_rotate(const int16_t* vec, int16_t* out, size_t dim) const;
    void quantize_pq(const int16_t* rotated, uint8_t* indices,
                     size_t dim) const;
    void dequantize_pq(const uint8_t* indices, int16_t* out,
                       size_t dim) const;
    void qjl_compress(const int16_t* residual, uint8_t* signs,
                      uint32_t* res_norm, size_t dim) const;
    void qjl_decompress(const uint8_t* signs, uint32_t res_norm,
                        int16_t* correction, size_t dim) const;

    // Batch API
    CompressedKey   compress_key(const int16_t* vec, size_t dim);
    CompressedValue compress_value(const int16_t* vec, size_t dim);
    void decompress_key(const CompressedKey& ck, int16_t* out) const;
    void decompress_value(const CompressedValue& cv, int16_t* out) const;

    // Convenience overloads
    CompressedKey   compress_key(const std::vector<int16_t>& vec);
    CompressedValue compress_value(const std::vector<int16_t>& vec);
    std::vector<int16_t> decompress_key(const CompressedKey& ck) const;
    std::vector<int16_t> decompress_value(const CompressedValue& cv) const;

    // Store/load
    void store_kv(uint32_t addr, const int16_t* key, const int16_t* val,
                  size_t dim);
    void load_k(uint32_t addr, int16_t* out, size_t dim) const;
    void load_v(uint32_t addr, int16_t* out, size_t dim) const;

    // Stateless
    static CompressedKey decide_compress_key(
        const int16_t* vec, size_t dim,
        const KVCacheEngineInfo& info = {});

    uint32_t occupancy() const { return occupancy_; }
    uint32_t tiles_compressed() const { return tiles_compressed_; }
    uint32_t tiles_decompressed() const { return tiles_decompressed_; }

private:
    KVCacheEngineInfo info_;
    std::vector<int16_t> centroids_;
    std::vector<int16_t> boundaries_;
    std::vector<int8_t>  sign_flips_;
    std::vector<std::vector<int8_t>> qjl_matrix_;
    int32_t sqrt_pi_2_;

    std::map<uint32_t, CompressedKey>   sram_k_;
    std::map<uint32_t, CompressedValue> sram_v_;
    uint32_t occupancy_          = 0;
    uint32_t tiles_compressed_   = 0;
    uint32_t tiles_decompressed_ = 0;

    void wht_inplace(int32_t* vec, size_t n, int width) const;
};

} // namespace lhsi

// -----------------------------------------------------------------------
// extern "C" API
// -----------------------------------------------------------------------

extern "C" {

typedef struct lhsi_kv_handle lhsi_kv_handle_t;

struct lhsi_kv_info_t {
    uint32_t vector_dim;
    uint32_t num_centroids;
    uint32_t pq_bits;
    uint32_t qjl_bits;
    uint32_t sram_depth;
    uint32_t norm_width;
    uint32_t norm_frac;
    uint32_t coord_width;
    uint32_t coord_frac;
    uint32_t rotation_seed;
    uint32_t compressed_key_bits;
    uint32_t compressed_val_bits;
    uint32_t uncompressed_bits;
};

lhsi_kv_handle_t* lhsi_kv_create(const lhsi_kv_info_t* info);
lhsi_kv_handle_t* lhsi_kv_create_default(void);
void              lhsi_kv_destroy(lhsi_kv_handle_t* h);
void              lhsi_kv_reset(lhsi_kv_handle_t* h);
lhsi_kv_info_t    lhsi_kv_info(const lhsi_kv_handle_t* h);

void lhsi_kv_compress_key(lhsi_kv_handle_t* h,
                          const int16_t* vec, size_t dim,
                          uint32_t* norm, uint8_t* indices,
                          uint32_t* res_norm, uint8_t* signs);

void lhsi_kv_compress_value(lhsi_kv_handle_t* h,
                            const int16_t* vec, size_t dim,
                            uint32_t* norm, uint8_t* indices);

void lhsi_kv_decompress_key(const lhsi_kv_handle_t* h,
                            uint32_t norm, const uint8_t* indices,
                            uint32_t res_norm, const uint8_t* signs,
                            size_t dim, int16_t* out);

void lhsi_kv_decompress_value(const lhsi_kv_handle_t* h,
                              uint32_t norm, const uint8_t* indices,
                              size_t dim, int16_t* out);

void lhsi_kv_store(lhsi_kv_handle_t* h, uint32_t addr,
                   const int16_t* key, const int16_t* val, size_t dim);

void lhsi_kv_load_k(const lhsi_kv_handle_t* h, uint32_t addr,
                    int16_t* out, size_t dim);

void lhsi_kv_load_v(const lhsi_kv_handle_t* h, uint32_t addr,
                    int16_t* out, size_t dim);

uint32_t lhsi_kv_occupancy(const lhsi_kv_handle_t* h);

} // extern "C"

#endif // KV_CACHE_ENGINE_REF_HPP
