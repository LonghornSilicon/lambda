#include "kv_cache_engine_ref.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace lhsi;

static int g_passed = 0;
static int g_total  = 0;
static int g_errors = 0;

#define CHECK(cond, ...) do { \
    g_total++; \
    if (cond) { g_passed++; } \
    else { g_errors++; fprintf(stderr, "  FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } \
} while(0)

// -----------------------------------------------------------------------
// PRNG for test vectors (matches Python _random_vector)
// -----------------------------------------------------------------------

static void random_vector(int16_t* out, size_t dim, uint64_t& state,
                          int width) {
    int32_t max_val = (1 << (width - 1)) - 1;
    int32_t scale = max_val / 8;
    for (size_t i = 0; i < dim; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t raw = static_cast<uint32_t>((state >> 33) & mask(16));
        int32_t val = (to_signed(raw, 16) * scale) >> 15;
        out[i] = static_cast<int16_t>(to_signed(val, width));
    }
}

// -----------------------------------------------------------------------
// Hex file reading (matches Python read_hex)
// -----------------------------------------------------------------------

static std::vector<int16_t> read_hex_file(const std::string& path,
                                           int width) {
    std::vector<int16_t> values;
    std::ifstream f(path);
    if (!f.is_open()) return values;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '/') continue;
        uint32_t val = static_cast<uint32_t>(strtoul(line.c_str(), nullptr, 16));
        values.push_back(static_cast<int16_t>(to_signed(val, width)));
    }
    return values;
}

// -----------------------------------------------------------------------
// Test 1: Canonical boundary cases
// -----------------------------------------------------------------------

static void test_canonical_boundary_cases() {
    printf("\n[1. Canonical boundary cases]\n");
    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;
    int16_t max_pos = static_cast<int16_t>((1 << (info.coord_width - 1)) - 1);

    // Zero vector
    std::vector<int16_t> zero(dim, 0);
    auto ck = engine.compress_key(zero);
    auto dk = engine.decompress_key(ck);
    bool all_zero = true;
    for (auto v : dk) if (v != 0) all_zero = false;
    CHECK(all_zero, "zero key round-trip");

    auto cv = engine.compress_value(zero);
    auto dv = engine.decompress_value(cv);
    all_zero = true;
    for (auto v : dv) if (v != 0) all_zero = false;
    CHECK(all_zero, "zero value round-trip");

    // Max positive
    std::vector<int16_t> maxvec(dim, max_pos);
    ck = engine.compress_key(maxvec);
    CHECK(ck.norm > 0, "max-pos norm > 0");
    CHECK(ck.indices.size() == dim, "max-pos indices size");
    CHECK(ck.signs.size() == dim, "max-pos signs size");

    // Single spike
    std::vector<int16_t> spike(dim, 0);
    spike[0] = max_pos;
    ck = engine.compress_key(spike);
    dk = engine.decompress_key(ck);
    CHECK(dk[0] != 0, "spike reconstructs non-zero at [0]");

    printf("  %d/%d passed\n", g_passed, g_total);
}

// -----------------------------------------------------------------------
// Test 2: Round-trip MSE
// -----------------------------------------------------------------------

static void test_round_trip_mse() {
    printf("\n[2. Round-trip MSE]\n");
    int before = g_passed;
    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;
    int32_t max_val = (1 << (info.coord_width - 1)) - 1;
    int64_t mse_threshold = static_cast<int64_t>(max_val) * max_val;

    uint64_t seed = 99999;
    for (int t = 0; t < 20; t++) {
        std::vector<int16_t> vec(dim);
        random_vector(vec.data(), dim, seed, info.coord_width);

        auto ck = engine.compress_key(vec);
        auto dk = engine.decompress_key(ck);
        int64_t mse_k = 0;
        for (size_t i = 0; i < dim; i++) {
            int32_t diff = to_signed(
                static_cast<int64_t>(vec[i]) - dk[i], info.coord_width);
            mse_k += static_cast<int64_t>(diff) * diff;
        }
        mse_k /= static_cast<int64_t>(dim);
        CHECK(mse_k < mse_threshold, "key MSE vec %d: %ld >= %ld",
              t, (long)mse_k, (long)mse_threshold);

        auto cv = engine.compress_value(vec);
        auto dv = engine.decompress_value(cv);
        int64_t mse_v = 0;
        for (size_t i = 0; i < dim; i++) {
            int32_t diff = to_signed(
                static_cast<int64_t>(vec[i]) - dv[i], info.coord_width);
            mse_v += static_cast<int64_t>(diff) * diff;
        }
        mse_v /= static_cast<int64_t>(dim);
        CHECK(mse_v < mse_threshold, "val MSE vec %d: %ld >= %ld",
              t, (long)mse_v, (long)mse_threshold);
    }
    printf("  %d/%d passed (this section)\n", g_passed - before,
           g_total - (g_total - (g_passed - before + (g_errors > 0 ? 1 : 0))));
}

// -----------------------------------------------------------------------
// Test 3: K/V asymmetry
// -----------------------------------------------------------------------

static void test_kv_asymmetry() {
    printf("\n[3. K/V asymmetry]\n");
    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;

    uint64_t seed = 42424242;
    std::vector<int16_t> vec(dim);
    random_vector(vec.data(), dim, seed, info.coord_width);

    auto ck = engine.compress_key(vec);
    auto cv = engine.compress_value(vec);

    CHECK(ck.signs.size() == dim, "K has QJL signs");
    CHECK(ck.norm == cv.norm, "K and V norms match");
    CHECK(ck.indices == cv.indices, "K and V PQ indices match");
    CHECK(info.compressed_key_bits() > info.compressed_val_bits(),
          "K compressed > V compressed");

    printf("  %d/%d total passed\n", g_passed, g_total);
}

// -----------------------------------------------------------------------
// Test 4: C API compatibility
// -----------------------------------------------------------------------

static void test_c_api() {
    printf("\n[4. C API compatibility]\n");
    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;

    uint64_t seed = 777;
    std::vector<int16_t> vec(dim);
    random_vector(vec.data(), dim, seed, info.coord_width);

    // C++ path
    auto ck_cpp = engine.compress_key(vec);

    // C API path
    lhsi_kv_handle_t* h = lhsi_kv_create_default();
    uint32_t c_norm, c_res_norm;
    std::vector<uint8_t> c_indices(dim), c_signs(dim);
    lhsi_kv_compress_key(h, vec.data(), dim,
                         &c_norm, c_indices.data(),
                         &c_res_norm, c_signs.data());

    CHECK(c_norm == ck_cpp.norm, "C API norm matches C++");
    CHECK(c_res_norm == ck_cpp.residual_norm, "C API res_norm matches C++");

    bool indices_match = true;
    for (size_t i = 0; i < dim; i++)
        if (c_indices[i] != ck_cpp.indices[i]) indices_match = false;
    CHECK(indices_match, "C API indices match C++");

    bool signs_match = true;
    for (size_t i = 0; i < dim; i++)
        if (c_signs[i] != ck_cpp.signs[i]) signs_match = false;
    CHECK(signs_match, "C API signs match C++");

    // Decompress via C API
    std::vector<int16_t> c_out(dim);
    lhsi_kv_decompress_key(h, c_norm, c_indices.data(),
                           c_res_norm, c_signs.data(),
                           dim, c_out.data());

    auto dk_cpp = engine.decompress_key(ck_cpp);
    bool decomp_match = true;
    for (size_t i = 0; i < dim; i++)
        if (c_out[i] != dk_cpp[i]) decomp_match = false;
    CHECK(decomp_match, "C API decompress matches C++");

    lhsi_kv_destroy(h);
    printf("  %d/%d total passed\n", g_passed, g_total);
}

// -----------------------------------------------------------------------
// Test 5: Replay against hex test vectors
// -----------------------------------------------------------------------

static void test_replay_hex() {
    printf("\n[5. Replay hex vectors]\n");

    std::string tv_dir = "../../rtl/tb/testvectors/";
    auto inputs = read_hex_file(tv_dir + "input_vectors.hex", 16);
    auto exp_dk = read_hex_file(tv_dir + "expected_decompressed_k.hex", 16);
    auto exp_dv = read_hex_file(tv_dir + "expected_decompressed_v.hex", 16);

    if (inputs.empty()) {
        printf("  SKIPPED (no test vectors)\n");
        return;
    }

    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;
    size_t num_vectors = inputs.size() / dim;
    int mismatches_k = 0, mismatches_v = 0;

    for (size_t t = 0; t < num_vectors; t++) {
        const int16_t* vec = &inputs[t * dim];

        auto ck = engine.compress_key(vec, dim);
        auto dk = engine.decompress_key(ck);

        bool match_k = true;
        for (size_t i = 0; i < dim; i++)
            if (dk[i] != exp_dk[t * dim + i]) match_k = false;
        if (!match_k) mismatches_k++;

        auto cv = engine.compress_value(vec, dim);
        auto dv = engine.decompress_value(cv);

        bool match_v = true;
        for (size_t i = 0; i < dim; i++)
            if (dv[i] != exp_dv[t * dim + i]) match_v = false;
        if (!match_v) mismatches_v++;
    }

    g_total += 2;
    if (mismatches_k == 0) g_passed++;
    else { g_errors++; fprintf(stderr, "  K mismatches: %d/%zu\n",
                               mismatches_k, num_vectors); }
    if (mismatches_v == 0) g_passed++;
    else { g_errors++; fprintf(stderr, "  V mismatches: %d/%zu\n",
                               mismatches_v, num_vectors); }

    printf("  %zu vectors: K %s, V %s\n", num_vectors,
           mismatches_k == 0 ? "PASS" : "FAIL",
           mismatches_v == 0 ? "PASS" : "FAIL");
}

// -----------------------------------------------------------------------
// Test 6: Compression ratio
// -----------------------------------------------------------------------

static void test_compression_ratio() {
    printf("\n[6. Compression ratio]\n");
    KVCacheEngineInfo info;

    double cr_k = info.compression_ratio_k();
    double cr_v = info.compression_ratio_v();
    double cr_combined = 2.0 * info.uncompressed_bits() /
        (info.compressed_key_bits() + info.compressed_val_bits());

    CHECK(cr_k >= 3.0, "K ratio %.2f < 3.0", cr_k);
    CHECK(cr_v >= 3.0, "V ratio %.2f < 3.0", cr_v);
    CHECK(cr_combined >= 3.0, "combined ratio %.2f < 3.0", cr_combined);

    printf("  K=%.2fx  V=%.2fx  combined=%.2fx\n", cr_k, cr_v, cr_combined);
}

// -----------------------------------------------------------------------
// Test 7: Store/load round-trip
// -----------------------------------------------------------------------

static void test_store_load() {
    printf("\n[7. Store/load round-trip]\n");
    KVCacheEngineInfo info;
    KVCacheEngine engine(info);
    size_t dim = info.vector_dim;

    uint64_t seed = 11111;
    std::vector<int16_t> key(dim), val(dim);
    random_vector(key.data(), dim, seed, info.coord_width);
    random_vector(val.data(), dim, seed, info.coord_width);

    engine.store_kv(0, key.data(), val.data(), dim);
    CHECK(engine.occupancy() == 1, "occupancy after store");

    std::vector<int16_t> loaded_k(dim), loaded_v(dim);
    engine.load_k(0, loaded_k.data(), dim);
    engine.load_v(0, loaded_v.data(), dim);

    // Loaded values should match direct compress/decompress
    KVCacheEngine engine2(info);
    auto ck = engine2.compress_key(key);
    auto dk = engine2.decompress_key(ck);
    auto cv = engine2.compress_value(val);
    auto dv = engine2.decompress_value(cv);

    bool k_match = true, v_match = true;
    for (size_t i = 0; i < dim; i++) {
        if (loaded_k[i] != dk[i]) k_match = false;
        if (loaded_v[i] != dv[i]) v_match = false;
    }
    CHECK(k_match, "store/load K matches direct compress/decompress");
    CHECK(v_match, "store/load V matches direct compress/decompress");

    // Reset
    engine.reset();
    CHECK(engine.occupancy() == 0, "occupancy after reset");

    printf("  %d/%d total passed\n", g_passed, g_total);
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

int main() {
    test_canonical_boundary_cases();
    test_round_trip_mse();
    test_kv_asymmetry();
    test_c_api();
    test_replay_hex();
    test_compression_ratio();
    test_store_load();

    printf("\n============================================================\n");
    printf("%d/%d tests passed", g_passed, g_total);
    if (g_errors > 0)
        printf(" (%d FAILED)", g_errors);
    printf("\n");

    if (g_errors > 0) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
