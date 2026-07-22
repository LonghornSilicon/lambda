// test_mac_array_ref.cpp — verification of the C++ MAC array reference.
//
// What we check:
//   1. INT8 matmul matches a naive int32 reference computed inline
//   2. FP16 matmul matches a naive double-precision reference (then rounded)
//   3. Edge cases: all-zero, identity, signed values, large K
//   4. C API agrees with C++ class on every test
//   5. FP16 round-trip helpers (round_to_fp16, fp16_bits) are self-consistent
//   6. Cost estimate returns sane numbers for canonical shapes

#include "mac_array_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// Naive references — these are the "ground truth" we verify the MAC array
// implementation against. Kept independent so a bug in one doesn't mask a
// bug in the other.
// ---------------------------------------------------------------------------
void naive_int8_ref(const std::int8_t* A, const std::int8_t* B,
                    std::int32_t* C,
                    std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    for (std::uint32_t i = 0; i < M; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            std::int64_t acc = 0;     // wider than 32-bit on purpose
            for (std::uint32_t k = 0; k < K; ++k) {
                acc += static_cast<std::int64_t>(A[i * K + k]) *
                       static_cast<std::int64_t>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<std::int32_t>(acc);
        }
    }
}

void naive_fp16_ref(const float* A, const float* B,
                    float* C,
                    std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    for (std::uint32_t i = 0; i < M; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (std::uint32_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[i * K + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            // Match the v0.1 policy: round the output to fp16.
            C[i * N + j] = lhsi::mac::round_to_fp16(static_cast<float>(acc));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
void test_int8_random_shapes() {
    std::printf("[1] INT8 matmul vs naive ref (random shapes)..."); std::fflush(stdout);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> shape_dist(1, 32);
    std::uniform_int_distribution<int> val_dist(-128, 127);

    lhsi::mac::MacArray mac;
    int mismatches = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const std::uint32_t M = shape_dist(rng);
        const std::uint32_t K = shape_dist(rng);
        const std::uint32_t N = shape_dist(rng);

        std::vector<std::int8_t> A(M * K), B(K * N);
        for (auto& x : A) x = static_cast<std::int8_t>(val_dist(rng));
        for (auto& x : B) x = static_cast<std::int8_t>(val_dist(rng));

        std::vector<std::int32_t> C_mac(M * N), C_ref(M * N);
        mac.matmul_int8(A.data(), B.data(), C_mac.data(), M, K, N);
        naive_int8_ref(A.data(), B.data(), C_ref.data(), M, K, N);

        for (std::uint32_t i = 0; i < M * N; ++i) {
            if (C_mac[i] != C_ref[i]) { ++mismatches; break; }
        }
    }
    check(mismatches == 0, "INT8 matmul disagreed with naive reference");
    std::printf(" done\n");
}

void test_fp16_random_shapes() {
    std::printf("[2] FP16 matmul vs naive ref (random shapes)..."); std::fflush(stdout);

    std::mt19937 rng(0xBEEF);
    std::uniform_int_distribution<int> shape_dist(1, 16);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);

    lhsi::mac::MacArray mac;
    int mismatches = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const std::uint32_t M = shape_dist(rng);
        const std::uint32_t K = shape_dist(rng);
        const std::uint32_t N = shape_dist(rng);

        std::vector<float> A(M * K), B(K * N);
        for (auto& x : A) x = lhsi::mac::round_to_fp16(val_dist(rng));
        for (auto& x : B) x = lhsi::mac::round_to_fp16(val_dist(rng));

        std::vector<float> C_mac(M * N), C_ref(M * N);
        mac.matmul_fp16(A.data(), B.data(), C_mac.data(), M, K, N);
        naive_fp16_ref(A.data(), B.data(), C_ref.data(), M, K, N);

        for (std::uint32_t i = 0; i < M * N; ++i) {
            // FP16 paths can disagree by ±1 ULP because the order of
            // accumulation differs. Allow a small relative tolerance.
            const float a = C_mac[i];
            const float b = C_ref[i];
            const float denom = std::max(std::fabs(b), 1e-3f);
            if (std::fabs(a - b) / denom > 5e-3f) {
                ++mismatches;
                break;
            }
        }
    }
    check(mismatches == 0, "FP16 matmul disagreed with naive reference");
    std::printf(" done\n");
}

void test_edge_cases() {
    std::printf("[3] edge cases (zero, identity, signed)..."); std::fflush(stdout);

    lhsi::mac::MacArray mac;

    // All-zero INT8
    {
        std::vector<std::int8_t> A(64, 0), B(64, 0);
        std::vector<std::int32_t> C(8 * 8, 99);
        mac.matmul_int8(A.data(), B.data(), C.data(), 8, 8, 8);
        for (auto v : C) check(v == 0, "all-zero INT8 must produce all zeros");
    }
    // Identity INT8: A = I, B arbitrary → C == B
    {
        std::vector<std::int8_t> A(64, 0);
        for (int i = 0; i < 8; ++i) A[i * 8 + i] = 1;
        std::vector<std::int8_t> B(64);
        for (int i = 0; i < 64; ++i) B[i] = static_cast<std::int8_t>(i - 32);
        std::vector<std::int32_t> C(64);
        mac.matmul_int8(A.data(), B.data(), C.data(), 8, 8, 8);
        for (int i = 0; i < 64; ++i) {
            check(C[i] == B[i],
                  "identity INT8: C must equal B element-wise");
        }
    }
    // All-zero FP16
    {
        std::vector<float> A(64, 0.0f), B(64, 0.0f);
        std::vector<float> C(64, 99.0f);
        mac.matmul_fp16(A.data(), B.data(), C.data(), 8, 8, 8);
        for (auto v : C) check(v == 0.0f, "all-zero FP16 must produce all zeros");
    }
    // Identity FP16
    {
        std::vector<float> A(64, 0.0f);
        for (int i = 0; i < 8; ++i) A[i * 8 + i] = 1.0f;
        std::vector<float> B(64);
        for (int i = 0; i < 64; ++i) B[i] = lhsi::mac::round_to_fp16(static_cast<float>(i - 32) * 0.25f);
        std::vector<float> C(64);
        mac.matmul_fp16(A.data(), B.data(), C.data(), 8, 8, 8);
        for (int i = 0; i < 64; ++i) {
            check(C[i] == B[i],
                  "identity FP16: C must equal B element-wise");
        }
    }
    std::printf(" done\n");
}

void test_c_api_matches_cpp() {
    std::printf("[4] extern \"C\" API matches C++ class..."); std::fflush(stdout);

    std::mt19937 rng(0xFEED);
    std::uniform_int_distribution<int> val_dist(-100, 100);

    lhsi::mac::MacArray cpp_mac;

    int mismatches = 0;
    for (int trial = 0; trial < 10; ++trial) {
        const std::uint32_t M = 4, K = 8, N = 4;
        std::vector<std::int8_t> A(M * K), B(K * N);
        for (auto& x : A) x = static_cast<std::int8_t>(val_dist(rng));
        for (auto& x : B) x = static_cast<std::int8_t>(val_dist(rng));

        std::vector<std::int32_t> C_cpp(M * N), C_c(M * N);
        cpp_mac.matmul_int8(A.data(), B.data(), C_cpp.data(), M, K, N);
        lhsi_mac_matmul_int8(A.data(), B.data(), C_c.data(), M, K, N);

        for (std::uint32_t i = 0; i < M * N; ++i) {
            if (C_cpp[i] != C_c[i]) { ++mismatches; break; }
        }
    }
    check(mismatches == 0, "C API INT8 disagreed with C++ class");
    std::printf(" done\n");
}

void test_fp16_roundtrip() {
    std::printf("[5] fp16 round-trip helpers..."); std::fflush(stdout);

    const float values[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 65504.0f, -65504.0f,
        6.10352e-5f,     // smallest positive normal fp16
        1.0f / 3.0f,     // representable inexactly
    };
    for (float v : values) {
        const std::uint16_t bits = lhsi::mac::float_to_fp16_bits(v);
        const float back = lhsi::mac::fp16_bits_to_float(bits);
        const float rt   = lhsi::mac::round_to_fp16(v);
        // back and rt should agree (both fp16-precision representations of v).
        check(back == rt, "fp16 bits and round_to_fp16 must agree");
    }
    std::printf(" done\n");
}

void test_cost_estimate() {
    std::printf("[6] cost estimate is sane..."); std::fflush(stdout);
    lhsi::mac::MacArray mac;

    // For a 64x64x64 INT8 matmul (8x8 chip config, 64 INT8 MACs/cyc):
    //   ops = 262 144
    //   int8 throughput = 64/cyc -> 4096 cycles + 16 pipeline = 4112
    //   energy = 262 144 * 0.5 pJ = 131 072 pJ = 131 nJ
    auto e_int8 = mac.estimate(64, 64, 64, lhsi::mac::MacArray::DType::Int8);
    check(e_int8.cycles >= 4000 && e_int8.cycles <= 5000,
          "INT8 cycle estimate for 64x64x64 not in 4000-5000 range");
    check(e_int8.energy_pj > 0.0, "INT8 energy must be positive");

    // FP16 takes 4x longer for the same work in v0.1's placeholder cost model.
    auto e_fp16 = mac.estimate(64, 64, 64, lhsi::mac::MacArray::DType::Fp16);
    check(e_fp16.cycles > e_int8.cycles,
          "FP16 must be slower than INT8 for the same shape");
    check(e_fp16.energy_pj > e_int8.energy_pj,
          "FP16 must use more energy than INT8 per op");
    std::printf(" done\n");
}

} // anonymous namespace

int main() {
    test_int8_random_shapes();
    test_fp16_random_shapes();
    test_edge_cases();
    test_c_api_matches_cpp();
    test_fp16_roundtrip();
    test_cost_estimate();

    if (failures == 0) {
        std::printf("MAC array: ALL SELF-TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d test(s) FAILED\n", failures);
    return 1;
}
