// test_precision_controller_ref.cpp — bit-exact verification of the C++
// reference model against the same RTL testvectors that gate the
// SystemVerilog TB and the Python model.
//
// What we check, in order:
//   1. C++ stateful streaming model: 143/143 replay-tile decisions match
//      the RTL TB's expected.hex
//   2. C++ stateless decide() static method: agrees with the streaming
//      model on every tile
//   3. C API (lhsi_pc_*): produces identical results via the extern "C"
//      shim
//   4. The two canonical edge cases (spike=10 INT8, spike=11 FP16)
//   5. Accumulator reset between consecutive tiles works
//
// Build + run:    make test
// Or manually:    g++ -std=c++17 -O2 -Wall -Wextra test_precision_controller_ref.cpp
//                     precision_controller_ref.cpp -o test_ref && ./test_ref

#include "precision_controller_ref.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t N_TILE = 64 * 64;   // matches default BLOCK_M*BLOCK_N

// ---------------------------------------------------------------------------
// File location: walk up from this source file to find rtl/tb/testvectors/.
// We try the canonical paths the repo uses for both Python and C++ runs.
// ---------------------------------------------------------------------------
std::string find_repo_root() {
    const char* candidates[] = {
        "../..",                                           // running from sw/reference_model
        "../../../..",                                     // belt + suspenders
        ".",
    };
    for (auto c : candidates) {
        std::ifstream check(std::string(c) + "/rtl/tb/testvectors/expected.hex");
        if (check.good()) return c;
    }
    return ".";
}

std::vector<std::vector<std::int32_t>> load_tiles(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<std::int32_t> flat;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Parse the hex byte.
        const unsigned long byte_v = std::strtoul(line.c_str(), nullptr, 16);
        const std::uint8_t b = static_cast<std::uint8_t>(byte_v & 0xFFu);
        // Convert int8 two's complement → signed int.
        std::int32_t s = (b & 0x80) ? (static_cast<std::int32_t>(b) - 256) : b;
        flat.push_back(s);
    }
    if (flat.size() % N_TILE != 0) {
        std::fprintf(stderr, "ERROR: scores.hex has %zu entries; not a "
                             "multiple of N=%zu\n", flat.size(), N_TILE);
        std::exit(2);
    }
    std::vector<std::vector<std::int32_t>> tiles;
    for (std::size_t i = 0; i < flat.size(); i += N_TILE) {
        tiles.emplace_back(flat.begin() + i, flat.begin() + i + N_TILE);
    }
    return tiles;
}

std::vector<bool> load_expected(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<bool> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        out.push_back(std::strtoul(line.c_str(), nullptr, 16) & 1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    }
}

void test_canonical_boundary_cases() {
    std::cout << "[1] canonical edge cases (spike=10 / spike=11)..." << std::flush;

    auto bg1_spike = [](std::int32_t spike) {
        std::vector<std::int32_t> tile(N_TILE, 1);
        tile[N_TILE / 2] = spike;
        return tile;
    };

    // LHS = 10*4096 = 40960  <  RHS = 10*(4095+10) = 41050  → INT8
    {
        lhsi::PrecisionController pc;
        bool d = pc.process_tile(bg1_spike(10));
        check(!d, "spike=10 must decide INT8 (LHS=40960 < RHS=41050)");
    }
    // LHS = 11*4096 = 45056  >  RHS = 10*(4095+11) = 41060  → FP16
    {
        lhsi::PrecisionController pc;
        bool d = pc.process_tile(bg1_spike(11));
        check(d, "spike=11 must decide FP16 (LHS=45056 > RHS=41060)");
    }
    std::cout << " done\n";
}

void test_stateful_matches_stateless(const std::vector<std::vector<std::int32_t>>& tiles) {
    std::cout << "[2] stateful streaming agrees with stateless decide()..." << std::flush;
    lhsi::PrecisionController pc;
    int mismatches = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        bool stateful  = pc.process_tile(tiles[i]);
        bool stateless = lhsi::PrecisionController::decide(
            tiles[i].data(), tiles[i].size());
        if (stateful != stateless) ++mismatches;
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "  FAIL: stateful vs stateless disagreed on %d/%zu tiles\n",
                     mismatches, tiles.size());
        ++failures;
    }
    std::cout << " done\n";
}

void test_c_api_matches_cpp(const std::vector<std::vector<std::int32_t>>& tiles) {
    std::cout << "[3] extern \"C\" API matches C++ class..." << std::flush;
    lhsi::PrecisionController cpp_pc;
    lhsi_pc_handle_t* c_pc = lhsi_pc_create();
    int mismatches = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        bool cpp_d = cpp_pc.process_tile(tiles[i]);
        int  c_d   = lhsi_pc_process_tile(c_pc, tiles[i].data(), tiles[i].size());
        if (cpp_d != (c_d != 0)) ++mismatches;
    }
    lhsi_pc_destroy(c_pc);
    if (mismatches != 0) {
        std::fprintf(stderr, "  FAIL: C API disagreed on %d/%zu tiles\n",
                     mismatches, tiles.size());
        ++failures;
    }
    std::cout << " done\n";
}

void test_full_replay_bit_exact(
    const std::vector<std::vector<std::int32_t>>& tiles,
    const std::vector<bool>& expected) {

    std::cout << "[4] full replay 143/143 bit-exact vs RTL..." << std::flush;

    if (tiles.size() != expected.size()) {
        std::fprintf(stderr,
            "\n  FAIL: scores.hex implies %zu tiles but expected.hex has %zu lines\n",
            tiles.size(), expected.size());
        ++failures;
        return;
    }

    lhsi::PrecisionController pc;
    int mismatches = 0;
    std::vector<std::size_t> bad_indices;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        bool got = pc.process_tile(tiles[i]);
        if (got != expected[i]) {
            ++mismatches;
            if (bad_indices.size() < 10) bad_indices.push_back(i);
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "\n  FAIL: %d/%zu tiles disagreed with RTL "
                             "expected.hex\n", mismatches, tiles.size());
        for (auto idx : bad_indices) {
            std::fprintf(stderr, "    tile %zu: expected=%s\n", idx,
                         expected[idx] ? "FP16" : "INT8");
        }
        ++failures;
        return;
    }
    std::cout << " done\n";
    std::printf("Tests: %zu  Pass: %zu  Fail: 0  ALL TESTS PASSED\n",
                tiles.size(), tiles.size());
}

void test_accumulator_reset() {
    std::cout << "[5] accumulator reset between tiles..." << std::flush;
    lhsi::PrecisionController pc;

    std::vector<std::int32_t> big_tile(N_TILE, 127);   // max sum + max max
    (void)pc.process_tile(big_tile);

    std::vector<std::int32_t> zero_tile(N_TILE, 0);
    bool d = pc.process_tile(zero_tile);
    check(!d, "all-zero tile after large tile must decide INT8");
    std::cout << " done\n";
}

} // anonymous namespace

int main() {
    const std::string root = find_repo_root();
    const std::string scores_path   = root + "/rtl/tb/testvectors/scores.hex";
    const std::string expected_path = root + "/rtl/tb/testvectors/expected.hex";

    const auto tiles    = load_tiles(scores_path);
    const auto expected = load_expected(expected_path);

    test_canonical_boundary_cases();
    test_stateful_matches_stateless(tiles);
    test_c_api_matches_cpp(tiles);
    test_full_replay_bit_exact(tiles, expected);
    test_accumulator_reset();

    if (failures == 0) {
        std::printf("ALL SELF-TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d test(s) FAILED\n", failures);
    return 1;
}
