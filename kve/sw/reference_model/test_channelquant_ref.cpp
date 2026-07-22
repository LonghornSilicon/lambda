// test_channelquant_ref.cpp — C++ leg of the 3-way ChannelQuant parity gate.
//
// Loads the SAME vendored golden hex vectors that tb_channelquant.sv drives
// (rtl/tb/testvectors/channelquant/hex/<vector>/) and asserts the C++ reference
// reproduces, bit-for-bit:
//   - per-axis fp16 scales           (val_scales / key_scales)
//   - packed byte stream             (val_payload / key_payload, int4/int8)
//   - reconstructed K/V_hat fp32      (expected_v_hat / expected_k_hat)
//   - CQ-4+ fp16 outlier sidecar     (sidecar)
// for all 9 vectors (CQ-8/CQ-4/CQ-4+, D in {64,128}, full g=G and partial g<G
// key groups). Mirrors the checks in tb_channelquant.sv. Exit code = #failures.
//
// Build/run:  make test-cq   (sw/reference_model/Makefile)
#include "channelquant_ref.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace lhsi::cq;

static const char* TVDIR =
    "../../rtl/tb/testvectors/channelquant/hex";   // relative to sw/reference_model

// ---- read a $readmemh-style hex file: whitespace-separated hex tokens -------
static std::vector<uint32_t> read_hex(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "FATAL: cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<uint32_t> out;
    std::string tok;
    while (f >> tok) out.push_back(static_cast<uint32_t>(std::strtoul(tok.c_str(), nullptr, 16)));
    return out;
}

struct Vec { const char* name; int D, T, bits, G, tier, k; };
static const Vec VECS[9] = {
    {"d64_T128_G64__CQ8",      64, 128, 8, 0,   0, 0},
    {"d64_T128_G64__CQ4",      64, 128, 4, 64,  1, 0},
    {"d64_T128_G64__CQ4plus",  64, 128, 4, 64,  2, 2},
    {"d64_T70_G64__CQ8",       64, 70,  8, 0,   0, 0},
    {"d64_T70_G64__CQ4",       64, 70,  4, 64,  1, 0},
    {"d64_T70_G64__CQ4plus",   64, 70,  4, 64,  2, 2},
    {"d128_T100_G128__CQ8",   128, 100, 8, 0,   0, 0},
    {"d128_T100_G128__CQ4",   128, 100, 4, 128, 1, 0},
    {"d128_T100_G128__CQ4plus",128,100, 4, 128, 2, 2},
};

// compare helper: returns #mismatches, prints up to `lim`
template <class A, class B>
static int cmp(const char* what, const std::vector<A>& got, const std::vector<B>& exp,
               const char* vname, int lim = 4) {
    int fails = 0;
    if (got.size() != exp.size()) {
        std::printf("  [%s] %s size %zu != golden %zu\n", vname, what, got.size(), exp.size());
        return static_cast<int>(got.size() > exp.size() ? got.size() : exp.size());
    }
    for (size_t i = 0; i < got.size(); i++)
        if (static_cast<uint64_t>(got[i]) != static_cast<uint64_t>(exp[i])) {
            if (++fails <= lim)
                std::printf("  [%s] %s[%zu]: got %llx exp %llx\n", vname, what, i,
                            (unsigned long long)got[i], (unsigned long long)exp[i]);
        }
    return fails;
}

static std::string vp(const Vec& v, const char* file) {
    return std::string(TVDIR) + "/" + v.name + "/" + file;
}

int main() {
    int total_fail = 0;
    for (const Vec& v : VECS) {
        int n = v.D * v.T;
        int fv = 0, fk = 0;

        // ---------- VALUES (per-token, all tiers) ----------------------------
        {
            auto in_v = read_hex(vp(v, "input_v.f16.hex"));
            auto g_sc = read_hex(vp(v, "val_scales.f16.hex"));
            auto g_pay = read_hex(vp(v, "val_payload.u8.hex"));
            auto g_hat = read_hex(vp(v, "expected_v_hat.f32.hex"));
            std::vector<uint16_t> V(in_v.begin(), in_v.end());

            ValueBlob b = compress_values(V, v.T, v.D, v.bits);
            auto hat = decompress_values(b);
            fv += cmp("v scale", b.scales, g_sc, v.name);
            fv += cmp("v payload", b.payload, g_pay, v.name);
            fv += cmp("v_hat", hat, g_hat, v.name);
        }

        // ---------- KEYS -----------------------------------------------------
        if (v.tier == 0) {
            // CQ-8 keys are per-token int8 — same path as values
            auto in_k = read_hex(vp(v, "input_k.f16.hex"));
            auto g_sc = read_hex(vp(v, "key_scales.f16.hex"));
            auto g_pay = read_hex(vp(v, "key_payload.u8.hex"));
            auto g_hat = read_hex(vp(v, "expected_k_hat.f32.hex"));
            std::vector<uint16_t> K(in_k.begin(), in_k.end());

            ValueBlob b = compress_values(K, v.T, v.D, 8);
            auto hat = decompress_values(b);
            fk += cmp("k scale", b.scales, g_sc, v.name);
            fk += cmp("k payload", b.payload, g_pay, v.name);
            fk += cmp("k_hat", hat, g_hat, v.name);
        } else {
            auto in_k = read_hex(vp(v, "input_k.f16.hex"));
            auto g_sc = read_hex(vp(v, "key_scales.f16.hex"));
            auto g_pay = read_hex(vp(v, "key_payload.u8.hex"));
            auto g_hat = read_hex(vp(v, "expected_k_hat.f32.hex"));
            auto mask = read_hex(vp(v, "outlier_mask.u8.hex"));      // 1 -> outlier channel
            std::vector<uint16_t> K(in_k.begin(), in_k.end());

            std::vector<int> outlier;
            for (int c = 0; c < v.D; c++) if (mask[c] & 1) outlier.push_back(c);
            if (static_cast<int>(outlier.size()) != v.k)
                std::printf("  [%s] WARN mask popcount %zu != k %d\n", v.name, outlier.size(), v.k);

            KeyBlob b = compress_keys(K, v.T, v.D, 4, v.G, outlier);
            auto hat = decompress_keys(b);
            fk += cmp("k scale", b.scales, g_sc, v.name);
            fk += cmp("k payload", b.payload, g_pay, v.name);
            fk += cmp("k_hat", hat, g_hat, v.name);
            if (v.k > 0) {
                auto side = read_hex(vp(v, "sidecar.f16.hex"));
                fk += cmp("sidecar", b.sidecar, side, v.name);
            }
        }

        std::printf("%-26s D=%3d T=%3d G=%3d tier=%d : V %s / K %s  (%d+%d mism)\n",
                    v.name, v.D, v.T, v.G, v.tier,
                    fv ? "FAIL" : "PASS", fk ? "FAIL" : "PASS", fv, fk);
        (void)n;
        total_fail += fv + fk;
    }
    std::printf("============================================================\n");
    if (total_fail == 0)
        std::printf("CHANNELQUANT C++ PARITY: ALL 9 VECTORS BIT-EXACT (V+K, all tiers)\n");
    else
        std::printf("CHANNELQUANT C++ PARITY: %d TOTAL MISMATCHES\n", total_fail);
    std::printf("============================================================\n");
    return total_fail == 0 ? 0 : 1;
}
