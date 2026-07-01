// channelquant_ref.cpp — ChannelQuant C++ reference codec implementation.
// Mirrors ChannelQuant/reference/channelquant_ref.py compress_*/decompress_*.
#include "channelquant_ref.hpp"
#include <algorithm>

namespace lhsi { namespace cq {

// ---- values: per-token scale over D dims (contract §2) ----------------------
ValueBlob compress_values(const std::vector<uint16_t>& V, int T, int D, int bits) {
    ValueBlob b;
    b.T = T; b.D = D; b.bits = bits;
    b.scales.resize(T);
    b.codes.resize(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; t++) {
        uint16_t s = scale_from_amax(amax_bits(V, t * D, 1, D), bits);   // one scale per token
        b.scales[t] = s;
        for (int d = 0; d < D; d++)
            b.codes[static_cast<size_t>(t) * D + d] = quant_code(V[t * D + d], s, bits);
    }
    b.payload = (bits == 4) ? pack_int4(b.codes) : pack_int8(b.codes);
    return b;
}

std::vector<uint32_t> decompress_values(const ValueBlob& b) {
    std::vector<uint32_t> out(static_cast<size_t>(b.T) * b.D);
    for (int t = 0; t < b.T; t++)
        for (int d = 0; d < b.D; d++) {
            size_t i = static_cast<size_t>(t) * b.D + d;
            out[i] = dequant_f32(b.codes[i], b.scales[t]);
        }
    return out;
}

// ---- keys: per-channel scale over a token group (contract §3/§4) ------------
static std::vector<std::pair<int,int>> group_bounds(int T, int G) {
    std::vector<std::pair<int,int>> g;
    if (G <= 0) { g.push_back({0, T}); return g; }
    for (int s = 0; s < T; s += G) g.push_back({s, std::min(s + G, T)});
    return g;
}

KeyBlob compress_keys(const std::vector<uint16_t>& K, int T, int D, int bits,
                      int G, const std::vector<int>& outlier_idx) {
    KeyBlob b;
    b.T = T; b.D = D; b.bits = bits; b.G = G;
    // sorted outlier set -> keep = complement (contract §4: INT4 over non-outliers)
    b.outlier = outlier_idx;
    std::sort(b.outlier.begin(), b.outlier.end());
    std::vector<char> is_out(D, 0);
    for (int c : b.outlier) is_out[c] = 1;
    for (int c = 0; c < D; c++) if (!is_out[c]) b.keep.push_back(c);
    const int nk = static_cast<int>(b.keep.size());

    b.groups = group_bounds(T, G);
    for (auto& gb : b.groups) {
        int a = gb.first, bb = gb.second, g = bb - a;
        std::vector<int> codes(static_cast<size_t>(g) * nk);
        for (int ci = 0; ci < nk; ci++) {
            int cc = b.keep[ci];
            uint16_t s = scale_from_amax(amax_bits(K, a * D + cc, D, g), bits); // per-channel over g tokens
            b.scales.push_back(s);
            for (int t = 0; t < g; t++)
                codes[static_cast<size_t>(t) * nk + ci] = quant_code(K[(a + t) * D + cc], s, bits);
        }
        std::vector<uint8_t> pay = pack_int4(codes);   // keys are INT4 in CQ-4/CQ-4+
        b.payload.insert(b.payload.end(), pay.begin(), pay.end());
    }
    // outlier sidecar: identity fp16 of K at outlier channels, t-major (contract §4.2)
    b.sidecar.resize(static_cast<size_t>(T) * b.outlier.size());
    for (int t = 0; t < T; t++)
        for (size_t ci = 0; ci < b.outlier.size(); ci++)
            b.sidecar[static_cast<size_t>(t) * b.outlier.size() + ci] = K[t * D + b.outlier[ci]];
    return b;
}

std::vector<uint32_t> decompress_keys(const KeyBlob& b) {
    std::vector<uint32_t> out(static_cast<size_t>(b.T) * b.D, 0);
    const int nk = static_cast<int>(b.keep.size());
    int sc_base = 0, nib = 0;
    for (auto& gb : b.groups) {
        int a = gb.first, bb = gb.second, g = bb - a;
        int gn = g * nk;
        // unpack this group's int4 payload back to codes[g, nk] (C-order)
        std::vector<int> codes(gn);
        for (int i = 0; i < gn; i++) {
            uint8_t byte = b.payload[nib + i / 2];
            int nibv = (i & 1) ? (byte >> 4) & 0x0F : byte & 0x0F;
            codes[i] = (nibv >= 8) ? nibv - 16 : nibv;   // sign-extend int4
        }
        nib += (gn + 1) / 2;
        for (int ci = 0; ci < nk; ci++) {
            int cc = b.keep[ci];
            uint16_t s = b.scales[sc_base + ci];
            for (int t = 0; t < g; t++)
                out[static_cast<size_t>(a + t) * b.D + cc] =
                    dequant_f32(codes[static_cast<size_t>(t) * nk + ci], s);
        }
        sc_base += nk;
    }
    // outlier columns: fp16 value widened to fp32 (identity)
    for (int t = 0; t < b.T; t++)
        for (size_t ci = 0; ci < b.outlier.size(); ci++) {
            uint16_t h = b.sidecar[static_cast<size_t>(t) * b.outlier.size() + ci];
            out[static_cast<size_t>(t) * b.D + b.outlier[ci]] = real_to_f32(f16_to_real(h));
        }
    return out;
}

}} // namespace lhsi::cq
