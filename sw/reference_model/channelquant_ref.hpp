#ifndef CHANNELQUANT_REF_HPP
#define CHANNELQUANT_REF_HPP
//
// channelquant_ref.hpp — ChannelQuant C++ reference codec (the 3-way parity C++
// leg). Implements ChannelQuant/docs/HW_CONTRACT.md §1-§5 bit-exactly.
//
// The fp16/fp32 helpers and the quant/dequant/scale core are a 1:1 port of the
// RTL behavioral model (rtl/cq_fp_pkg.sv + rtl/cq_units.sv), which is proven
// bit-exact vs the golden vectors (make sim_cq). Porting the *same* double-based
// arithmetic makes the C++ leg match the SV leg by construction, and both match
// the numpy reference (ChannelQuant/reference/channelquant_ref.py) — closing
// Python<->C++<->SV parity (contract §8). See NOTES.md for the double-vs-float32
// justification (identical codes on all 9 vectors).
//
#include <cstdint>
#include <cmath>
#include <vector>
#include <utility>
#include <cstddef>

namespace lhsi { namespace cq {

constexpr double EPS = 1.0 / 16384.0;   // 2^-14 (contract §1 scale floor)

// ---- round-half-to-even to int64 (numpy rint) ------------------------------
inline long long rint_ll(double x) {          // x >= 0
    double f = std::floor(x);
    long long fi = static_cast<long long>(f);
    double diff = x - f;
    if (diff < 0.5) return fi;
    if (diff > 0.5) return fi + 1;
    return (fi % 2 == 0) ? fi : fi + 1;        // tie -> even
}
inline long long srint_ll(double x) {
    return (x < 0.0) ? -rint_ll(-x) : rint_ll(x);
}

// ---- IEEE-754 half/single bit pattern <-> double (double is exact for both) --
inline double f16_to_real(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    double val;
    if (exp == 0)          val = std::ldexp(static_cast<double>(man), -24);      // 0/subnormal
    else if (exp == 0x1F)  val = (man == 0) ? 1e30 : 0.0;                        // inf/nan sentinel
    else                   val = std::ldexp(1.0 + std::ldexp(static_cast<double>(man), -10),
                                            static_cast<int>(exp) - 15);
    return sign ? -val : val;
}
inline double f32_to_real(uint32_t w) {
    uint32_t sign = (w >> 31) & 1u, exp = (w >> 23) & 0xFFu, man = w & 0x7FFFFFu;
    double val;
    if (exp == 0)          val = std::ldexp(static_cast<double>(man), -149);
    else if (exp == 0xFF)  val = (man == 0) ? 1e30 : 0.0;
    else                   val = std::ldexp(1.0 + std::ldexp(static_cast<double>(man), -23),
                                            static_cast<int>(exp) - 127);
    return sign ? -val : val;
}

// ---- double -> fp16 bits, round-half-to-even (numpy .astype(float16)) --------
// 1:1 port of cq_fp_pkg::real_to_f16.
inline uint16_t real_to_f16(double x) {
    if (x == 0.0) return 0;
    int sign = (x < 0.0) ? 1 : 0;
    double ax = (x < 0.0) ? -x : x;
    int e = 0;                                 // ax/2^e in [1,2)
    if (ax >= 1.0) { while (std::ldexp(ax, -(e + 1)) >= 1.0) e++; }
    else           { while (std::ldexp(ax, -e) < 1.0)        e--; }
    int biased = e + 15;
    if (biased >= 31)
        return static_cast<uint16_t>((sign << 15) | (0x1Fu << 10));            // overflow -> inf
    if (biased <= 0) {                                                         // subnormal
        long long mant = rint_ll(std::ldexp(ax, 24));
        if (mant >= 1024) return static_cast<uint16_t>((sign << 15) | (0x01u << 10));
        return static_cast<uint16_t>((sign << 15) | (mant & 0x3FF));
    }
    long long mant = rint_ll(std::ldexp(ax, -e) * 1024.0);                     // [1024,2048]
    if (mant >= 2048) {
        biased++; mant = 1024;
        if (biased >= 31) return static_cast<uint16_t>((sign << 15) | (0x1Fu << 10));
    }
    return static_cast<uint16_t>((sign << 15) | ((biased & 0x1F) << 10)
                                 | ((mant - 1024) & 0x3FF));
}

// ---- double -> fp32 bits, round-half-to-even --------------------------------
// 1:1 port of cq_fp_pkg::real_to_f32.
inline uint32_t real_to_f32(double x) {
    if (x == 0.0) return 0;
    int sign = (x < 0.0) ? 1 : 0;
    double ax = (x < 0.0) ? -x : x;
    int e = 0;
    if (ax >= 1.0) { while (std::ldexp(ax, -(e + 1)) >= 1.0) e++; }
    else           { while (std::ldexp(ax, -e) < 1.0)        e--; }
    int biased = e + 127;
    if (biased >= 255)
        return (static_cast<uint32_t>(sign) << 31) | (0xFFu << 23);
    if (biased <= 0) {
        long long mant = rint_ll(std::ldexp(ax, 149));
        if (mant >= (1LL << 23)) return (static_cast<uint32_t>(sign) << 31) | (0x01u << 23);
        return (static_cast<uint32_t>(sign) << 31) | static_cast<uint32_t>(mant & 0x7FFFFF);
    }
    long long mant = rint_ll(std::ldexp(ax, -e) * 8388608.0);                  // *2^23
    if (mant >= (1LL << 24)) {
        biased++; mant = (1LL << 23);
        if (biased >= 255) return (static_cast<uint32_t>(sign) << 31) | (0xFFu << 23);
    }
    return (static_cast<uint32_t>(sign) << 31)
           | (static_cast<uint32_t>(biased & 0xFF) << 23)
           | (static_cast<uint32_t>(mant - (1LL << 23)) & 0x7FFFFF);
}

// ---- quant core (contract §1), mirrors cq_units.sv --------------------------
inline int qmax_of(int bits) { return (1 << (bits - 1)) - 1; }
inline int qmin_of(int bits) { return -(1 << (bits - 1)); }

// max-abs over a strided run of fp16 words -> the winner's fp16 bits (sign cleared)
inline uint16_t amax_bits(const std::vector<uint16_t>& buf, int base, int stride, int count) {
    double best = -1.0; uint16_t bestbits = 0;
    for (int i = 0; i < count; i++) {
        uint16_t mag = buf[base + i * stride] & 0x7FFF;
        double m = f16_to_real(mag);
        if (m > best) { best = m; bestbits = mag; }
    }
    return bestbits;
}

// s = max(amax/qmax, EPS) cast to fp16  (cq_scale_unit)
inline uint16_t scale_from_amax(uint16_t amax_f16, int bits) {
    double s = f16_to_real(amax_f16) / qmax_of(bits);
    if (s < EPS) s = EPS;
    return real_to_f16(s);
}

// q = clamp(round_half_even(x/s), qmin, qmax)  (cq_quant_unit)
inline int quant_code(uint16_t x_f16, uint16_t scale_f16, int bits) {
    long long r = srint_ll(f16_to_real(x_f16) / f16_to_real(scale_f16));
    if (r > qmax_of(bits)) r = qmax_of(bits);
    if (r < qmin_of(bits)) r = qmin_of(bits);
    return static_cast<int>(r);
}

// x_hat = code * s, emitted as fp32 bits  (cq_dequant_unit)
inline uint32_t dequant_f32(int code, uint16_t scale_f16) {
    return real_to_f32(static_cast<double>(code) * f16_to_real(scale_f16));
}

// ---- nibble/byte packing (contract §5) --------------------------------------
// element 2i -> low nibble, 2i+1 -> high nibble; odd tail zero-padded.
inline std::vector<uint8_t> pack_int4(const std::vector<int>& codes) {
    std::vector<uint8_t> out((codes.size() + 1) / 2, 0);
    for (size_t i = 0; i < codes.size(); i++) {
        uint8_t nib = static_cast<uint8_t>(codes[i] & 0x0F);
        if (i & 1) out[i / 2] |= static_cast<uint8_t>(nib << 4);
        else       out[i / 2] |= nib;
    }
    return out;
}
inline std::vector<uint8_t> pack_int8(const std::vector<int>& codes) {
    std::vector<uint8_t> out(codes.size());
    for (size_t i = 0; i < codes.size(); i++)
        out[i] = static_cast<uint8_t>(codes[i] & 0xFF);
    return out;
}

// ---- WHT-rotated 3-bit VALUE path (fixed Walsh-Hadamard; Abhiram Bandi + Chaithu) ----
// In-place radix-2 Walsh-Hadamard over fp16 bits, ADD/SUB ONLY (no normalization). Each
// butterfly is real_to_f16(exact double sum) — matches numpy fp16 add and the RTL fp16
// adder. D must be a power of two. Self-inverse up to the 1/D scale applied on decode.
inline void fwht_raw_f16(std::vector<uint16_t>& x, int D) {
    for (int h = 1; h < D; h <<= 1)
        for (int i = 0; i < D; i += (h << 1))
            for (int j = i; j < i + h; j++) {
                double a = f16_to_real(x[j]);
                double b = f16_to_real(x[j + h]);
                x[j]     = real_to_f16(a + b);
                x[j + h] = real_to_f16(a - b);
            }
}

// 8 signed 3-bit codes -> 3 bytes. code&0x7 at bit offset 3*i (little-endian within the stream).
inline std::vector<uint8_t> pack_int3(const std::vector<int>& codes) {
    std::vector<uint8_t> out((codes.size() * 3 + 7) / 8, 0);
    for (size_t i = 0; i < codes.size(); i++) {
        uint32_t c = static_cast<uint32_t>(codes[i]) & 0x7u;
        size_t bit = i * 3, byte = bit >> 3, off = bit & 7;
        out[byte] |= static_cast<uint8_t>((c << off) & 0xFF);
        if (off > 5) out[byte + 1] |= static_cast<uint8_t>(c >> (8 - off));
    }
    return out;
}


// ---- compressed-blob containers ---------------------------------------------
struct ValueBlob {                 // per-token path (values all tiers; CQ-8 keys)
    int T = 0, D = 0, bits = 0;
    std::vector<uint16_t> scales;  // [T] fp16 bits
    std::vector<int>      codes;   // [T*D] signed codes (row-major)
    std::vector<uint8_t>  payload; // packed byte stream
};

struct KeyBlob {                   // per-channel grouped path (CQ-4 / CQ-4+)
    int T = 0, D = 0, bits = 0, G = 0;
    std::vector<std::pair<int,int>> groups;   // [(a,b), ...]
    std::vector<int>      keep;               // non-outlier channel indices
    std::vector<int>      outlier;            // outlier channel indices (sorted)
    std::vector<uint16_t> scales;             // per group, nk fp16 each (concatenated)
    std::vector<uint8_t>  payload;            // per group, packed int4 (concatenated)
    std::vector<uint16_t> sidecar;            // [T*k] fp16 bits, t-major (outlier cols)
};

// ---- compress / decompress (single KV head, shape [T, D] row-major fp16) -----
ValueBlob compress_values(const std::vector<uint16_t>& V_f16, int T, int D, int bits);
std::vector<uint32_t> decompress_values(const ValueBlob& b);   // [T*D] fp32 bits

// ---- WHT-rotated INT3 value path (header-only; uses the ValueBlob container) ----
// compress: per token, rotate the fp16 row (raw WHT) then per-token amax + INT3 quant.
inline ValueBlob compress_values_wht3(const std::vector<uint16_t>& V, int T, int D) {
    ValueBlob b; b.T = T; b.D = D; b.bits = 3;
    b.scales.resize(T); b.codes.resize(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; t++) {
        std::vector<uint16_t> row(V.begin() + static_cast<size_t>(t) * D,
                                  V.begin() + static_cast<size_t>(t) * D + D);
        fwht_raw_f16(row, D);
        uint16_t s = scale_from_amax(amax_bits(row, 0, 1, D), 3);
        b.scales[t] = s;
        for (int d = 0; d < D; d++)
            b.codes[static_cast<size_t>(t) * D + d] = quant_code(row[d], s, 3);
    }
    b.payload = pack_int3(b.codes);
    return b;
}

// decompress: dequant each rotated code to fp16, inverse WHT (raw), then x(1/D) -> fp32.
inline std::vector<uint32_t> decompress_values_wht3(const ValueBlob& b) {
    std::vector<uint32_t> out(static_cast<size_t>(b.T) * b.D);
    const double inv_d = 1.0 / static_cast<double>(b.D);           // exact (D is 2^k)
    for (int t = 0; t < b.T; t++) {
        std::vector<uint16_t> r(b.D);
        for (int d = 0; d < b.D; d++) {
            int code = b.codes[static_cast<size_t>(t) * b.D + d];
            r[d] = real_to_f16(static_cast<double>(code) * f16_to_real(b.scales[t]));
        }
        fwht_raw_f16(r, b.D);
        for (int d = 0; d < b.D; d++)
            out[static_cast<size_t>(t) * b.D + d] = real_to_f32(f16_to_real(r[d]) * inv_d);
    }
    return out;
}

KeyBlob compress_keys(const std::vector<uint16_t>& K_f16, int T, int D, int bits,
                      int G, const std::vector<int>& outlier_idx);
std::vector<uint32_t> decompress_keys(const KeyBlob& b);        // [T*D] fp32 bits

}} // namespace lhsi::cq

#endif // CHANNELQUANT_REF_HPP
