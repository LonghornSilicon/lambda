// test_wht_value_ref.cpp — emit the WHT-rotated INT3 value reconstruction for a real
// Qwen value slice, so the numpy mirror can be checked bit-for-bit. (WHT value rotation:
// Abhiram Bandi + Chaithu Talasila.)
//
//   ./test_wht_value_ref  val_in.hex  vhat_out.hex
// val_in.hex : "D T bits\n" then T lines of D fp16 hex words (dump-multi format).
// vhat_out.hex : T lines of D fp32 hex words (the reconstructed V̂).
#include "channelquant_ref.hpp"
#include <cstdio>
#include <vector>
#include <cstdint>
using namespace lhsi::cq;

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s val_in.hex vhat_out.hex\n", argv[0]); return 1; }
    FILE* f = std::fopen(argv[1], "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int D = 0, T = 0, bits = 0;
    if (std::fscanf(f, "%d %d %d", &D, &T, &bits) != 3) { std::fprintf(stderr, "bad header\n"); return 1; }
    std::vector<uint16_t> V(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            unsigned x = 0; if (std::fscanf(f, "%x", &x) != 1) { std::fprintf(stderr, "short read\n"); return 1; }
            V[static_cast<size_t>(t) * D + d] = static_cast<uint16_t>(x);
        }
    std::fclose(f);

    ValueBlob b = compress_values_wht3(V, T, D);
    std::vector<uint32_t> vhat = decompress_values_wht3(b);

    FILE* o = std::fopen(argv[2], "w");
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) std::fprintf(o, "%08x ", vhat[static_cast<size_t>(t) * D + d]);
        std::fprintf(o, "\n");
    }
    std::fclose(o);
    std::fprintf(stderr, "T=%d D=%d payload_bytes=%zu bits_per_val=%.3f\n",
                 T, D, b.payload.size(), b.payload.size() * 8.0 / (static_cast<double>(T) * D));
    return 0;
}
