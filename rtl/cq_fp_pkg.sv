// cq_fp_pkg.sv — IEEE-754 half/single helpers for ChannelQuant parity
//
// Part of the TurboQuant+ -> ChannelQuant revamp. These are *behavioral* helpers
// used by the ChannelQuant datapath model and the parity testbench to reproduce
// the reference codec (reference/channelquant_ref.py) BIT-EXACTLY against the
// golden vectors. They lean on iverilog's 64-bit `real` ($realtobits/$bitstoreal);
// iverilog has no $shortrealtobits, so fp16/fp32 round + expand are done by hand.
//
// Why double is sufficient for bit-exact parity (verified empirically over all 9
// golden vectors, 2026-06-22):
//   * quant divide  q = rint(x/s): float32 and float64 give identical codes
//     (max|Δcode| = 0 across every vector) — the inputs are fp16 and |q|<=127.
//   * dequant x_hat = q*s: q (<=8b) * s_fp16 (<=11b mantissa) needs <=19 bits, so
//     the product is EXACT in float32 and double alike (max|Δ| = 0).
// So the only rounding the model must reproduce is the fp16 cast of the scale
// (contract §1) and reading the fp16/fp32 golden words. The rest stays in double.

`ifndef CQ_FP_PKG_SV
`define CQ_FP_PKG_SV

package cq_fp_pkg;

  // ---- fp16 (binary16) bit pattern -> real (exact; fp16 subset of double) ----
  function automatic real f16_to_real(input logic [15:0] h);
    int unsigned sign, exp, man;
    real m, val;
    begin
      sign = h[15];
      exp  = h[14:10];
      man  = h[9:0];
      if (exp == 0) begin
        // zero or subnormal: value = man * 2^-24
        val = man * (2.0 ** -24);
      end else if (exp == 5'h1F) begin
        // inf/nan — golden vectors never contain these; map to a large sentinel
        val = (man == 0) ? 1.0e30 : 0.0;
      end else begin
        m   = 1.0 + man * (2.0 ** -10);          // implicit leading 1
        val = m * (2.0 ** (int'(exp) - 15));     // bias 15
      end
      f16_to_real = sign ? -val : val;
    end
  endfunction

  // ---- fp32 (binary32) bit pattern -> real (exact; fp32 subset of double) -----
  function automatic real f32_to_real(input logic [31:0] w);
    int unsigned sign, exp, man;
    real m, val;
    begin
      sign = w[31];
      exp  = w[30:23];
      man  = w[22:0];
      if (exp == 0) begin
        val = man * (2.0 ** -149);               // subnormal
      end else if (exp == 8'hFF) begin
        val = (man == 0) ? 1.0e30 : 0.0;
      end else begin
        m   = 1.0 + man * (2.0 ** -23);
        val = m * (2.0 ** (int'(exp) - 127));    // bias 127
      end
      f32_to_real = sign ? -val : val;
    end
  endfunction

  // ---- real -> fp16 bit pattern, round-half-to-even (numpy .astype(float16)) --
  // Used to cast the per-axis scale to fp16 (contract §1). Inputs here are always
  // finite, positive, normal-range scales (>= EPS = 2^-14), so the normal path is
  // the only one exercised; subnormal/overflow handled defensively.
  function automatic logic [15:0] real_to_f16(input real x);
    int          sign;
    real         ax, frac;
    int          e;            // unbiased exponent
    int          biased;
    longint      mant;         // 11-bit significand (incl. implicit 1) pre-round
    longint      round_bits;
    real         scaled;
    logic [15:0] out;
    begin
      if (x == 0.0) return 16'h0000;
      sign = (x < 0.0) ? 1 : 0;
      ax   = (x < 0.0) ? -x : x;

      // normalize: find e such that 1.0 <= ax / 2^e < 2.0
      e = 0;
      if (ax >= 1.0) begin
        while (ax / (2.0 ** (e+1)) >= 1.0) e = e + 1;
      end else begin
        while (ax / (2.0 ** e) < 1.0) e = e - 1;
      end
      biased = e + 15;

      if (biased >= 31) begin                    // overflow -> inf
        out = {sign[0], 5'h1F, 10'h000};
        return out;
      end else if (biased <= 0) begin
        // subnormal: value = m * 2^-24, round-half-even m = round(ax / 2^-24)
        scaled = ax / (2.0 ** -24);
        mant   = rint_ll(scaled);
        if (mant >= 1024) begin                  // rounded up into normal range
          out = {sign[0], 5'h01, 10'h000};
          return out;
        end
        out = {sign[0], 5'h00, mant[9:0]};
        return out;
      end

      // normal: significand = ax / 2^e in [1,2); stored mantissa = round((sig-1)*2^10)
      scaled     = (ax / (2.0 ** e)) * (2.0 ** 10);  // in [2^10, 2^11)
      mant       = rint_ll(scaled);                  // round-half-even
      // mant in [1024, 2048]; subtract implicit 1<<10
      if (mant >= 2048) begin                        // carried to next exponent
        biased = biased + 1;
        mant   = 1024;
        if (biased >= 31) begin
          out = {sign[0], 5'h1F, 10'h000};
          return out;
        end
      end
      out = {sign[0], biased[4:0], mant[9:0] - 10'h000 /*low 10 of (mant-1024)*/};
      // mant-1024 fits in 10 bits since mant in [1024,2047]
      out[9:0] = (mant - 1024);
      real_to_f16 = out;
    end
  endfunction

  // ---- real -> fp32 bit pattern, round-half-to-even -------------------------
  // Used by the dequant unit to emit x_hat = q*s as an IEEE binary32 word for a
  // bit-exact compare against expected_*_hat.f32. For ChannelQuant dequant the
  // product is exactly representable (<=19 significant bits), so rounding here is
  // exact; the round-half-even path is implemented generally for safety.
  function automatic logic [31:0] real_to_f32(input real x);
    int          sign;
    real         ax, scaled;
    int          e, biased;
    longint      mant;
    logic [31:0] out;
    begin
      if (x == 0.0) return 32'h0000_0000;
      sign = (x < 0.0) ? 1 : 0;
      ax   = (x < 0.0) ? -x : x;

      e = 0;
      if (ax >= 1.0) begin
        while (ax / (2.0 ** (e+1)) >= 1.0) e = e + 1;
      end else begin
        while (ax / (2.0 ** e) < 1.0) e = e - 1;
      end
      biased = e + 127;

      if (biased >= 255) begin                       // overflow -> inf
        return {sign[0], 8'hFF, 23'h0};
      end else if (biased <= 0) begin                // subnormal
        scaled = ax / (2.0 ** -149);
        mant   = rint_ll(scaled);
        if (mant >= (longint'(1) << 23))
          return {sign[0], 8'h01, 23'h0};
        out = {sign[0], 8'h00, mant[22:0]};
        return out;
      end

      scaled = (ax / (2.0 ** e)) * (2.0 ** 23);      // in [2^23, 2^24)
      mant   = rint_ll(scaled);                      // round-half-even
      if (mant >= (longint'(1) << 24)) begin         // carry into next exponent
        biased = biased + 1;
        mant   = (longint'(1) << 23);
        if (biased >= 255) return {sign[0], 8'hFF, 23'h0};
      end
      out        = {sign[0], biased[7:0], 23'h0};
      out[22:0]  = (mant - (longint'(1) << 23));     // strip implicit 1
      real_to_f32 = out;
    end
  endfunction

  // round-half-to-even of a non-negative real to a longint (numpy rint)
  function automatic longint rint_ll(input real x);
    real     f, diff;
    longint  fi;
    begin
      f    = $floor(x);
      fi   = longint'(f);
      diff = x - f;
      if (diff < 0.5)        rint_ll = fi;
      else if (diff > 0.5)   rint_ll = fi + 1;
      else                   rint_ll = (fi % 2 == 0) ? fi : fi + 1;  // tie -> even
    end
  endfunction

  // signed round-half-to-even of a real to a longint (handles negatives)
  // ---- fp16 add/sub (round-half-even) — the WHT butterfly ops -----------------
  // Match channelquant_ref.hpp: real_to_f16(exact double sum). Behavioral core;
  // the synthesizable fp16 adder is the synthesis tier (cf. cq_units_syn.sv).
  function automatic logic [15:0] fp16_add(input logic [15:0] a, input logic [15:0] b);
    fp16_add = real_to_f16(f16_to_real(a) + f16_to_real(b));
  endfunction
  function automatic logic [15:0] fp16_sub(input logic [15:0] a, input logic [15:0] b);
    fp16_sub = real_to_f16(f16_to_real(a) - f16_to_real(b));
  endfunction

  // dequant straight to fp16 (WHT inverse input): real_to_f16(code * f16_to_real(scale)).
  function automatic logic [15:0] cq_dequant_f16(input logic signed [7:0] code,
                                                 input logic [15:0] scale_f16);
    cq_dequant_f16 = real_to_f16(real'(code) * f16_to_real(scale_f16));
  endfunction

  function automatic longint srint_ll(input real x);
    begin
      srint_ll = (x < 0.0) ? -rint_ll(-x) : rint_ll(x);
    end
  endfunction

endpackage

`endif
