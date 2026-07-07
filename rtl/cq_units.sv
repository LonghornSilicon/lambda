// cq_units.sv — ChannelQuant datapath compute cores (combinational).
//
// These are the numerically-critical primitives of the revamped KVE datapath,
// implementing docs/HW_CONTRACT.md §1-§4 exactly. They replace the TurboQuant+
// quantizer/decompressor cores and back the amax_unit / scale_bank scale math.
// Bit-exactness vs the golden vectors is proven by tb_channelquant.sv, which
// instantiates each of these and drives them with the golden inputs.
//
// Numeric model: behavioral fp via cq_fp_pkg (iverilog 64-bit `real`), which is
// bit-exact to the numpy reference (see cq_fp_pkg.sv header). The fixed-point /
// fp16-hardware lowering (an actual fp16 reciprocal+rounder, residual-buffer
// SRAM, scale-bank double-buffering) is the synthesis phase (TEARDOWN.md P4);
// these cores pin *what* that hardware must compute, bit-for-bit.

`include "cq_fp_pkg.sv"
`default_nettype none

// ---- amax -> per-axis fp16 scale (contract §1) ------------------------------
//   s = max(amax / qmax(bits), EPS) cast to fp16.
// `amax_f16` is the max-absolute over the active axis (a fp16 value: the largest
// |element|, sign cleared). Driven by the amax reduction in amax_unit/TB.
module cq_scale_unit (
    input  wire [15:0] amax_f16,
    input  wire [3:0]  bits,        // 4 or 8
    output reg  [15:0] scale_f16
);
  import cq_fp_pkg::*;
  localparam real EPS = 2.0 ** -14;
  real a, s;
  int  qmax;
  always @* begin
    a    = f16_to_real(amax_f16);
    qmax = (1 << (bits - 1)) - 1;
    s    = a / qmax;
    if (s < EPS) s = EPS;
    scale_f16 = real_to_f16(s);
  end
endmodule

// ---- quantizer: x (fp16) by fp16 scale -> signed int code (contract §1) -----
//   q = clamp( round_half_to_even(x / s), qmin, qmax )
module cq_quant_unit (
    input  wire [15:0]      x_f16,
    input  wire [15:0]      scale_f16,
    input  wire [3:0]       bits,
    output reg  signed [7:0] code
);
  import cq_fp_pkg::*;
  real    x, s;
  int     qmax, qmin;
  longint r;
  always @* begin
    x    = f16_to_real(x_f16);
    s    = f16_to_real(scale_f16);
    qmax = (1 << (bits - 1)) - 1;
    qmin = -(1 << (bits - 1));
    r    = srint_ll(x / s);
    if (r > qmax) r = qmax;
    if (r < qmin) r = qmin;
    code = r[7:0];                 // two's-complement, low `bits` are the datum
  end
endmodule

// ---- dequantizer: code * fp16 scale -> fp32 reconstruction (contract §1) -----
module cq_dequant_unit (
    input  wire signed [7:0] code,
    input  wire [15:0]       scale_f16,
    output reg  [31:0]       xhat_f32
);
  import cq_fp_pkg::*;
  real s, hat;
  always @* begin
    s        = f16_to_real(scale_f16);
    hat      = code * s;
    xhat_f32 = real_to_f32(hat);
  end
endmodule

// ---- int4 nibble packer (contract §5): element 2i -> low, 2i+1 -> high -------
module cq_pack2 (
    input  wire signed [7:0] c_lo,    // element 2i
    input  wire signed [7:0] c_hi,    // element 2i+1 (zero if odd tail)
    output wire [7:0]        byte_out
);
  assign byte_out = {c_hi[3:0], c_lo[3:0]};
endmodule

`default_nettype wire
