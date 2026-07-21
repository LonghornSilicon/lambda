// cq_wht_value.sv — WHT-rotated INT3 value codec for one token (compress+decompress).
//
// Mirrors channelquant_ref.hpp compress_values_wht3 + decompress_values_wht3 for a single
// [D] fp16 row: forward WHT -> per-token amax -> INT3 scale/quant -> dequant to fp16 ->
// inverse WHT -> x(1/D) [exact 2^-k] -> fp32 V̂. Combinational (the bit-exact arithmetic
// proof; the streaming/pipelined schedule is a later refinement). D power of two.
// (WHT value rotation: Abhiram Bandi + Chaithu Talasila.)
`timescale 1ns/1ps

module cq_wht_value #(
    parameter int D  = 128,
    parameter int DW = 16
)(
    input  wire [D*DW-1:0] in_vec,      // D fp16 (one token, original space)
    output wire [D*32-1:0] vhat_vec     // D fp32 (reconstructed V̂)
);
    import cq_fp_pkg::*;

    // ---- forward WHT ----
    wire [D*DW-1:0] rot;
    wht_unit #(.D(D), .DW(DW)) u_fwd (.in_vec(in_vec), .out_vec(rot));

    // ---- per-token amax (max fp16 magnitude, sign cleared) ----
    reg [DW-1:0] amax;
    integer k;
    always @* begin
        amax = 16'h0000;
        for (k = 0; k < D; k = k + 1)
            if ((rot[k*DW +: DW] & 16'h7FFF) > (amax & 16'h7FFF))
                amax = rot[k*DW +: DW] & 16'h7FFF;
    end

    // ---- scale (INT3) ----
    wire [DW-1:0] scale;
    cq_scale_unit u_sc (.amax_f16(amax), .bits(4'd3), .scale_f16(scale));

    // ---- quant each rotated channel, dequant back to fp16 ----
    wire [D*DW-1:0] rhat;
    genvar d;
    generate
        for (d = 0; d < D; d = d + 1) begin: g_q
            wire signed [7:0] code;
            cq_quant_unit u_q (.x_f16(rot[d*DW +: DW]), .scale_f16(scale), .bits(4'd3), .code(code));
            assign rhat[d*DW +: DW] = cq_dequant_f16(code, scale);
        end
    endgenerate

    // ---- inverse WHT ----
    wire [D*DW-1:0] yinv;
    wht_unit #(.D(D), .DW(DW)) u_inv (.in_vec(rhat), .out_vec(yinv));

    // ---- x(1/D) exact 2^-k, emit fp32 ----
    localparam real INV_D = 1.0 / real'(D);
    generate
        for (d = 0; d < D; d = d + 1) begin: g_out
            assign vhat_vec[d*32 +: 32] = real_to_f32(f16_to_real(yinv[d*DW +: DW]) * INV_D);
        end
    endgenerate
endmodule
