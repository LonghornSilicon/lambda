// cq_value_path_wht.sv — Path-B streaming VALUE path for CQ-3-rot.
//
// KVE side of Path B: forward-WHT the token row, per-token amax + INT3 quant -> store
// rotated codes + fp16 scale. On read it emits the ROTATED fp16 reconstruction per channel
// (dec_rot_f16) — the inverse WHT + x(1/D) is applied ONCE on the P·V output downstream
// (wht_inverse_out on MatE), not here. Chaining cq_value_path_wht -> wht_inverse_out equals
// the full reference codec (cq_wht_value / channelquant_ref.hpp). (Abhiram Bandi + Chaithu.)
`timescale 1ns/1ps
module cq_value_path_wht #(parameter int D = 128, parameter int DW = 16)(
    input  wire [D*DW-1:0]        in_vec,       // one token, original space (fp16)
    output wire [D*8-1:0]         out_codes,    // rotated INT3 codes (signed 8b/lane)
    output wire [DW-1:0]          out_scale,    // fp16 per-token scale
    // read side: rotated fp16 reconstruction of channel dec_idx (inverse is external)
    input  wire [D*8-1:0]         dec_codes,
    input  wire [DW-1:0]          dec_scale,
    input  wire [$clog2(D)-1:0]   dec_idx,
    output wire [DW-1:0]          dec_rot_f16
);
    import cq_fp_pkg::*;
    wire [D*DW-1:0] rot;
    wht_unit #(.D(D), .DW(DW)) u_fwd (.in_vec(in_vec), .out_vec(rot));
    reg [DW-1:0] amax; integer k;
    always @* begin
        amax = 16'h0000;
        for (k = 0; k < D; k = k + 1)
            if ((rot[k*DW +: DW] & 16'h7FFF) > (amax & 16'h7FFF)) amax = rot[k*DW +: DW] & 16'h7FFF;
    end
    wire [DW-1:0] scale;
    cq_scale_unit u_sc (.amax_f16(amax), .bits(4'd3), .scale_f16(scale));
    assign out_scale = scale;
    genvar d;
    generate
        for (d = 0; d < D; d = d + 1) begin: g_q
            wire signed [7:0] c;
            cq_quant_unit u_q (.x_f16(rot[d*DW +: DW]), .scale_f16(scale), .bits(4'd3), .code(c));
            assign out_codes[d*8 +: 8] = c;
        end
    endgenerate
    // rotated fp16 dequant of the requested channel (NO inverse WHT — Path B external)
    assign dec_rot_f16 = cq_dequant_f16($signed(dec_codes[dec_idx*8 +: 8]), dec_scale);
endmodule
