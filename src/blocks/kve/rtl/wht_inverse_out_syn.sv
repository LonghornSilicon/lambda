// wht_inverse_out_syn.sv — SYNTHESIZABLE Path-B output stage: inverse WHT + x(1/D).
//
// Bit-exact synthesis-tier twin of wht_inverse_out (kve/rtl/wht_inverse_out.sv): same ports,
// same numeric contract, but NO `real` math — it wires the verified synthesizable butterfly
// (wht_unit_syn) then multiplies each fp16 lane by 1/D = 2^-k (k=log2(D)) EXACTLY by adjusting
// the fp32 exponent. Since 1/D is a power of two the product needs no rounding, so the fp32
// result equals real_to_f32(f16_to_real(y)*INV_D) bit-for-bit. So a full-chip yosys synthesis
// (GF180) can elaborate the KVE inverse/output path.
// (WHT value rotation: Abhiram Bandi + Chaithu Talasila.) D power of two.
`timescale 1ns/1ps
`default_nettype none

// fp16 y -> fp32 word for value = f16_to_real(y) * 2^-K   (K = log2(D), 0..~10).
// Multiply-by-power-of-two is exact: convert the fp16 significand to fp32 and subtract K from
// the (biased) fp32 exponent. For K<=10 and fp16's smallest magnitude (2^-24) the result stays a
// normal fp32 (>= ~2^-34, well above fp32's 2^-149 subnormal floor), so no underflow occurs;
// subnormal fp32 output is handled defensively but never triggers for this datapath.
module f16_scale2mk_to_f32 #(parameter int K = 7) (
    input  wire [15:0] y,
    output reg  [31:0] w
);
    wire        s   = y[15];
    wire [4:0]  e5  = y[14:10];
    wire [9:0]  m10 = y[9:0];

    // MSB index of a subnormal-fp16 mantissa (0..9) — own always block so no loop-var latch.
    integer   i;
    reg [3:0] p;
    always @* begin
        p = 4'd0;
        for (i = 0; i < 10; i = i + 1) if (m10[i]) p = i[3:0];
    end

    reg  [23:0]        frac;      // fp32 mantissa field candidate (23 used)
    reg  signed [11:0] ue;        // unbiased exponent after -K
    reg  signed [11:0] biased;
    always @* begin
        frac   = 24'd0;
        ue     = 12'sd0;
        biased = 12'sd0;
        if (e5 == 5'd0 && m10 == 10'd0) begin
            w = {s, 31'b0};                        // zero
        end else if (e5 != 5'd0) begin
            // normal fp16: value = 1.m10 * 2^(e5-15); *2^-K -> exp = e5-15-K
            ue     = $signed({7'b0, e5}) - 12'sd15 - K;
            biased = ue + 12'sd127;
            if (biased >= 12'sd255)      w = {s, 8'hFF, 23'b0};                 // overflow (defensive)
            else if (biased >= 12'sd1)   w = {s, biased[7:0], m10, 13'b0};      // normal fp32
            else begin
                // subnormal fp32 (defensive; unreached for K<=10)
                frac = ({1'b1, m10, 13'b0}) >> (12'sd1 - biased);
                w = {s, 8'h00, frac[22:0]};
            end
        end else begin
            // subnormal fp16: value = m10 * 2^-24; *2^-K -> m10 * 2^(-24-K)
            ue     = $signed({8'b0, p}) - 12'sd24 - K;        // exponent of leading one
            biased = ue + 12'sd127;
            frac   = ({1'b0, m10, 13'b0}) << (4'd10 - p);     // drop implicit 1, left-justify
            if (biased >= 12'sd1)  w = {s, biased[7:0], frac[22:0]};
            else begin
                // subnormal fp32 (defensive; unreached)
                w = {s, 8'h00, (frac >> (12'sd1 - biased))};
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
module wht_inverse_out_syn #(
    parameter int D  = 128,
    parameter int DW = 16
)(
    input  wire [D*DW-1:0] rot_out,     // Sum_t A[t]*Vhat_rot[t] in rotated space (fp16)
    output wire [D*32-1:0] vhat_out     // attention output Vhat (fp32), rotation undone
);
    localparam int K = $clog2(D);       // 1/D = 2^-K, exact

    wire [D*DW-1:0] y;
    wht_unit_syn #(.D(D), .DW(DW)) u_inv (.in_vec(rot_out), .out_vec(y));   // self-inverse butterfly

    genvar d;
    generate
        for (d = 0; d < D; d = d + 1) begin: g_out
            f16_scale2mk_to_f32 #(.K(K)) u_sc (.y(y[d*DW +: DW]), .w(vhat_out[d*32 +: 32]));
        end
    endgenerate
endmodule

`default_nettype wire
