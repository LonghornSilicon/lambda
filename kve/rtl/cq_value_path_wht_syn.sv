// cq_value_path_wht_syn.sv — SYNTHESIZABLE Path-B streaming VALUE path for CQ-3-rot.
//
// Bit-exact synthesis-tier twin of cq_value_path_wht (kve/rtl/cq_value_path_wht.sv):
// same ports, same numeric contract, but NO `real` math — it wires the already-verified
// synthesizable cores (wht_unit_syn / cq_scale_unit_syn / cq_quant_comb / cq_dequant_unit_syn)
// plus a fixed-function fp32->fp16 round-half-even converter for the read-side rotated dequant.
// So a full-chip yosys synthesis (GF180) can elaborate the KVE value path.
//
//   forward WHT (wht_unit_syn) -> per-token amax (|rot| max) -> INT3 scale (cq_scale_unit_syn,
//   bits=3) -> per-channel INT3 codes (cq_quant_comb, bits=3). Read side: dequant the requested
//   channel to fp32 EXACTLY (cq_dequant_unit_syn; product code*scale fits fp32) then round that
//   exact fp32 to fp16 (round-half-to-even) — equals the behavioral cq_dequant_f16 oracle
//   (real_to_f16(real(code)*f16_to_real(scale))) with no double rounding.
// (WHT value rotation: Abhiram Bandi + Chaithu Talasila.)
`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// fp32 -> fp16, round-half-to-even. Bit-exact to cq_fp_pkg::real_to_f16 applied
// to the exact value f32_to_real(w). Inputs here are exact products so there is
// no double-rounding. Handles zero / normal / subnormal / (defensive) overflow.
// ---------------------------------------------------------------------------
module f32_to_f16_rne (
    input  wire [31:0] w,
    output reg  [15:0] h
);
    wire        s   = w[31];
    wire [7:0]  e32 = w[30:23];
    wire [22:0] m32 = w[22:0];

    // MSB index of a subnormal-fp32 mantissa — own always block so the loop var never latches.
    integer   i;
    reg [4:0] p;
    always @* begin
        p = 5'd0;
        for (i = 0; i < 23; i = i + 1) if (m32[i]) p = i[4:0];
    end

    // Normalized form: value = (-1)^s * sig * 2^(E-23), sig in [2^23, 2^24) (MSB@bit23).
    reg  [23:0]       sig;
    reg  signed [9:0] E;        // unbiased exponent of the leading one
    reg               is_zero;
    always @* begin
        is_zero = 1'b0;
        sig     = 24'd0;
        E       = 10'sd0;
        if (e32 == 8'd0) begin
            if (m32 == 23'd0) begin
                is_zero = 1'b1;
            end else begin
                sig = {1'b0, m32} << (5'd23 - p);
                E   = $signed({5'b0, p}) - 10'sd149;
            end
        end else begin
            sig = {1'b1, m32};
            E   = $signed({2'b0, e32}) - 10'sd127;
        end
    end

    // rounding
    reg  [10:0]       keep;
    reg               gbit, sbit, lsb, rup;
    reg  [11:0]       mr;
    reg  signed [9:0] Ee;
    reg  [5:0]        biased;
    reg  signed [9:0] rsh_s;
    reg  [5:0]        rsh;
    reg  [23:0]       smask;
    reg  [10:0]       m10;
    reg               gsub, ssub, rupsub;
    reg  [11:0]       m10r;
    always @* begin
        // default assignments (latch-free)
        keep=11'd0; gbit=1'b0; sbit=1'b0; lsb=1'b0; rup=1'b0; mr=12'd0;
        Ee=10'sd0; biased=6'd0; rsh_s=10'sd0; rsh=6'd0; smask=24'd0;
        m10=11'd0; gsub=1'b0; ssub=1'b0; rupsub=1'b0; m10r=12'd0;
        if (is_zero) begin
            h = {s, 15'b0};
        end else if (E >= -10'sd14) begin
            // ---- fp16 normal ----
            keep = sig[23:13];
            gbit = sig[12];
            sbit = |sig[11:0];
            lsb  = keep[0];
            rup  = gbit & (sbit | lsb);
            mr   = {1'b0, keep} + (rup ? 12'd1 : 12'd0);
            Ee   = E;
            if (mr[11]) begin           // carried to 2^11 -> renormalize
                mr = mr >> 1;           // 0x400
                Ee = E + 10'sd1;
            end
            biased = Ee[5:0] + 6'd15;   // Ee>=-14 -> biased>=1
            if (biased >= 6'd31) h = {s, 5'b11111, 10'b0};   // overflow -> inf (defensive)
            else                 h = {s, biased[4:0], mr[9:0]};
        end else begin
            // ---- fp16 subnormal (value < 2^-14): round to k*2^-24 ----
            rsh_s = -E - 10'sd1;        // >= 14
            rsh   = rsh_s[5:0];
            if (rsh_s > 10'sd24) begin
                h = {s, 15'b0};         // underflow to signed zero
            end else begin
                m10   = sig >> rsh;                 // 0..1024
                gsub  = sig[rsh - 6'd1];
                smask = (24'd1 << (rsh - 6'd1)) - 24'd1;
                ssub  = |(sig & smask);
                rupsub= gsub & (ssub | m10[0]);
                m10r  = {1'b0, m10} + (rupsub ? 12'd1 : 12'd0);
                if (m10r[10]) h = {s, 5'b00001, 10'b0};       // rounded up to smallest normal
                else          h = {s, 5'b00000, m10r[9:0]};
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
module cq_value_path_wht_syn #(parameter int D = 128, parameter int DW = 16)(
    input  wire [D*DW-1:0]        in_vec,       // one token, original space (fp16)
    output wire [D*8-1:0]         out_codes,    // rotated INT3 codes (signed 8b/lane)
    output wire [DW-1:0]          out_scale,    // fp16 per-token scale
    // read side: rotated fp16 reconstruction of channel dec_idx (inverse is external)
    input  wire [D*8-1:0]         dec_codes,
    input  wire [DW-1:0]          dec_scale,
    input  wire [$clog2(D)-1:0]   dec_idx,
    output wire [DW-1:0]          dec_rot_f16
);
    // forward WHT (synthesizable butterfly)
    wire [D*DW-1:0] rot;
    wht_unit_syn #(.D(D), .DW(DW)) u_fwd (.in_vec(in_vec), .out_vec(rot));

    // per-token amax = max |rot| over channels (synth-friendly combinational max — copied
    // verbatim from the behavioral oracle; magnitude via sign-bit mask)
    reg [DW-1:0] amax; integer k;
    always @* begin
        amax = 16'h0000;
        for (k = 0; k < D; k = k + 1)
            if ((rot[k*DW +: DW] & 16'h7FFF) > (amax & 16'h7FFF)) amax = rot[k*DW +: DW] & 16'h7FFF;
    end

    // INT3 scale
    wire [DW-1:0] scale;
    cq_scale_unit_syn u_sc (.amax_f16(amax), .bits(4'd3), .scale_f16(scale));
    assign out_scale = scale;

    // per-channel INT3 quant
    genvar d;
    generate
        for (d = 0; d < D; d = d + 1) begin: g_q
            wire signed [7:0] c;
            cq_quant_comb u_q (.x_f16(rot[d*DW +: DW]), .scale_f16(scale), .bits(4'd3), .code(c));
            assign out_codes[d*8 +: 8] = c;
        end
    endgenerate

    // read side: rotated fp16 dequant of the requested channel (NO inverse WHT — Path B external).
    // exact fp32 product (cq_dequant_unit_syn) then round-half-even to fp16.
    wire signed [7:0] dcode = $signed(dec_codes[dec_idx*8 +: 8]);
    wire [31:0]       dhat_f32;
    cq_dequant_unit_syn u_dq (.code(dcode), .scale_f16(dec_scale), .xhat_f32(dhat_f32));
    f32_to_f16_rne u_r16 (.w(dhat_f32), .h(dec_rot_f16));
endmodule

`default_nettype wire
