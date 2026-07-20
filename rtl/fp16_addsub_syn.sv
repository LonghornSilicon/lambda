// fp16_addsub_syn.sv — synthesizable IEEE-754 half-precision add/subtract.
//
// Bit-exact to the behavioral oracle cq_fp_pkg::fp16_add / fp16_sub
// (= real_to_f16(f16_to_real(a) ± f16_to_real(b)), round-half-to-even). No `real`
// math — align / add / normalize / round-half-even, subnormals handled. Synthesis-tier
// fp16 adder for the WHT butterfly (cf. cq_units_syn.sv for the base codec's cores).
// Inputs are finite (golden value tensors have no inf/nan).
`timescale 1ns/1ps

module fp16_addsub_syn (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        sub,     // 0 = a+b, 1 = a-b
    output reg  [15:0] y
);
    // ---- unpack (b's sign flipped for subtract) ----
    wire        sa = a[15];
    wire        sb = b[15] ^ sub;
    wire [4:0]  ea = a[14:10];
    wire [4:0]  eb = b[14:10];
    wire [4:0]  ea_e = (ea == 5'd0) ? 5'd1 : ea;
    wire [4:0]  eb_e = (eb == 5'd0) ? 5'd1 : eb;
    wire [10:0] sa_m = (ea == 5'd0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};
    wire [10:0] sb_m = (eb == 5'd0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

    // ---- pick big (larger magnitude): larger exp, ties by significand ----
    wire aBig = (ea_e > eb_e) || ((ea_e == eb_e) && (sa_m >= sb_m));
    wire [4:0]  eBig = aBig ? ea_e : eb_e;
    wire [10:0] mBig = aBig ? sa_m : sb_m;
    wire        sBig = aBig ? sa   : sb;
    wire [10:0] mSm  = aBig ? sb_m : sa_m;
    wire        sSm  = aBig ? sb   : sa;
    wire [4:0]  ediff = eBig - (aBig ? eb_e : ea_e);

    // ---- align small right by ediff, capturing sticky; significands <<3 (GRS) ----
    wire [13:0] bigExt = {mBig, 3'b000};
    wire [13:0] smExt0 = {mSm,  3'b000};
    reg  [13:0] smExt;
    reg         sticky;
    always @* begin
        if (ediff >= 5'd14) begin smExt = 14'd0; sticky = |smExt0; end
        else begin
            smExt  = smExt0 >> ediff;
            sticky = |(smExt0 & (({14'd0,1'b1} << ediff) - 14'd1));
        end
    end
    wire [13:0] smAln = {smExt[13:1], smExt[0] | sticky};

    // ---- add / subtract magnitudes (bigExt >= smAln always) ----
    wire        sameSign = (sBig == sSm);
    wire [14:0] mag = sameSign ? ({1'b0, bigExt} + {1'b0, smAln})
                               : ({1'b0, bigExt} - {1'b0, smAln});

    // ---- normalize: leading 1 -> bit 13; track exponent = eBig + shift ----
    reg  [13:0]       norm;
    reg  signed [8:0] exp;
    reg  [4:0]        lz;
    always @* begin
        lz = 5'd0;                                 // default (avoids a latch; reassigned below)
        if (mag[14]) begin                         // add carry-out: >>1, exp+1
            norm    = mag[14:1];
            norm[0] = norm[0] | mag[0];            // preserve sticky
            exp     = $signed({4'd0, eBig}) + 9'sd1;
        end else begin                             // subtract: normalize leading zeros
            casez (mag[13:0])                      // priority encoder (no loop -> no latch)
                14'b1?????????????: lz = 5'd0;
                14'b01????????????: lz = 5'd1;
                14'b001???????????: lz = 5'd2;
                14'b0001??????????: lz = 5'd3;
                14'b00001?????????: lz = 5'd4;
                14'b000001????????: lz = 5'd5;
                14'b0000001???????: lz = 5'd6;
                14'b00000001??????: lz = 5'd7;
                14'b000000001?????: lz = 5'd8;
                14'b0000000001????: lz = 5'd9;
                14'b00000000001???: lz = 5'd10;
                14'b000000000001??: lz = 5'd11;
                14'b0000000000001?: lz = 5'd12;
                14'b00000000000001: lz = 5'd13;
                default:            lz = 5'd14;    // all zero -> true zero result
            endcase
            norm = (lz == 5'd14) ? 14'd0 : (mag[13:0] << lz);
            exp  = $signed({4'd0, eBig}) - $signed({4'd0, lz});
        end
    end

    // ---- round-half-to-even on guard bits norm[2:0]; keep norm[13:3] ----
    wire [10:0] pre_round = norm[13:3];
    wire        round_up  = norm[2] & (norm[1] | norm[0] | pre_round[0]);
    wire [11:0] rounded   = {1'b0, pre_round} + (round_up ? 12'd1 : 12'd0);
    wire        rnd_carry = rounded[11];
    wire [10:0] mant_final = rnd_carry ? rounded[11:1] : rounded[10:0];
    wire signed [8:0] exp_adj = exp + (rnd_carry ? 9'sd1 : 9'sd0);

    // ---- subnormal repack temporaries (hoisted) ----
    reg  [7:0]  sh;
    reg  [11:0] sub_pre;
    reg         gsub, ssub;
    reg  [11:0] subm;

    // ---- repack: zero / subnormal / normal / overflow ----
    wire is_zero = (mag == 15'd0);
    always @* begin
        sh = 8'd0; sub_pre = 12'd0; gsub = 1'b0; ssub = 1'b0; subm = 12'd0;
        if (is_zero) begin
            y = 16'd0;
        end else if (exp_adj >= 9'sd31) begin
            y = {sBig, 5'b11111, 10'b0};           // overflow -> inf
        end else if (exp_adj <= 9'sd0) begin       // subnormal
            sh = 8'd1 - exp_adj[7:0];              // >= 1
            if (sh >= 8'd12) begin
                y = {sBig, 15'b0};                 // underflow -> signed zero
            end else begin
                sub_pre = {1'b0, mant_final} >> sh;
                gsub = |(({1'b0, mant_final} >> (sh - 8'd1)) & 12'd1);
                ssub = |({1'b0, mant_final} & ((12'd1 << (sh - 8'd1)) - 12'd1));
                subm = {1'b0, sub_pre[10:0]} + ((gsub & (ssub | sub_pre[0])) ? 12'd1 : 12'd0);
                y = subm[10] ? {sBig, 5'b00001, subm[9:0]}   // promoted to smallest normal
                             : {sBig, 5'b00000, subm[9:0]};
            end
        end else begin
            y = {sBig, exp_adj[4:0], mant_final[9:0]};
        end
    end
endmodule
