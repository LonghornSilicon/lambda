// rope.sv
//
// Synthesizable VecU rotary-position-embedding (RoPE) slice of the
// LonghornSilicon "Lambda" vector unit. Rotates each (even, odd) channel pair of
// a Q or K vector by the angle theta_i * pos:
//
//     x'_2i   = x_2i * cos(theta_i*pos) - x_2i+1 * sin(theta_i*pos)
//     x'_2i+1 = x_2i * sin(theta_i*pos) + x_2i+1 * cos(theta_i*pos)
//
// It models the ACTUAL hardware algorithm (architecture/dataflow_walkthrough §VecU;
// the codebook ROM holds the RoPE sin/cos tables), NOT an exact float rotation:
//   - a sin/cos LUT (ROM) indexed by (position, channel-pair). For head dim H there
//     are H/2 pairs i=0..H/2-1 with theta_i = 10000^(-2i/H); the ROM stores
//     COS[pos][i]=round_fp16(cos(pos*theta_i)), SIN[pos][i]=round_fp16(sin(...)).
//     The fp16 rounding of cos/sin IS the LUT approximation (see rope_ref.py).
//   - each product is an fp32 multiply and each pair-sum an fp32 add/sub, rounded
//     ONCE to fp16 on emit — the same IEEE datapath as mate_pv_fp16 / vecu_softmax,
//     so the error is the cos/sin LUT quantization, not fp16 product drift.
//   Bit-exact to sw/reference_model/rope_ref.py (see docs/rope_rtl.md).
//
// INTERFACE (house streaming style):
//   LOAD  — present one fp16 channel per clock on s_data with s_valid=1; assert
//           s_last=1 on the final channel (HEAD_DIM channels total). `pos` is
//           sampled on the FIRST channel of the vector (constant for the vector).
//   EMIT  — after the pipeline finishes, the block streams the HEAD_DIM rotated
//           channels: y_valid pulses each emit cycle with the fp16 value on y_data,
//           y_last on the final one. The consumer waits on the y_valid handshake,
//           so the (data-independent) latency is transparent.
//   busy is high from the first channel until the last rotated channel is emitted.
//
// MICRO-SEQUENCED: channels are buffered on load, then the rotation runs as a
// micro-sequence executing at most one fp32 op per register-to-register path (a few
// independent ops run in parallel per cycle, like vecu_softmax), so the GF180 ss
// corner closes with normal resizing. No latches.

`timescale 1ns/1ps

module rope #(
    parameter integer HEAD_DIM = 8,       // channels per Q/K vector (even)
    parameter integer MAX_POS  = 16,      // LUT positions 0..MAX_POS-1
    parameter integer FW       = 16       // fp16 width
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     s_valid,   // a channel is being presented
    input  wire [FW-1:0]            s_data,    // fp16 Q/K channel
    input  wire                     s_last,    // last channel of the vector
    input  wire [$clog2(MAX_POS)-1:0] pos,     // sequence position (sampled on first channel)

    output reg                      y_valid,   // a rotated channel is being emitted
    output reg  [FW-1:0]            y_data,    // fp16 rotated channel
    output reg                      y_last,    // last channel of the vector
    output reg                      busy
);

    localparam integer D_PAIRS = HEAD_DIM / 2;
    localparam integer CW      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM + 1);
    localparam integer PW      = (D_PAIRS  <= 1) ? 1 : $clog2(D_PAIRS + 1);
    localparam integer POSW    = $clog2(MAX_POS);

    // =======================================================================
    // fp16 -> fp32 widen (exact) — same datapath as vecu_softmax / mate_pv_fp16
    // =======================================================================
    function automatic [31:0] fp16_to_fp32;
        input [15:0] h;
        reg s; reg [4:0] e; reg [9:0] m; integer e2, jj; reg [10:0] mm; reg [7:0] eo;
        begin
            e2 = 0; jj = 0; mm = 11'b0; eo = 8'b0;
            s = h[15]; e = h[14:10]; m = h[9:0];
            if (e == 5'h1F)      fp16_to_fp32 = {s, 8'hFF, m, 13'b0};
            else if (e == 5'h0) begin
                if (m == 10'b0)  fp16_to_fp32 = {s, 31'b0};
                else begin
                    e2 = -14; mm = {1'b0, m};
                    for (jj = 0; jj < 10; jj = jj + 1)
                        if (!mm[10]) begin mm = mm << 1; e2 = e2 - 1; end
                    eo = (e2 + 127);
                    fp16_to_fp32 = {s, eo, mm[9:0], 13'b0};
                end
            end else begin
                eo = ({3'b0, e} - 8'd15 + 8'd127);
                fp16_to_fp32 = {s, eo, m, 13'b0};
            end
        end
    endfunction

    // =======================================================================
    // fp32 add / sub (correctly-rounded RTNE) — the MatE fp32 adder
    // =======================================================================
    function automatic [31:0] fp32_add;
        input [31:0] a;
        input [31:0] b;
        reg        sa, sb, sbig, ssmall, sres;
        reg [7:0]  ea, eb;
        reg [22:0] ma, mb;
        reg        a_nan, b_nan, a_inf, b_inf;
        reg [23:0] siga, sigb;
        integer    eea, eeb, E, d, s, msbp, want, avail, sh;
        reg [27:0] big, small0, small_sh, summ;
        reg        dropped, guard, roundb, sticky, roundup;
        reg [24:0] kept;
        reg [7:0]  EF;
        begin
            sa = a[31]; ea = a[30:23]; ma = a[22:0];
            sb = b[31]; eb = b[30:23]; mb = b[22:0];
            a_nan = (ea == 8'hFF) && (|ma);  a_inf = (ea == 8'hFF) && (~|ma);
            b_nan = (eb == 8'hFF) && (|mb);  b_inf = (eb == 8'hFF) && (~|mb);
            if (a_nan || b_nan)        fp32_add = 32'h7FC00000;
            else if (a_inf && b_inf)   fp32_add = (sa == sb) ? a : 32'h7FC00000;
            else if (a_inf)            fp32_add = a;
            else if (b_inf)            fp32_add = b;
            else begin
                siga = (ea == 8'h0) ? {1'b0, ma} : {1'b1, ma};
                sigb = (eb == 8'h0) ? {1'b0, mb} : {1'b1, mb};
                eea  = (ea == 8'h0) ? 1 : ea;
                eeb  = (eb == 8'h0) ? 1 : eb;
                if (eea > eeb || (eea == eeb && siga >= sigb)) begin
                    E = eea; d = eea - eeb; big = {siga, 3'b0}; small0 = {sigb, 3'b0}; sbig = sa; ssmall = sb;
                end else begin
                    E = eeb; d = eeb - eea; big = {sigb, 3'b0}; small0 = {siga, 3'b0}; sbig = sb; ssmall = sa;
                end
                if (d == 0)          small_sh = small0;
                else if (d > 27)     small_sh = (|small0) ? 28'b1 : 28'b0;
                else begin
                    small_sh = small0 >> d;
                    if (|(small0 & ((28'b1 << d) - 28'b1))) small_sh[0] = 1'b1;
                end
                sres = sbig;
                if (sbig == ssmall) summ = big + small_sh;
                else                summ = big - small_sh;
                if (summ == 28'b0)   fp32_add = 32'h00000000;
                else begin
                    if (summ[27]) begin
                        dropped = summ[0]; summ = summ >> 1; summ[0] = summ[0] | dropped; E = E + 1;
                    end
                    msbp = 0;
                    for (s = 0; s < 27; s = s + 1) if (summ[s]) msbp = s;
                    want  = 26 - msbp; avail = E - 1;
                    sh    = (want <= avail) ? want : avail;
                    summ  = summ << sh; E = E - sh;
                    kept   = {1'b0, summ[26:3]};
                    guard  = summ[2]; roundb = summ[1]; sticky = summ[0];
                    roundup = guard & (roundb | sticky | kept[0]);
                    kept   = kept + {24'b0, roundup};
                    if (kept[24]) begin kept = kept >> 1; E = E + 1; end
                    if (E >= 255) fp32_add = {sres, 8'hFF, 23'b0};
                    else begin
                        EF = kept[23] ? E[7:0] : 8'h0;
                        fp32_add = {sres, EF, kept[22:0]};
                    end
                end
            end
        end
    endfunction

    function automatic [31:0] fp32_sub;
        input [31:0] a;
        input [31:0] b;
        begin fp32_sub = fp32_add(a, b ^ 32'h80000000); end
    endfunction

    // =======================================================================
    // fp32 multiply (correctly-rounded RTNE)
    // =======================================================================
    function automatic [31:0] fp32_mul;
        input [31:0] a;
        input [31:0] b;
        reg        sa, sb, sy;
        reg [7:0]  ea, eb;
        reg [22:0] ma, mb;
        reg        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
        reg [23:0] sig_a, sig_b;
        integer    Ea, Eb, Ep, msb, exp, sh, k;
        reg [47:0] P;
        reg        guard, sticky;
        reg [24:0] sig;
        begin
            sa = a[31]; ea = a[30:23]; ma = a[22:0];
            sb = b[31]; eb = b[30:23]; mb = b[22:0];
            sy = sa ^ sb;
            a_nan = (ea == 8'hFF) && (|ma);  a_inf = (ea == 8'hFF) && (~|ma);  a_zero = (ea == 8'h0) && (~|ma);
            b_nan = (eb == 8'hFF) && (|mb);  b_inf = (eb == 8'hFF) && (~|mb);  b_zero = (eb == 8'h0) && (~|mb);
            if (a_nan || b_nan)                                    fp32_mul = 32'h7FC00000;
            else if (a_inf || b_inf) begin
                if ((a_inf && b_zero) || (b_inf && a_zero))        fp32_mul = 32'h7FC00000;
                else                                              fp32_mul = {sy, 8'hFF, 23'b0};
            end
            else if (a_zero || b_zero)                             fp32_mul = {sy, 31'b0};
            else begin
                if (ea == 8'h0) begin
                    sig_a = ma; Ea = -149;
                    for (k = 0; k < 24; k = k + 1) if (!sig_a[23]) begin sig_a = sig_a << 1; Ea = Ea - 1; end
                end else begin sig_a = {1'b1, ma}; Ea = ea - 150; end
                if (eb == 8'h0) begin
                    sig_b = mb; Eb = -149;
                    for (k = 0; k < 24; k = k + 1) if (!sig_b[23]) begin sig_b = sig_b << 1; Eb = Eb - 1; end
                end else begin sig_b = {1'b1, mb}; Eb = eb - 150; end
                P   = sig_a * sig_b;
                Ep  = Ea + Eb;
                msb = P[47] ? 47 : 46;
                sh  = msb - 23;
                sig    = P >> sh;
                guard  = P[sh-1];
                sticky = |(P & ((48'b1 << (sh-1)) - 48'b1));
                exp = msb + Ep + 127;
                sig = sig + {24'b0, (guard & (sticky | sig[0]))};
                if (sig[24]) begin sig = sig >> 1; exp = exp + 1; end
                if (exp >= 255)     fp32_mul = {sy, 8'hFF, 23'b0};
                else if (exp <= 0)  fp32_mul = {sy, 31'b0};
                else                fp32_mul = {sy, exp[7:0], sig[22:0]};
            end
        end
    endfunction

    // =======================================================================
    // fp32 -> fp16 (RTNE, overflow->inf, underflow->subnormal/0)
    // =======================================================================
    function automatic [15:0] fp32_to_fp16;
        input [31:0] a;
        reg        s;
        reg [7:0]  e;
        reg [22:0] m;
        reg [23:0] sig;
        integer    he, drop;
        reg [12:0] kept;
        reg        guard, sticky, roundup;
        begin
            s = a[31]; e = a[30:23]; m = a[22:0];
            if (e == 8'hFF)      fp32_to_fp16 = (|m) ? {s, 5'h1F, 10'b1000000000} : {s, 5'h1F, 10'b0};
            else if (e == 8'h0)  fp32_to_fp16 = {s, 15'b0};
            else begin
                sig = {1'b1, m};
                he  = e - 112;
                if (he >= 31) fp32_to_fp16 = {s, 5'h1F, 10'b0};
                else begin
                    drop = (he <= 0) ? (14 - he) : 13;
                    if (drop > 25) drop = 25;
                    kept    = (sig >> drop);
                    guard   = (sig >> (drop - 1)) & 1'b1;
                    sticky  = |(sig & (((32'b1 << (drop - 1)) - 32'b1)));
                    roundup = guard & (sticky | kept[0]);
                    kept    = kept + {12'b0, roundup};
                    if (he <= 0) fp32_to_fp16 = {s, 2'b00, kept};
                    else begin
                        if (kept[11]) begin he = he + 1; kept = kept >> 1; end
                        if (he >= 31) fp32_to_fp16 = {s, 5'h1F, 10'b0};
                        else          fp32_to_fp16 = {s, he[4:0], kept[9:0]};
                    end
                end
            end
        end
    endfunction

    // =======================================================================
    // sin/cos LUT (the codebook ROM RoPE table). Generated by rope_ref.py for
    // HEAD_DIM=8, MAX_POS=16, BASE=10000; index = pos*D_PAIRS + pair.
    // (Regenerate with `python rope_ref.py --emit-sv` if the geometry changes.)
    // =======================================================================
    function automatic [15:0] rope_cos;
        input integer idx;
        begin
            case (idx)
                 0: rope_cos = 16'h3C00;   1: rope_cos = 16'h3C00;   2: rope_cos = 16'h3C00;   3: rope_cos = 16'h3C00;
                 4: rope_cos = 16'h3853;   5: rope_cos = 16'h3BF6;   6: rope_cos = 16'h3C00;   7: rope_cos = 16'h3C00;
                 8: rope_cos = 16'hB6A9;   9: rope_cos = 16'h3BD7;  10: rope_cos = 16'h3C00;  11: rope_cos = 16'h3C00;
                12: rope_cos = 16'hBBEC;  13: rope_cos = 16'h3BA5;  14: rope_cos = 16'h3BFF;  15: rope_cos = 16'h3C00;
                16: rope_cos = 16'hB93B;  17: rope_cos = 16'h3B5E;  18: rope_cos = 16'h3BFE;  19: rope_cos = 16'h3C00;
                20: rope_cos = 16'h348A;  21: rope_cos = 16'h3B05;  22: rope_cos = 16'h3BFD;  23: rope_cos = 16'h3C00;
                24: rope_cos = 16'h3BAE;  25: rope_cos = 16'h3A9A;  26: rope_cos = 16'h3BFC;  27: rope_cos = 16'h3C00;
                28: rope_cos = 16'h3A08;  29: rope_cos = 16'h3A1E;  30: rope_cos = 16'h3BFB;  31: rope_cos = 16'h3C00;
                32: rope_cos = 16'hB0A8;  33: rope_cos = 16'h3993;  34: rope_cos = 16'h3BF9;  35: rope_cos = 16'h3C00;
                36: rope_cos = 16'hBB4A;  37: rope_cos = 16'h38F9;  38: rope_cos = 16'h3BF8;  39: rope_cos = 16'h3C00;
                40: rope_cos = 16'hBAB6;  41: rope_cos = 16'h3853;  42: rope_cos = 16'h3BF6;  43: rope_cos = 16'h3C00;
                44: rope_cos = 16'h1C88;  45: rope_cos = 16'h3742;  46: rope_cos = 16'h3BF4;  47: rope_cos = 16'h3C00;
                48: rope_cos = 16'h3AC0;  49: rope_cos = 16'h35CC;  50: rope_cos = 16'h3BF1;  51: rope_cos = 16'h3C00;
                52: rope_cos = 16'h3B42;  53: rope_cos = 16'h3448;  54: rope_cos = 16'h3BEF;  55: rope_cos = 16'h3C00;
                56: rope_cos = 16'h3060;  57: rope_cos = 16'h3170;  58: rope_cos = 16'h3BEC;  59: rope_cos = 16'h3C00;
                60: rope_cos = 16'hBA14;  61: rope_cos = 16'h2C87;  62: rope_cos = 16'h3BE9;  default: rope_cos = 16'h3C00;
            endcase
        end
    endfunction

    function automatic [15:0] rope_sin;
        input integer idx;
        begin
            case (idx)
                 0: rope_sin = 16'h0000;   1: rope_sin = 16'h0000;   2: rope_sin = 16'h0000;   3: rope_sin = 16'h0000;
                 4: rope_sin = 16'h3ABB;   5: rope_sin = 16'h2E64;   6: rope_sin = 16'h211F;   7: rope_sin = 16'h1419;
                 8: rope_sin = 16'h3B46;   9: rope_sin = 16'h325B;  10: rope_sin = 16'h251F;  11: rope_sin = 16'h1819;
                12: rope_sin = 16'h3084;  13: rope_sin = 16'h34BA;  14: rope_sin = 16'h27AE;  15: rope_sin = 16'h1A25;
                16: rope_sin = 16'hBA0E;  17: rope_sin = 16'h363B;  18: rope_sin = 16'h291E;  19: rope_sin = 16'h1C19;
                20: rope_sin = 16'hBBAC;  21: rope_sin = 16'h37AC;  22: rope_sin = 16'h2A66;  23: rope_sin = 16'h1D1F;
                24: rope_sin = 16'hB478;  25: rope_sin = 16'h3884;  26: rope_sin = 16'h2BAD;  27: rope_sin = 16'h1E25;
                28: rope_sin = 16'h3942;  29: rope_sin = 16'h3927;  30: rope_sin = 16'h2C7A;  31: rope_sin = 16'h1F2B;
                32: rope_sin = 16'h3BEA;  33: rope_sin = 16'h39BD;  34: rope_sin = 16'h2D1D;  35: rope_sin = 16'h2019;
                36: rope_sin = 16'h3698;  37: rope_sin = 16'h3A44;  38: rope_sin = 16'h2DC1;  39: rope_sin = 16'h209C;
                40: rope_sin = 16'hB85A;  41: rope_sin = 16'h3ABB;  42: rope_sin = 16'h2E64;  43: rope_sin = 16'h211F;
                44: rope_sin = 16'hBC00;  45: rope_sin = 16'h3B21;  46: rope_sin = 16'h2F07;  47: rope_sin = 16'h21A2;
                48: rope_sin = 16'hB84B;  49: rope_sin = 16'h3B75;  50: rope_sin = 16'h2FA9;  51: rope_sin = 16'h2225;
                52: rope_sin = 16'h36B9;  53: rope_sin = 16'h3BB5;  54: rope_sin = 16'h3026;  55: rope_sin = 16'h22A8;
                56: rope_sin = 16'h3BED;  57: rope_sin = 16'h3BE2;  58: rope_sin = 16'h3077;  59: rope_sin = 16'h232B;
                60: rope_sin = 16'h3934;  61: rope_sin = 16'h3BFB;  62: rope_sin = 16'h30C8;  default: rope_sin = 16'h23AE;
            endcase
        end
    endfunction

    // =======================================================================
    // Micro-sequenced datapath. Channels buffered on load, then per pair a short
    // micro-sequence computes the two rotated outputs (products fp32, one fp16
    // rounding on emit) and streams them out, one value/cycle.
    //
    //   step0 READ : widen x0,x1,cos,sin
    //   step1 MUL1 : pa=x0*cos, pb=x1*sin        (for y0 = pa-pb)
    //   step2 MUL2 : pc=x0*sin, pd=x1*cos ; y0f=pa-pb
    //   step3 ADD  : y1f=pc+pd
    //   step4 EMIT0: y_data=fp16(y0f)
    //   step5 EMIT1: y_data=fp16(y1f) ; advance pair / finish
    // =======================================================================
    localparam [1:0] S_IDLE = 2'd0, S_LOAD = 2'd1, S_RUN = 2'd2;
    reg [1:0]        state;
    reg [2:0]        step;
    reg [POSW-1:0]   cur_pos;
    reg [FW-1:0]     x_mem [0:HEAD_DIM-1];
    reg [CW-1:0]     wr_ptr;
    reg [PW-1:0]     pair_i;

    reg [31:0] x0_32, x1_32, c32, s32;
    reg [31:0] pa, pb, pc, pd, y0f, y1f;

    // combinational cones (each path at most one fp32 op from registered inputs)
    wire [31:0] lut_i  = cur_pos * D_PAIRS + pair_i;
    wire [31:0] r_x0   = fp16_to_fp32(x_mem[{pair_i, 1'b0}]);         // x_mem[2*pair]
    wire [31:0] r_x1   = fp16_to_fp32(x_mem[{pair_i, 1'b1}]);         // x_mem[2*pair+1]
    wire [31:0] r_cos  = fp16_to_fp32(rope_cos(lut_i));
    wire [31:0] r_sin  = fp16_to_fp32(rope_sin(lut_i));
    wire [31:0] m_pa   = fp32_mul(x0_32, c32);
    wire [31:0] m_pb   = fp32_mul(x1_32, s32);
    wire [31:0] m_pc   = fp32_mul(x0_32, s32);
    wire [31:0] m_pd   = fp32_mul(x1_32, c32);
    wire [31:0] a_y0   = fp32_sub(pa, pb);
    wire [31:0] a_y1   = fp32_add(pc, pd);
    wire [FW-1:0] w_y0 = fp32_to_fp16(y0f);
    wire [FW-1:0] w_y1 = fp32_to_fp16(y1f);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; step <= 3'd0;
            y_valid <= 1'b0; y_last <= 1'b0; busy <= 1'b0;
            wr_ptr <= {CW{1'b0}}; pair_i <= {PW{1'b0}}; cur_pos <= {POSW{1'b0}};
        end else begin
            y_valid <= 1'b0; y_last <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (s_valid) begin
                        busy    <= 1'b1;
                        cur_pos <= pos;                    // sample position on first channel
                        x_mem[0] <= s_data;
                        wr_ptr  <= { {(CW-1){1'b0}}, 1'b1 };
                        if (s_last) begin pair_i <= {PW{1'b0}}; step <= 3'd0; state <= S_RUN; end
                        else state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    if (s_valid) begin
                        x_mem[wr_ptr] <= s_data;
                        wr_ptr <= wr_ptr + 1'b1;
                        if (s_last) begin pair_i <= {PW{1'b0}}; step <= 3'd0; state <= S_RUN; end
                    end
                end
                S_RUN: begin
                    case (step)
                        3'd0: begin x0_32 <= r_x0; x1_32 <= r_x1; c32 <= r_cos; s32 <= r_sin; step <= 3'd1; end
                        3'd1: begin pa <= m_pa; pb <= m_pb; step <= 3'd2; end
                        3'd2: begin pc <= m_pc; pd <= m_pd; y0f <= a_y0; step <= 3'd3; end
                        3'd3: begin y1f <= a_y1; step <= 3'd4; end
                        3'd4: begin y_data <= w_y0; y_valid <= 1'b1; y_last <= 1'b0; step <= 3'd5; end
                        default: begin                     // step5: emit y1, advance
                            y_data <= w_y1; y_valid <= 1'b1;
                            y_last <= (pair_i == D_PAIRS - 1);
                            step   <= 3'd0;
                            if (pair_i == D_PAIRS - 1) begin state <= S_IDLE; busy <= 1'b0; end
                            else pair_i <= pair_i + 1'b1;
                        end
                    endcase
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
