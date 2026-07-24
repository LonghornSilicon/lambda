// rmsnorm.sv
//
// Synthesizable VecU RMSNorm slice of the LonghornSilicon "Lambda" vector unit.
// For one D-element fp16 vector x (with a learned per-channel gain g) it emits
//
//     y_i = x_i * rsqrt(mean(x^2) + eps) * g_i
//
// It models the ACTUAL hardware algorithm (architecture/dataflow_walkthrough §VecU),
// NOT `x/np.sqrt(mean(x**2)+eps)*g`:
//   - sum-of-squares over the vector in an fp32 accumulator (each x_i widened,
//     squared with fp32_mul, summed with fp32_add) — like the MatE accumulator, so
//     the error is the rsqrt LUT, not fp16 sum drift;
//   - mean = ss * (1/D)  (1/D a folded fp32 constant), v = mean + eps;
//   - rsqrt(v) via a 64-entry LUT with linear interpolation over the reduced
//     mantissa f in [1,2): v = 2^E * f, look up rsqrt(f), then apply 2^(-floor(E/2))
//     and, when E is odd, an extra 1/sqrt(2) (rsqrt(v) = rsqrt(f) * 2^(-E/2)).
//     LUT entry R[j] = round_fp16(1/sqrt(1 + j/64)); the fp16 rounding IS the LUT
//     approximation modeled here (see rmsnorm_ref.py).
//   - y_i = round_fp16( x_i * scale * g_i ), products fp32, one fp16 rounding.
//   Bit-exact to sw/reference_model/rmsnorm_ref.py (see docs/rmsnorm_rtl.md).
//
// INTERFACE (house streaming style):
//   LOAD — present one fp16 element per clock on s_data and its gain on g_data with
//          s_valid=1; assert s_last=1 on the final element (D elements). Both are
//          buffered (depth D).
//   EMIT — after the pipeline finishes, the block streams the D normalized elements:
//          y_valid pulses each emit cycle with the fp16 value on y_data, y_last on
//          the final one. The consumer waits on the y_valid handshake.
//   busy is high from the first element until the last output is emitted.
//
// MICRO-SEQUENCED: at most one fp32 op per register-to-register path (like
// vecu_softmax), so the GF180 ss corner closes with normal resizing. No latches.

`timescale 1ns/1ps

module rmsnorm #(
    parameter integer D  = 16,     // vector length (== gain length); 1/D folded below
    parameter integer FW = 16      // fp16 width
) (
    input  wire            clk,
    input  wire            rst_n,

    input  wire            s_valid,   // an (element, gain) pair is being presented
    input  wire [FW-1:0]   s_data,    // fp16 input element x_i
    input  wire [FW-1:0]   g_data,    // fp16 learned gain g_i
    input  wire            s_last,    // last element of the vector

    output reg             y_valid,   // a normalized element is being emitted
    output reg  [FW-1:0]   y_data,    // fp16 output y_i
    output reg             y_last,    // last element of the vector
    output reg             busy
);

    localparam integer PTRW = (D <= 1) ? 1 : $clog2(D + 1);

    // fp32 constants (fold 1/D for the default D=16; regenerate INV_D32 for other D)
    localparam [31:0] INV_D32      = 32'h3D800000;  // 1/16
    localparam [31:0] EPS32        = 32'h37800000;  // 2^-16
    localparam [31:0] INV_SQRT2_32 = 32'h3F3504F3;  // 1/sqrt(2)
    localparam [15:0] RSQRT_TOP16  = 16'h39A8;      // round_fp16(1/sqrt(2)), last-interval anchor

    // =======================================================================
    // fp16 -> fp32 widen (exact)
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
    // fp32 add / sub (correctly-rounded RTNE)
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
    // fp32 -> fp16 (RTNE)
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
    // rsqrt LUT: R[j] = round_fp16(1/sqrt(1 + j/64)), j = 0..63 (over f in [1,2)).
    // Generated by rmsnorm_ref.py; regenerate with `python rmsnorm_ref.py --emit-sv`.
    // =======================================================================
    function automatic [15:0] rsqrt_lut;
        input [5:0] j;
        begin
            case (j)
                 0: rsqrt_lut = 16'h3C00;   1: rsqrt_lut = 16'h3BF0;   2: rsqrt_lut = 16'h3BE1;   3: rsqrt_lut = 16'h3BD2;
                 4: rsqrt_lut = 16'h3BC3;   5: rsqrt_lut = 16'h3BB4;   6: rsqrt_lut = 16'h3BA6;   7: rsqrt_lut = 16'h3B98;
                 8: rsqrt_lut = 16'h3B8B;   9: rsqrt_lut = 16'h3B7E;  10: rsqrt_lut = 16'h3B71;  11: rsqrt_lut = 16'h3B64;
                12: rsqrt_lut = 16'h3B57;  13: rsqrt_lut = 16'h3B4B;  14: rsqrt_lut = 16'h3B3F;  15: rsqrt_lut = 16'h3B33;
                16: rsqrt_lut = 16'h3B28;  17: rsqrt_lut = 16'h3B1C;  18: rsqrt_lut = 16'h3B11;  19: rsqrt_lut = 16'h3B06;
                20: rsqrt_lut = 16'h3AFC;  21: rsqrt_lut = 16'h3AF1;  22: rsqrt_lut = 16'h3AE7;  23: rsqrt_lut = 16'h3ADD;
                24: rsqrt_lut = 16'h3AD3;  25: rsqrt_lut = 16'h3AC9;  26: rsqrt_lut = 16'h3ABF;  27: rsqrt_lut = 16'h3AB6;
                28: rsqrt_lut = 16'h3AAC;  29: rsqrt_lut = 16'h3AA3;  30: rsqrt_lut = 16'h3A9A;  31: rsqrt_lut = 16'h3A91;
                32: rsqrt_lut = 16'h3A88;  33: rsqrt_lut = 16'h3A80;  34: rsqrt_lut = 16'h3A77;  35: rsqrt_lut = 16'h3A6F;
                36: rsqrt_lut = 16'h3A66;  37: rsqrt_lut = 16'h3A5E;  38: rsqrt_lut = 16'h3A56;  39: rsqrt_lut = 16'h3A4E;
                40: rsqrt_lut = 16'h3A47;  41: rsqrt_lut = 16'h3A3F;  42: rsqrt_lut = 16'h3A37;  43: rsqrt_lut = 16'h3A30;
                44: rsqrt_lut = 16'h3A29;  45: rsqrt_lut = 16'h3A21;  46: rsqrt_lut = 16'h3A1A;  47: rsqrt_lut = 16'h3A13;
                48: rsqrt_lut = 16'h3A0C;  49: rsqrt_lut = 16'h3A05;  50: rsqrt_lut = 16'h39FF;  51: rsqrt_lut = 16'h39F8;
                52: rsqrt_lut = 16'h39F1;  53: rsqrt_lut = 16'h39EB;  54: rsqrt_lut = 16'h39E4;  55: rsqrt_lut = 16'h39DE;
                56: rsqrt_lut = 16'h39D8;  57: rsqrt_lut = 16'h39D1;  58: rsqrt_lut = 16'h39CB;  59: rsqrt_lut = 16'h39C5;
                60: rsqrt_lut = 16'h39BF;  61: rsqrt_lut = 16'h39B9;  62: rsqrt_lut = 16'h39B4;  default: rsqrt_lut = 16'h39AE;
            endcase
        end
    endfunction

    // fp32 of frac17 / 2^17 (frac17 in [0, 2^17-1])
    function automatic [31:0] frac17_to_fp32;
        input [16:0] f;
        integer p, k; reg [7:0] exp; reg [40:0] fsh;
        begin
            if (f == 17'b0) frac17_to_fp32 = 32'b0;
            else begin
                p = 0;
                for (k = 0; k < 17; k = k + 1) if (f[k]) p = k;
                fsh = {24'b0, f} << (23 - p);
                exp = (p - 17) + 127;
                frac17_to_fp32 = {1'b0, exp, fsh[22:0]};
            end
        end
    endfunction

    // 2^n as an fp32 pattern (n modest signed)
    function automatic [31:0] pow2_fp32;
        input integer n;
        integer e;
        begin
            e = 127 + n;
            if (e >= 255)     pow2_fp32 = 32'h7F800000;
            else if (e <= 0)  pow2_fp32 = 32'b0;
            else              pow2_fp32 = {1'b0, e[7:0], 23'b0};
        end
    endfunction

    // =======================================================================
    // Micro-sequenced datapath.
    //   S_ACC  (per elem, 2 cyc): sq = x*x ; ss += sq
    //   S_PREP (rsqrt sequence)  : mean=ss/D ; v=mean+eps ; decompose ; interp ;
    //                              apply 2^(-k) [* 1/sqrt2 if E odd] -> scale
    //   S_EMIT (per elem, 3 cyc) : t1 = x*scale ; t2 = t1*g ; y = fp16(t2)
    // =======================================================================
    localparam [2:0] S_IDLE = 3'd0, S_LOAD = 3'd1, S_ACC = 3'd2, S_PREP = 3'd3, S_EMIT = 3'd4;
    reg [2:0]        state;
    reg [3:0]        step;
    reg [FW-1:0]     x_mem [0:D-1];
    reg [FW-1:0]     g_mem [0:D-1];
    reg [PTRW-1:0]   wr_ptr, ap, rd, count;

    reg [31:0] ss, mean32, v32, scale32;
    reg [31:0] acc_sq, rq_lo, rq_hi, rq_frac, rq_diff, rq_term, rq_r;
    reg signed [9:0] rq_k;
    reg        rq_odd;
    reg [31:0] et1, et2;

    // ---- ACC combinational cones ----
    wire [31:0] a_x   = fp16_to_fp32(x_mem[ap]);
    wire [31:0] a_sq  = fp32_mul(a_x, a_x);
    wire [31:0] a_ss  = fp32_add(ss, acc_sq);

    // ---- PREP: mean / v / rsqrt decompose (combinational from ss / mean32 / v32) ----
    wire [31:0] p_mean = fp32_mul(ss, INV_D32);
    wire [31:0] p_v    = fp32_add(mean32, EPS32);
    wire        v_sub  = (v32[30:23] == 8'h0);
    wire [7:0]  eff_e  = v_sub ? EPS32[30:23] : v32[30:23];
    wire [22:0] eff_m  = v_sub ? EPS32[22:0]  : v32[22:0];
    wire signed [9:0] Eexp = $signed({2'b0, eff_e}) - 10'sd127;
    wire [5:0]  r_idx  = eff_m[22:17];
    wire [16:0] r_frac = eff_m[16:0];
    wire [15:0] lo16   = rsqrt_lut(r_idx);
    wire [15:0] hi16   = (r_idx < 6'd63) ? rsqrt_lut(r_idx + 6'd1) : RSQRT_TOP16;
    wire [31:0] p_diff = fp32_sub(rq_hi, rq_lo);
    wire [31:0] p_term = fp32_mul(rq_frac, rq_diff);
    wire [31:0] p_r    = fp32_add(rq_lo, rq_term);
    wire [31:0] p_rk   = fp32_mul(rq_r, pow2_fp32(-rq_k));
    wire [31:0] p_scale = rq_odd ? fp32_mul(rq_r, INV_SQRT2_32) : rq_r;

    // ---- EMIT combinational cones ----
    wire [31:0] e_t1  = fp32_mul(fp16_to_fp32(x_mem[rd]), scale32);
    wire [31:0] e_t2  = fp32_mul(et1, fp16_to_fp32(g_mem[rd]));
    wire [FW-1:0] e_y = fp32_to_fp16(et2);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; step <= 4'd0;
            y_valid <= 1'b0; y_last <= 1'b0; busy <= 1'b0;
            wr_ptr <= {PTRW{1'b0}}; ap <= {PTRW{1'b0}}; rd <= {PTRW{1'b0}}; count <= {PTRW{1'b0}};
            ss <= 32'b0;
        end else begin
            y_valid <= 1'b0; y_last <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (s_valid) begin
                        busy <= 1'b1;
                        x_mem[0] <= s_data; g_mem[0] <= g_data;
                        wr_ptr <= { {(PTRW-1){1'b0}}, 1'b1 };
                        if (s_last) begin
                            count <= { {(PTRW-1){1'b0}}, 1'b1 };
                            ss <= 32'b0; ap <= {PTRW{1'b0}}; step <= 4'd0; state <= S_ACC;
                        end else state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    if (s_valid) begin
                        x_mem[wr_ptr] <= s_data; g_mem[wr_ptr] <= g_data;
                        wr_ptr <= wr_ptr + 1'b1;
                        if (s_last) begin
                            count <= wr_ptr + 1'b1;
                            ss <= 32'b0; ap <= {PTRW{1'b0}}; step <= 4'd0; state <= S_ACC;
                        end
                    end
                end
                // -------- sum of squares (fp32 accumulator) --------
                S_ACC: begin
                    case (step)
                        4'd0: begin acc_sq <= a_sq; step <= 4'd1; end
                        default: begin                    // step1: accumulate, advance
                            ss <= a_ss; step <= 4'd0;
                            if (ap == count - 1'b1) begin step <= 4'd0; state <= S_PREP; end
                            else ap <= ap + 1'b1;
                        end
                    endcase
                end
                // -------- rsqrt(mean(x^2)+eps) via LUT + interp --------
                S_PREP: begin
                    case (step)
                        4'd0: begin mean32 <= p_mean; step <= 4'd1; end
                        4'd1: begin v32 <= p_v; step <= 4'd2; end
                        4'd2: begin
                            rq_lo <= fp16_to_fp32(lo16); rq_hi <= fp16_to_fp32(hi16);
                            rq_frac <= frac17_to_fp32(r_frac);
                            rq_k <= Eexp >>> 1; rq_odd <= Eexp[0];
                            step <= 4'd3;
                        end
                        4'd3: begin rq_diff <= p_diff; step <= 4'd4; end
                        4'd4: begin rq_term <= p_term; step <= 4'd5; end
                        4'd5: begin rq_r <= p_r; step <= 4'd6; end
                        4'd6: begin rq_r <= p_rk; step <= 4'd7; end
                        default: begin                    // step7: apply odd 1/sqrt2 -> scale
                            scale32 <= p_scale;
                            rd <= {PTRW{1'b0}}; step <= 4'd0; state <= S_EMIT;
                        end
                    endcase
                end
                // -------- emit y_i = x_i * scale * g_i --------
                S_EMIT: begin
                    case (step)
                        4'd0: begin et1 <= e_t1; step <= 4'd1; end
                        4'd1: begin et2 <= e_t2; step <= 4'd2; end
                        default: begin                    // step2: round + emit
                            y_data <= e_y; y_valid <= 1'b1; y_last <= (rd == count - 1'b1);
                            step <= 4'd0;
                            if (rd == count - 1'b1) begin state <= S_IDLE; busy <= 1'b0; end
                            else rd <= rd + 1'b1;
                        end
                    endcase
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
