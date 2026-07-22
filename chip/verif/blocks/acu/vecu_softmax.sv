// vecu_softmax.sv
//
// Synthesizable VecU decode online-softmax slice of the LonghornSilicon "Lambda"
// vector unit. For DECODE there is one attention row; this block turns a stream of
// L fp16 scores into the L fp16 attention weights (probabilities):
//
//     p_j = exp(s_j - m) / Σ_k exp(s_k - m)          (m = max_k s_k)
//
// It models the ACTUAL hardware algorithm (architecture/dataflow_walkthrough.md
// Stage 9; paper/lambda.tex §VecU), NOT a plain softmax:
//   - exp() via a 64-entry LUT for x ∈ [-16, 0] with linear interpolation. Entry
//     E[i] = round_fp16(exp(-0.25·i)), i=0..63 (E[0]=1.0 anchors the running max
//     exactly); the last interval interpolates toward exp(-16); x < -16 clamps 0.
//   - the online-softmax recurrence with the exp(m_old-m_new) rescale
//     (Milakov & Gimelshein 2018; FlashAttention running-max/running-sum core):
//     keep m (running max, fp16) and ℓ (running sum-of-exp, fp32 accumulator);
//     per score  m_new=max(m,s);  ℓ = ℓ·exp(m-m_new) + exp(s-m_new).
//   - emit p_j = exp(s_j - m_final) · (1/ℓ_final), reciprocal in fp16, product
//     rounded to fp16.
//
// The exp/interp and ℓ run in fp32 (like the MatE accumulator) so the block's
// error reflects the LUT approximation, not fp16 sum drift. The fp16/fp32
// primitives (fp16↔fp32, fp32_add/mul, fp32_to_fp16) are the same IEEE datapath
// used by mate_pv_fp16 / mate_qkt; fp16_div is the one new op (reciprocal 1/ℓ).
// Bit-exact to sw/reference_model/vecu_softmax_ref.py (see docs/vecu_softmax_rtl.md).
//
// INTERFACE (house streaming style):
//   LOAD  — present one fp16 score per clock on s_data with s_valid=1; assert
//           s_last=1 on the final score. Scores are buffered (depth N).
//   EMIT  — after the pipeline drains, the block streams the L weights: w_valid
//           pulses each cycle with the fp16 weight on w_data, w_last on the final
//           one. The consumer just waits on the w_valid handshake, so the (data-
//           independent) pipeline latency is transparent.
//   busy is high from the first score until the last weight is emitted.
//
// MICRO-SEQUENCED (this revision): scores are buffered on load, then the recurrence
// and the emit run as micro-sequences that execute ONE fp32 op per cycle (each
// register-to-register path holds at most one fp32 add or multiply). This rebalances
// the earlier 3-stage pipeline — whose longest stage (~263 ns at GF180 ss) forced a
// 300 ns clock + an aggressive resize that ~2×'d the cell count — into even ~½-length
// stages that close ss with NORMAL resizing, bringing area back down. The op sequence
// and rounding are UNCHANGED, so the result stays bit-exact to vecu_softmax_ref.py
// (added latency only — decode is latency-tolerant; the consumer waits on w_valid).
//
// Synthesis: micro-sequenced FSM + reused intermediate registers + score buffer +
// fp32 accumulator; no latches.

`timescale 1ns/1ps

module vecu_softmax #(
    parameter integer N  = 16,     // max scores per row (buffer depth)
    parameter integer FW = 16      // fp16 width
) (
    input  wire            clk,
    input  wire            rst_n,

    input  wire            s_valid,   // a score is being presented
    input  wire [FW-1:0]   s_data,    // fp16 attention score
    input  wire            s_last,    // last score of the row

    output reg             w_valid,   // an attention weight is being emitted
    output reg  [FW-1:0]   w_data,    // fp16 attention weight (probability)
    output reg             w_last,    // last weight of the row
    output reg             busy
);

    localparam integer PTRW    = (N <= 1) ? 1 : $clog2(N + 1);
    localparam [FW-1:0] NEG_INF = 16'hFC00;   // -inf fp16 (initial running max)

    // =======================================================================
    // fp16 -> fp32 widen (exact)
    // =======================================================================
    function automatic [31:0] fp16_to_fp32;
        input [15:0] h;
        reg s; reg [4:0] e; reg [9:0] m; integer e2, jj; reg [10:0] mm; reg [7:0] eo;
        begin
            e2 = 0; jj = 0; mm = 11'b0; eo = 8'b0;          // default-init locals (no latch inference)
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
    // fp16 divide (RTNE) — used only for the reciprocal 1/ℓ (ℓ a positive normal)
    // =======================================================================
    function automatic [15:0] fp16_div;
        input [15:0] a;
        input [15:0] b;
        reg        sa, ea_z; reg [4:0] ea; reg [9:0] ma;
        reg        sb; reg [4:0] eb; reg [9:0] mb;
        reg [10:0] sia, sib;
        integer    Ea, Eb, Eq, msb, sh, exp, tsh, k;
        reg [23:0] num;
        reg [13:0] Q;
        reg [23:0] rem;
        reg        sign, guard, sticky;
        reg [11:0] sig;
        begin
            sa = a[15]; ea = a[14:10]; ma = a[9:0];
            sb = b[15]; eb = b[14:10]; mb = b[9:0];
            sign = sa ^ sb;
            if (ea == 5'h0 && ma == 10'h0) fp16_div = {sign, 15'b0};
            else begin
                // normalize a
                if (ea == 5'h0) begin
                    sia = {1'b0, ma}; Ea = -24;
                    for (k = 0; k < 10; k = k + 1) if (!sia[10]) begin sia = sia << 1; Ea = Ea - 1; end
                end else begin sia = {1'b1, ma}; Ea = ea - 25; end
                // normalize b
                if (eb == 5'h0) begin
                    sib = {1'b0, mb}; Eb = -24;
                    for (k = 0; k < 10; k = k + 1) if (!sib[10]) begin sib = sib << 1; Eb = Eb - 1; end
                end else begin sib = {1'b1, mb}; Eb = eb - 25; end
                num = sia << 13;
                Q   = num / sib;
                rem = num - (Q * sib);
                Eq  = Ea - Eb - 13;
                msb = Q[13] ? 13 : 12;
                sh  = msb - 10;
                exp = msb + Eq + 15;
                if (exp < 1) begin
                    tsh = sh + (1 - exp);
                    if (tsh >= 14) fp16_div = {sign, 15'b0};
                    else begin
                        sig    = Q >> tsh;
                        guard  = (Q >> (tsh - 1)) & 1'b1;
                        sticky = (|(Q & ((14'b1 << (tsh-1)) - 14'b1))) | (|rem);
                        sig    = sig + {11'b0, (guard & (sticky | sig[0]))};
                        fp16_div = {sign, 5'b0, sig[9:0]};         // subnormal (exp field 0)
                    end
                end else begin
                    sig    = Q >> sh;
                    guard  = (Q >> (sh - 1)) & 1'b1;
                    sticky = (|(Q & ((14'b1 << (sh-1)) - 14'b1))) | (|rem);
                    sig    = sig + {11'b0, (guard & (sticky | sig[0]))};
                    if (sig[11]) begin sig = sig >> 1; exp = exp + 1; end
                    if (exp >= 31) fp16_div = {sign, 5'h1F, 10'b0};
                    else           fp16_div = {sign, exp[4:0], sig[9:0]};
                end
            end
        end
    endfunction

    // =======================================================================
    // fp16 signed compare  (a > b ?)  via the sortable-key transform
    // =======================================================================
    function automatic fp16_gt;
        input [15:0] a;
        input [15:0] b;
        reg [15:0] ka, kb;
        begin
            ka = a[15] ? ~a : (a | 16'h8000);
            kb = b[15] ? ~b : (b | 16'h8000);
            fp16_gt = (ka > kb);
        end
    endfunction

    // =======================================================================
    // 64-entry exp LUT (fp16), the fp32 index conversion, and the interpolation
    // =======================================================================
    function automatic [15:0] exp_lut_entry;
        input [5:0] i;
        begin
            case (i)
                6'd0 : exp_lut_entry = 16'h3C00;  6'd1 : exp_lut_entry = 16'h3A3B;
                6'd2 : exp_lut_entry = 16'h38DA;  6'd3 : exp_lut_entry = 16'h378F;
                6'd4 : exp_lut_entry = 16'h35E3;  6'd5 : exp_lut_entry = 16'h3496;
                6'd6 : exp_lut_entry = 16'h3324;  6'd7 : exp_lut_entry = 16'h3190;
                6'd8 : exp_lut_entry = 16'h3055;  6'd9 : exp_lut_entry = 16'h2EBF;
                6'd10: exp_lut_entry = 16'h2D41;  6'd11: exp_lut_entry = 16'h2C17;
                6'd12: exp_lut_entry = 16'h2A5F;  6'd13: exp_lut_entry = 16'h28F7;
                6'd14: exp_lut_entry = 16'h27BB;  6'd15: exp_lut_entry = 16'h2605;
                6'd16: exp_lut_entry = 16'h24B0;  6'd17: exp_lut_entry = 16'h234E;
                6'd18: exp_lut_entry = 16'h21B0;  6'd19: exp_lut_entry = 16'h206E;
                6'd20: exp_lut_entry = 16'h1EE6;  6'd21: exp_lut_entry = 16'h1D60;
                6'd22: exp_lut_entry = 16'h1C2F;  6'd23: exp_lut_entry = 16'h1A85;
                6'd24: exp_lut_entry = 16'h1914;  6'd25: exp_lut_entry = 16'h17E8;
                6'd26: exp_lut_entry = 16'h1628;  6'd27: exp_lut_entry = 16'h14CC;
                6'd28: exp_lut_entry = 16'h1378;  6'd29: exp_lut_entry = 16'h11D1;
                6'd30: exp_lut_entry = 16'h1088;  6'd31: exp_lut_entry = 16'h0F0F;
                6'd32: exp_lut_entry = 16'h0D7F;  6'd33: exp_lut_entry = 16'h0C48;
                6'd34: exp_lut_entry = 16'h0AAB;  6'd35: exp_lut_entry = 16'h0931;
                6'd36: exp_lut_entry = 16'h080B;  6'd37: exp_lut_entry = 16'h064C;
                6'd38: exp_lut_entry = 16'h04E8;  6'd39: exp_lut_entry = 16'h03D2;
                6'd40: exp_lut_entry = 16'h02FA;  6'd41: exp_lut_entry = 16'h0251;
                6'd42: exp_lut_entry = 16'h01CE;  6'd43: exp_lut_entry = 16'h0168;
                6'd44: exp_lut_entry = 16'h0118;  6'd45: exp_lut_entry = 16'h00DA;
                6'd46: exp_lut_entry = 16'h00AA;  6'd47: exp_lut_entry = 16'h0084;
                6'd48: exp_lut_entry = 16'h0067;  6'd49: exp_lut_entry = 16'h0050;
                6'd50: exp_lut_entry = 16'h003F;  6'd51: exp_lut_entry = 16'h0031;
                6'd52: exp_lut_entry = 16'h0026;  6'd53: exp_lut_entry = 16'h001E;
                6'd54: exp_lut_entry = 16'h0017;  6'd55: exp_lut_entry = 16'h0012;
                6'd56: exp_lut_entry = 16'h000E;  6'd57: exp_lut_entry = 16'h000B;
                6'd58: exp_lut_entry = 16'h0008;  6'd59: exp_lut_entry = 16'h0007;
                6'd60: exp_lut_entry = 16'h0005;  6'd61: exp_lut_entry = 16'h0004;
                6'd62: exp_lut_entry = 16'h0003;  default: exp_lut_entry = 16'h0002;
            endcase
        end
    endfunction

    // exact fp32 of frac10/1024 (frac10 in [0,1023])
    function automatic [31:0] frac_to_fp32;
        input [9:0] frac10;
        integer p, k; reg [7:0] exp; reg [33:0] fsh;
        begin
            if (frac10 == 10'b0) frac_to_fp32 = 32'b0;
            else begin
                p = 0;
                for (k = 0; k < 10; k = k + 1) if (frac10[k]) p = k;
                fsh = {24'b0, frac10} << (23 - p);
                exp = (p - 10) + 127;
                frac_to_fp32 = {1'b0, exp, fsh[22:0]};
            end
        end
    endfunction

    // floor(neg * 4096) as a 17-bit integer; neg32 is an fp32 pattern, 0 <= neg < 16
    function automatic [16:0] neg_to_fixed;
        input [31:0] neg32;
        reg [7:0] e; reg [22:0] m; reg [23:0] sig; integer shift;
        begin
            e = neg32[30:23]; m = neg32[22:0];
            if (e == 8'h0) neg_to_fixed = 17'b0;
            else begin
                sig = {1'b1, m};
                shift = 138 - e;
                if (shift <= 0)       neg_to_fixed = 17'h1FFFF;    // clamped (caller filters neg>=16)
                else if (shift >= 24) neg_to_fixed = 17'b0;
                else                  neg_to_fixed = sig >> shift;
            end
        end
    endfunction

    // exp index — neg conversion + fixed-point LUT index. Returns {zero(1), i(6),
    // frac10(10)}. The LUT read (exp_lut_entry) and the fp32 sub/mul/add of the
    // interpolation are done in SEPARATE micro-sequence cycles so each cycle holds at
    // most one fp32 op. exp(x) is then lo + frac·(hi-lo), or 0 when zero (x ≤ -16).
    function automatic [16:0] neg_and_index;
        input [31:0] x32;
        reg [31:0] neg32; reg [16:0] fixed; reg zero; reg [5:0] i; reg [9:0] frac10;
        begin
            if ((x32 & 32'h7FFFFFFF) == 32'b0) neg32 = 32'b0;   // x == 0
            else                               neg32 = x32 ^ 32'h80000000;
            if (neg32[31])                     neg32 = 32'b0;   // x > 0 (should not occur)
            if ((neg32 & 32'h7FFFFFFF) >= 32'h41800000) begin   // neg >= 16 -> exp ~ 0
                zero = 1'b1; i = 6'b0; frac10 = 10'b0;
            end else begin
                zero   = 1'b0;
                fixed  = neg_to_fixed(neg32);
                i      = fixed[15:10];
                frac10 = fixed[9:0];
            end
            neg_and_index = {zero, i, frac10};
        end
    endfunction

    // =======================================================================
    // Micro-sequenced datapath (rebalanced — even ~one-fp32-op stages).
    //
    // Scores are BUFFERED on load (1/cycle, trivial), then the online-softmax
    // recurrence and the weight emit run as micro-sequences that execute ONE fp32
    // operation per cycle, registering the intermediate value each cycle. Every
    // register-to-register path therefore holds at most one fp32 add or multiply
    // (~half the 3-stage version's longest ~263 ns stage), so GF180 ss closes at a
    // faster clock with NORMAL (non-aggressive) resizing — the aggressive upsize that
    // ~2×'d the cell count is no longer needed, bringing area back down. The op
    // sequence and rounding are UNCHANGED, so the result stays bit-exact to
    // vecu_softmax_ref.py (added latency only — decode is latency-tolerant, and the
    // consumer waits on the w_valid handshake).
    //
    //   LOAD     : buffer scores (1/cycle)                                    fast
    //   COMPUTE  : per score, 8 cycles — read; max+sub; index/LUT; diff sub;
    //              interp mul; interp add; rescale mul; accumulate add
    //   PREP     : 1/ℓ reciprocal
    //   EMIT     : per weight, 8 cycles — read; sub; index/LUT; diff sub; interp
    //              mul; interp add; ·(1/ℓ) mul; round-to-fp16
    // Longest reg-to-reg path is now one fp32 op (~½ the old chain). The intermediate
    // registers are reused across scores/weights (no per-stage replication), and the
    // recurrence is single-issue (no overlap hazards).
    // =======================================================================
    localparam [2:0] S_IDLE = 3'd0, S_LOAD = 3'd1, S_COMPUTE = 3'd2, S_PREP = 3'd3, S_EMIT = 3'd4;
    reg [2:0]      state;
    reg [3:0]      cstep, estep;         // micro-step counters (0..7)
    reg [FW-1:0]   m_rec;                // running max (fp16) — also the final max for EMIT
    reg [31:0]     l32;                  // running sum-of-exp (fp32 accumulator)
    reg [FW-1:0]   inv_l16;              // 1/ℓ_final (fp16)
    reg [FW-1:0]   score_mem [0:N-1];    // buffered scores
    reg [PTRW-1:0] wr_ptr, cp_ptr, rd_ptr, count;

    reg [FW-1:0]   cur_reg;              // score under evaluation (registered read)
    // COMPUTE intermediates (one set, reused each score)
    reg [FW-1:0]   cs_mnew;
    reg [31:0]     cs_xr, cs_xc;
    reg            cs_zr, cs_zc;
    reg [31:0]     cs_lor, cs_hir, cs_fracr, cs_loc, cs_hic, cs_fracc;
    reg [31:0]     cs_diffr, cs_diffc, cs_termr, cs_termc, cs_resc, cs_e, cs_prod;
    // EMIT intermediates (one set, reused each weight)
    reg [31:0]     es_x, es_lo, es_hi, es_frac, es_diff, es_term, es_e, es_prod;
    reg            es_z;

    // ---- combinational cones — each is at most ONE fp32 op from registered inputs ----
    // COMPUTE
    wire [FW-1:0] c_mnew  = fp16_gt(cur_reg, m_rec) ? cur_reg : m_rec;
    wire [31:0]   c_xr    = fp32_sub(fp16_to_fp32(m_rec),   fp16_to_fp32(c_mnew));
    wire [31:0]   c_xc    = fp32_sub(fp16_to_fp32(cur_reg), fp16_to_fp32(c_mnew));
    wire [16:0]   c_nir   = neg_and_index(cs_xr);
    wire [16:0]   c_nic   = neg_and_index(cs_xc);
    wire [5:0]    c_ir    = c_nir[15:10]; wire [9:0] c_f10r = c_nir[9:0];
    wire [5:0]    c_ic    = c_nic[15:10]; wire [9:0] c_f10c = c_nic[9:0];
    wire [31:0]   c_lor   = fp16_to_fp32(exp_lut_entry(c_ir));
    wire [31:0]   c_hir   = fp16_to_fp32((c_ir < 6'd63) ? exp_lut_entry(c_ir + 6'd1) : 16'h0002);
    wire [31:0]   c_fracr = frac_to_fp32(c_f10r);
    wire [31:0]   c_loc   = fp16_to_fp32(exp_lut_entry(c_ic));
    wire [31:0]   c_hic   = fp16_to_fp32((c_ic < 6'd63) ? exp_lut_entry(c_ic + 6'd1) : 16'h0002);
    wire [31:0]   c_fracc = frac_to_fp32(c_f10c);
    wire [31:0]   c_diffr = fp32_sub(cs_hir, cs_lor);
    wire [31:0]   c_diffc = fp32_sub(cs_hic, cs_loc);
    wire [31:0]   c_termr = fp32_mul(cs_fracr, cs_diffr);
    wire [31:0]   c_termc = fp32_mul(cs_fracc, cs_diffc);
    wire [31:0]   c_resc  = cs_zr ? 32'b0 : fp32_add(cs_lor, cs_termr);
    wire [31:0]   c_e     = cs_zc ? 32'b0 : fp32_add(cs_loc, cs_termc);
    wire [31:0]   c_prod  = fp32_mul(l32, cs_resc);
    wire [31:0]   c_lnext = fp32_add(cs_prod, cs_e);
    // PREP
    wire [FW-1:0] inv_l_comb = fp16_div(16'h3C00, fp32_to_fp16(l32));
    // EMIT (cur_reg holds score_mem[rd_ptr]; m_rec is the final max)
    wire [31:0]   e_x     = fp32_sub(fp16_to_fp32(cur_reg), fp16_to_fp32(m_rec));
    wire [16:0]   e_ni    = neg_and_index(es_x);
    wire [5:0]    e_i     = e_ni[15:10]; wire [9:0] e_f10 = e_ni[9:0];
    wire [31:0]   e_lo    = fp16_to_fp32(exp_lut_entry(e_i));
    wire [31:0]   e_hi    = fp16_to_fp32((e_i < 6'd63) ? exp_lut_entry(e_i + 6'd1) : 16'h0002);
    wire [31:0]   e_frac  = frac_to_fp32(e_f10);
    wire [31:0]   e_diff  = fp32_sub(es_hi, es_lo);
    wire [31:0]   e_term  = fp32_mul(es_frac, es_diff);
    wire [31:0]   e_e     = es_z ? 32'b0 : fp32_add(es_lo, es_term);
    wire [31:0]   e_prod  = fp32_mul(es_e, fp16_to_fp32(inv_l16));
    wire [FW-1:0] e_w     = fp32_to_fp16(es_prod);

    always @(posedge clk) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            w_valid <= 1'b0; w_last <= 1'b0; busy <= 1'b0;
            m_rec   <= NEG_INF; l32 <= 32'b0; inv_l16 <= 16'b0;
            wr_ptr  <= {PTRW{1'b0}}; cp_ptr <= {PTRW{1'b0}}; rd_ptr <= {PTRW{1'b0}}; count <= {PTRW{1'b0}};
            cstep   <= 4'd0; estep <= 4'd0;
        end else begin
            w_valid <= 1'b0; w_last <= 1'b0;                // default: no weight this cycle
            case (state)
                // ---------------- buffer scores 1/cycle ----------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (s_valid) begin
                        busy <= 1'b1;
                        score_mem[0] <= s_data;
                        wr_ptr <= { {(PTRW-1){1'b0}}, 1'b1 };
                        if (s_last) begin
                            count <= { {(PTRW-1){1'b0}}, 1'b1 };
                            m_rec <= NEG_INF; l32 <= 32'b0; cp_ptr <= {PTRW{1'b0}}; cstep <= 4'd0;
                            state <= S_COMPUTE;
                        end else state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    if (s_valid) begin
                        score_mem[wr_ptr] <= s_data;
                        wr_ptr <= wr_ptr + 1'b1;
                        if (s_last) begin
                            count <= wr_ptr + 1'b1;
                            m_rec <= NEG_INF; l32 <= 32'b0; cp_ptr <= {PTRW{1'b0}}; cstep <= 4'd0;
                            state <= S_COMPUTE;
                        end
                    end
                end
                // -------- online-softmax recurrence, 1 fp32 op/cycle --------
                S_COMPUTE: begin
                    case (cstep)
                        4'd0: begin cur_reg <= score_mem[cp_ptr]; cstep <= 4'd1; end
                        4'd1: begin cs_mnew <= c_mnew; cs_xr <= c_xr; cs_xc <= c_xc; cstep <= 4'd2; end
                        4'd2: begin
                            cs_zr <= c_nir[16]; cs_lor <= c_lor; cs_hir <= c_hir; cs_fracr <= c_fracr;
                            cs_zc <= c_nic[16]; cs_loc <= c_loc; cs_hic <= c_hic; cs_fracc <= c_fracc;
                            cstep <= 4'd3;
                        end
                        4'd3: begin cs_diffr <= c_diffr; cs_diffc <= c_diffc; cstep <= 4'd4; end
                        4'd4: begin cs_termr <= c_termr; cs_termc <= c_termc; cstep <= 4'd5; end
                        4'd5: begin cs_resc <= c_resc; cs_e <= c_e; cstep <= 4'd6; end
                        4'd6: begin cs_prod <= c_prod; cstep <= 4'd7; end
                        default: begin                     // cstep 7: commit ℓ and the running max
                            l32 <= c_lnext; m_rec <= cs_mnew; cstep <= 4'd0;
                            if (cp_ptr == count - 1'b1) state <= S_PREP;
                            else cp_ptr <= cp_ptr + 1'b1;
                        end
                    endcase
                end
                S_PREP: begin
                    inv_l16 <= inv_l_comb;
                    rd_ptr  <= {PTRW{1'b0}}; estep <= 4'd0;
                    state   <= S_EMIT;
                end
                // -------- weight emit, 1 fp32 op/cycle (feed-forward) --------
                S_EMIT: begin
                    case (estep)
                        4'd0: begin cur_reg <= score_mem[rd_ptr]; estep <= 4'd1; end
                        4'd1: begin es_x <= e_x; estep <= 4'd2; end
                        4'd2: begin es_z <= e_ni[16]; es_lo <= e_lo; es_hi <= e_hi; es_frac <= e_frac; estep <= 4'd3; end
                        4'd3: begin es_diff <= e_diff; estep <= 4'd4; end
                        4'd4: begin es_term <= e_term; estep <= 4'd5; end
                        4'd5: begin es_e <= e_e; estep <= 4'd6; end
                        4'd6: begin es_prod <= e_prod; estep <= 4'd7; end
                        default: begin                     // estep 7: emit the weight
                            w_data <= e_w; w_valid <= 1'b1; w_last <= (rd_ptr == count - 1'b1);
                            estep  <= 4'd0;
                            if (rd_ptr == count - 1'b1) begin state <= S_IDLE; busy <= 1'b0; end
                            else rd_ptr <= rd_ptr + 1'b1;
                        end
                    endcase
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
