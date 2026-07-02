// cq_units_syn.sv — SYNTHESIZABLE fp16 ChannelQuant cores (P4b lowering).
//
// Fixed-function replacements for the behavioral `real`-math cores in
// cq_units.sv, bit-exact vs them (and thus vs the golden vectors). No `real`,
// no `$*bits`. Verified module-for-module against the behavioral oracle by
// tb_cq_syn.sv (make sim_syn) and synthesizable-checked with yosys.
//
// fp16 = {s[15], e[14:10], m[9:0]}: normal e∈[1,30] -> (-1)^s·1.m·2^(e-15);
// subnormal e=0 -> (-1)^s·0.m·2^-14. Golden data is finite (no inf/nan).
//
// This file grows one core at a time (dequant -> scale -> quant), each proven
// before the next. See rtl/P4B_PLAN.md.

`default_nettype none

// ===========================================================================
// dequant: xhat_f32 = code * s   (EXACT — product fits fp32's 23-bit mantissa)
// ===========================================================================
// |code| ≤ 128, sig_s ≤ 11 bits -> product ≤ 19 bits ≤ 23, so no rounding.
module cq_dequant_unit_syn (
    input  wire signed [7:0] code,
    input  wire [15:0]       scale_f16,
    output reg  [31:0]       xhat_f32
);
    wire       ssign = scale_f16[15];
    wire [4:0] sexp  = scale_f16[14:10];
    wire [9:0] sman  = scale_f16[9:0];

    // |s| = sig_s * 2^e_s  (sig_s 11-bit significand; e_s = exponent of its LSB)
    reg  [10:0]      sig_s;
    reg  signed [7:0] e_s;
    reg              s_zero;
    always @* begin
        if (sexp == 5'd0) begin              // subnormal / zero
            sig_s  = {1'b0, sman};
            e_s    = -8'sd24;
            s_zero = (sman == 10'd0);
        end else begin                        // normal (implicit 1)
            sig_s  = {1'b1, sman};
            e_s    = $signed({3'b000, sexp}) - 8'sd25;
            s_zero = 1'b0;
        end
    end

    wire        csign = code[7];
    wire [7:0]  cmag  = csign ? (~code + 8'd1) : code;   // |code|, 0..128
    wire        outsign = csign ^ ssign;
    wire        zero    = s_zero | (cmag == 8'd0);

    wire [18:0] prod = cmag * sig_s;          // ≤ 19 bits

    // leading-one index of prod
    integer i;
    reg [4:0] msb;
    always @* begin
        msb = 5'd0;
        for (i = 0; i < 19; i = i + 1) if (prod[i]) msb = i[4:0];
    end

    reg signed [9:0] uexp;
    reg        [9:0] biased;
    reg [41:0]       mant_wide;
    always @* begin
        // assign all intermediates unconditionally (no latches), select at the end
        uexp      = $signed({5'b0, msb}) + $signed({{2{e_s[7]}}, e_s});
        biased    = uexp + 10'sd127;
        // fractional bits = prod below the leading one, left-justified to 23
        mant_wide = ({23'b0, (prod & ((19'd1 << msb) - 19'd1))}) << (6'd23 - msb);
        xhat_f32  = zero ? 32'h0000_0000 : {outsign, biased[7:0], mant_wide[22:0]};
    end
endmodule

// ===========================================================================
// scale: s = max(amax / qmax, EPS) -> fp16    (qmax = 7 or 127; EPS = 2^-14)
// ===========================================================================
// Fixed-point divide of the amax significand by the small constant qmax with F
// guard bits + exact remainder, then round-half-even to the 10-bit fp16 mantissa.
// EPS clamp done exactly as an integer compare (sig_a·2^(ea+14) < qmax).
module cq_scale_unit_syn (
    input  wire [15:0] amax_f16,
    input  wire [3:0]  bits,        // 4 or 8
    output reg  [15:0] scale_f16
);
    localparam integer F = 24;

    wire [4:0] aexp = amax_f16[14:10];
    wire [9:0] aman = amax_f16[9:0];   // amax is non-negative (sign cleared)

    // a = sig_a * 2^ea
    reg  [10:0]      sig_a;
    reg  signed [7:0] ea;
    reg              a_zero;
    always @* begin
        if (aexp == 5'd0) begin
            sig_a  = {1'b0, aman};
            ea     = -8'sd24;
            a_zero = (aman == 10'd0);
        end else begin
            sig_a  = {1'b1, aman};
            ea     = $signed({3'b000, aexp}) - 8'sd25;
            a_zero = 1'b0;
        end
    end

    wire [8:0] qmax = (9'd1 << (bits - 4'd1)) - 9'd1;   // 7 (int4) or 127 (int8)

    // EPS clamp: s < 2^-14  <=>  sig_a * 2^(ea+14) < qmax   (exact integer test)
    integer sh;
    reg     clamp;
    always @* begin
        sh = ea + 14;
        if (a_zero)        clamp = 1'b1;
        else if (sh >= 0)  clamp = (({40'b0, sig_a} << sh) < {41'b0, qmax});
        else               clamp = ({40'b0, sig_a} < ({40'b0, qmax} << (-sh)));
    end

    // Q = floor(sig_a * 2^F / qmax), Rem = (sig_a * 2^F) mod qmax  (sticky tail)
    wire [47:0] numer      = {37'b0, sig_a} << F;
    wire [47:0] Q          = numer / {39'b0, qmax};
    wire [47:0] Rem        = numer % {39'b0, qmax};
    wire        sticky_rem = (Rem != 48'd0);

    // leading-one index of Q
    integer k;
    reg [5:0] L;
    always @* begin
        L = 6'd0;
        for (k = 0; k < 48; k = k + 1) if (Q[k]) L = k[5:0];
    end

    integer sh_m, sh_r, e_un, biasedi;
    reg [11:0] sig11, sig11_r;
    reg        rbit, lsb, sticky;
    reg [47:0] mask_lo;
    reg [9:0]  frac10;
    always @* begin
        sh_m = (L >= 6'd10) ? (L - 10) : 0;
        sh_r = (L >= 6'd11) ? (L - 11) : 0;
        sig11   = (Q >> sh_m) & 12'h7FF;               // bits [L:L-10], implicit 1 at bit10
        rbit    = (Q >> sh_r) & 48'd1;                 // bit [L-11]
        mask_lo = ({47'b0, 1'b1} << sh_r) - 48'd1;
        sticky  = sticky_rem | ((Q & mask_lo) != 48'd0);
        lsb     = sig11[0];
        sig11_r = sig11 + {11'b0, (rbit & (sticky | lsb))};
        if (sig11_r[11]) begin                          // rounded up to 2.0 -> carry
            frac10 = 10'd0;
            e_un   = ($signed({26'b0, L}) + $signed(ea)) - F + 1;
        end else begin
            frac10 = sig11_r[9:0];
            e_un   = ($signed({26'b0, L}) + $signed(ea)) - F;
        end
        biasedi   = e_un + 15;
        scale_f16 = clamp ? 16'h0400 : {1'b0, biasedi[4:0], frac10};
    end
endmodule

// ===========================================================================
// quant: q = clamp(round_half_even(x / s), qmin, qmax)   (bounded fp divide)
// ===========================================================================
// Sign-magnitude fixed-point divide of the significands with ONE guard bit + the
// exact division remainder as sticky -> round-half-even is exact (ties detected
// by roundbit=1 & sticky=0). Result clamped to the tier's signed int range.
// SEQUENTIAL (multi-cycle) form: the significand divide is done bit-serially
// (radix-2 restoring, <=21 steps) so the datapath has only short per-cycle paths
// instead of one ~443-level combinational divide cone — it place-and-routes and
// closes timing at a real clock. Bit-exact with the behavioral cq_quant_unit
// oracle (same guard-bit + exact-remainder round-half-even + clamp). Handshake:
// pulse `start` with valid operands; `done` pulses for one cycle with `code`
// valid. Only assert `start` when the unit is idle.
module cq_quant_unit_syn (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [15:0]       x_f16,
    input  wire [15:0]       scale_f16,
    input  wire [3:0]        bits,       // 4 or 8
    output reg  signed [7:0] code,
    output reg               done
);
    // ---- combinational unpack of the LIVE operands (sampled at `start`) --------
    wire        xsign = x_f16[15];
    wire [4:0]  xexp  = x_f16[14:10];
    wire [9:0]  xman  = x_f16[9:0];
    reg  [10:0]       sig_x_c;
    reg  signed [7:0] ex_c;
    always @* begin
        if (xexp == 5'd0) begin sig_x_c = {1'b0, xman}; ex_c = -8'sd24;                       end
        else              begin sig_x_c = {1'b1, xman}; ex_c = $signed({3'b000, xexp}) - 8'sd25; end
    end
    wire        ssign = scale_f16[15];
    wire [4:0]  sexp  = scale_f16[14:10];
    wire [9:0]  sman  = scale_f16[9:0];
    reg  [10:0]       sig_s_c;
    reg  signed [7:0] es_c;
    always @* begin
        if (sexp == 5'd0) begin sig_s_c = {1'b0, sman}; es_c = -8'sd24;                       end
        else              begin sig_s_c = {1'b1, sman}; es_c = $signed({3'b000, sexp}) - 8'sd25; end
    end
    wire [7:0]        qmax_m_c = (8'd1 << (bits - 4'd1)) - 8'd1;   // 7 or 127
    integer           dexp_c, P_c;
    reg  [4:0]        P_sh;
    reg  [20:0]       N_c;
    always @* begin
        dexp_c = $signed(ex_c) - $signed(es_c);
        P_c    = dexp_c + 1;                          // meaningful only in [0,9]
        P_sh   = (P_c < 0) ? 5'd0 : (P_c > 9 ? 5'd9 : P_c[4:0]);
        N_c    = {10'b0, sig_x_c} << P_sh;            // <= 20 bits
    end
    wire early_zero_c  = (dexp_c <= -2) || (sig_x_c == 11'd0);   // |q| < 0.5 (or x==0)
    wire early_clamp_c = (dexp_c >= 9)  || (sig_s_c == 11'd0);   // > 127 -> clamp

    // ---- multi-cycle state ----------------------------------------------------
    localparam S_IDLE = 2'd0, S_DIV = 2'd1, S_FIN = 2'd2;
    reg [1:0]  st;
    reg [4:0]  iter;
    reg [20:0] q_work;     // holds N, shifts out; ends holding the quotient
    reg [10:0] q_rem;      // running remainder
    reg [10:0] divisor_r;  // sig_s
    reg [7:0]  qmax_r;
    reg        sign_r, force_zero_r, force_clamp_r;

    // one radix-2 restoring step (combinational)
    wire [11:0] rem_sh  = {q_rem, q_work[20]};
    wire [11:0] div12   = {1'b0, divisor_r};
    wire        ge      = (rem_sh >= div12);
    wire [11:0] rem_sub = rem_sh - div12;
    wire [10:0] rem_nx  = ge ? rem_sub[10:0] : rem_sh[10:0];

    // final round-half-even from the quotient (q_work) + remainder (q_rem)
    wire [20:0] int_part = q_work >> 1;
    wire        roundbit = q_work[0];
    wire        sticky   = (q_rem != 11'd0);
    reg  [21:0] r_div;
    always @* begin
        if (!roundbit)   r_div = {1'b0, int_part};
        else if (sticky) r_div = int_part + 21'd1;
        else             r_div = int_part + {20'd0, int_part[0]};   // tie -> even
    end
    wire [21:0] r_mag = force_zero_r  ? 22'd0   :
                        force_clamp_r ? 22'd200 : r_div;
    wire [7:0]  qmin_c = ~qmax_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 1'b0; code <= 8'sd0;
            q_work <= 21'd0; q_rem <= 11'd0; iter <= 5'd0;
            divisor_r <= 11'd0; qmax_r <= 8'd0;
            sign_r <= 1'b0; force_zero_r <= 1'b0; force_clamp_r <= 1'b0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    sign_r        <= xsign ^ ssign;
                    qmax_r        <= qmax_m_c;
                    divisor_r     <= sig_s_c;
                    force_zero_r  <= early_zero_c;
                    force_clamp_r <= early_clamp_c && !early_zero_c;
                    if (early_zero_c || early_clamp_c) begin
                        q_work <= 21'd0; q_rem <= 11'd0;   // r_mag comes from force_*
                        st     <= S_FIN;
                    end else begin
                        q_work <= N_c;
                        q_rem  <= 11'd0;
                        iter   <= 5'd21;                   // process bits [20:0]
                        st     <= S_DIV;
                    end
                end
                S_DIV: begin
                    q_rem  <= rem_nx;
                    q_work <= {q_work[19:0], ge};          // shift in quotient bit
                    iter   <= iter - 5'd1;
                    if (iter == 5'd1) st <= S_FIN;
                end
                S_FIN: begin
                    // apply sign + clamp to [qmin, qmax]
                    if (!sign_r)
                        code <= (r_mag > {14'b0, qmax_r}) ? $signed({1'b0, qmax_r})
                                                          : $signed(r_mag[7:0]);
                    else
                        code <= (r_mag > ({14'b0, qmax_r} + 22'd1)) ? qmin_c
                                                                    : (~r_mag[7:0]) + 8'd1;
                    done <= 1'b1;
                    st   <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule

// ===========================================================================
// cq_quant_comb — COMBINATIONAL quant (the pre-serialization form).
// ===========================================================================
// Same math as cq_quant_unit_syn but single-cycle (a ~443-level divide cone).
// Used ONLY by the not-yet-serialized PARALLEL key path (cq_key_path), which is
// not in the synthesized top; it keeps that path's golden check bit-exact until
// it too is serialized onto the shared sequential divider. Do NOT instantiate in
// anything that goes through place-and-route.
module cq_quant_comb (
    input  wire [15:0]       x_f16,
    input  wire [15:0]       scale_f16,
    input  wire [3:0]        bits,       // 4 or 8
    output reg  signed [7:0] code
);
    wire        xsign = x_f16[15];
    wire [4:0]  xexp  = x_f16[14:10];
    wire [9:0]  xman  = x_f16[9:0];
    reg  [10:0]      sig_x;
    reg  signed [7:0] ex;
    always @* begin
        if (xexp == 5'd0) begin sig_x = {1'b0, xman}; ex = -8'sd24;                       end
        else              begin sig_x = {1'b1, xman}; ex = $signed({3'b000, xexp}) - 8'sd25; end
    end
    wire        ssign = scale_f16[15];
    wire [4:0]  sexp  = scale_f16[14:10];
    wire [9:0]  sman  = scale_f16[9:0];
    reg  [10:0]      sig_s;
    reg  signed [7:0] es;
    always @* begin
        if (sexp == 5'd0) begin sig_s = {1'b0, sman}; es = -8'sd24;                       end
        else              begin sig_s = {1'b1, sman}; es = $signed({3'b000, sexp}) - 8'sd25; end
    end
    wire [7:0] qmax_m = (8'd1 << (bits - 4'd1)) - 8'd1;
    wire [7:0] qmin_c = ~qmax_m;
    integer     dexp, P;
    reg [20:0]  N, quo, rem;
    reg [20:0]  int_part;
    reg         roundbit, sticky, sign_q;
    reg [21:0]  r_mag;
    always @* begin
        sign_q = xsign ^ ssign;
        dexp   = $signed(ex) - $signed(es);
        N = 21'd0; quo = 21'd0; rem = 21'd0;
        int_part = 21'd0; roundbit = 1'b0; sticky = 1'b0; P = 0;
        if (dexp <= -2 || sig_x == 11'd0) begin
            r_mag = 22'd0;
        end else if (dexp >= 9 || sig_s == 11'd0) begin
            r_mag = 22'd200;
        end else begin
            P        = dexp + 1;
            N        = {10'b0, sig_x} << P;
            quo      = N / {10'b0, sig_s};
            rem      = N % {10'b0, sig_s};
            int_part = quo >> 1;
            roundbit = quo[0];
            sticky   = (rem != 21'd0);
            if (!roundbit)   r_mag = {1'b0, int_part};
            else if (sticky) r_mag = int_part + 21'd1;
            else             r_mag = int_part + {20'd0, int_part[0]};
        end
        if (!sign_q) begin
            code = (r_mag > {14'b0, qmax_m}) ? $signed({1'b0, qmax_m}) : $signed(r_mag[7:0]);
        end else begin
            if (r_mag > ({14'b0, qmax_m} + 22'd1)) code = qmin_c;
            else                                   code = (~r_mag[7:0]) + 8'd1;
        end
    end
endmodule

`default_nettype wire
