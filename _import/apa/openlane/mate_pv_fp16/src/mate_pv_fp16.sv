// mate_pv_fp16.sv
//
// Synthesizable FP16 P·V MAC tile — the FP16 datapath of the LonghornSilicon
// "Lambda" MatE matrix engine. Companion to the signed-off INT8 `mate_pv`; same
// streaming interface, same registered-product (prod_reg) pipeline, but IEEE-754
// binary16 arithmetic with an internal binary32 (fp32) accumulator. Computes one
// attention output row
//
//     o[n] = round_fp16( Σ_t  A[t] · V[t][n] )        (n = 0 .. N-1 channels)
//
// realising the ACU MAC-array reference's FP16 contract (sw/reference_model/
// mac_array_ref.{hpp,py} `matmul_fp16`, fp16_accumulator_bits=32): each per-lane
// fp16×fp16 product is EXACT in fp32 (11b×11b significand ⇒ ≤22b, fits fp32's
// 24b; product magnitude 2^-48..2^31 is always an fp32 normal), the token-dim
// reduction is accumulated in fp32 with round-to-nearest-even at every add, and
// the emitted result is rounded to fp16 (RTNE, overflow→inf) exactly ONCE.
//
// Accumulation ORDER is sequential in token arrival order — the natural order of
// a streaming token-reduction MAC. That IS the hardware; it faithfully realises
// the reference's "fp32 accumulate, round-to-fp16-once" semantics. This RTL is
// bit-exact to a sequential-fp32 golden (which is itself bit-exact to numpy's
// sequential float32 accumulation over 4000+ random rows). The reference's numpy
// BLAS `@` sums in a different (pairwise, machine-dependent) fp32 order, so it
// agrees with sequential fp32 on ~99.9% of output lanes and differs by a few ULP
// on the rest — well within the FP16 path's documented rel_err<5e-3 tolerance,
// an inherent floating-point reduction-order artifact (BLAS order is not itself
// canonical), not a correctness gap. See docs/mate_pv_fp16_rtl.md.
//
// INTERFACE (identical house style to mate_pv, fp16 widths): present one token
// per clock with s_valid=1, its scalar fp16 A-code on a_data and its N-wide
// packed fp16 V-row on v_data; assert s_last=1 on the final token of the row.
// c_valid pulses when the row's Σ is ready, with the N fp16 results on c_data.
// Accumulators auto-reset (to +0.0) after the row for the next one.
//
// PIPELINED (mirrors mate_pv): the per-lane fp16×fp16→fp32 product is registered
// (prod_reg) between the multiply and the fp32 accumulate, so the long
// mult→fp32-add path is split into two stages (fp16 mult in stage 1, fp32 add in
// stage 2). The row sum is emitted 2 cycles after s_last. The VALUE is unchanged
// — same sequential Σ, just delayed — so it stays bit-exact to the golden.
//
// Latency  : 2 cycles after s_last.  Throughput: 1 token/cycle, 1 row per K tokens.
// Synthesis: registered fp32 product + registered fp32 accumulators; no latches.

`timescale 1ns/1ps

module mate_pv_fp16 #(
    parameter integer N   = 8,    // head-dim channels computed in parallel (lanes)
    parameter integer FW  = 16,   // fp16 operand/result width (IEEE-754 binary16)
    parameter integer PW  = 32    // internal fp32 product / accumulator width (binary32)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    s_valid,   // a token is being presented
    input  wire        [FW-1:0]    a_data,    // A[t]      : this token's attention fp16 code
    input  wire        [N*FW-1:0]  v_data,    // V[t][0..N-1]: this token's value row (packed fp16)
    input  wire                    s_last,    // last token of the output row

    output reg                     c_valid,   // pulses 2 cycles after s_last
    output reg  [N*FW-1:0]         c_data     // N fp16 dot-product results (round-to-fp16)
);

    genvar gi;
    integer i;

    // =======================================================================
    // FP16 × FP16 → FP32 (exact).  Signed IEEE-754 binary16 multiply; result is
    // an fp32 bit-pattern.  Handles zero / subnormal / inf / nan operands.
    // =======================================================================
    function automatic [31:0] fp16_mul;
        input [15:0] a;
        input [15:0] b;
        reg        sa, sb, sy;
        reg [4:0]  ea, eb;
        reg [9:0]  ma, mb;
        reg        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
        reg [10:0] siga, sigb;
        integer    Ea, Eb, Ep, msb, eb32, k, msba, msbb;
        reg [21:0] P;
        reg [24:0] Pal;
        begin
            sa = a[15]; ea = a[14:10]; ma = a[9:0];
            sb = b[15]; eb = b[14:10]; mb = b[9:0];
            sy = sa ^ sb;
            a_nan  = (ea == 5'h1F) && (|ma);  a_inf  = (ea == 5'h1F) && (~|ma);  a_zero = (ea == 5'h0) && (~|ma);
            b_nan  = (eb == 5'h1F) && (|mb);  b_inf  = (eb == 5'h1F) && (~|mb);  b_zero = (eb == 5'h0) && (~|mb);
            if (a_nan || b_nan) begin
                fp16_mul = 32'h7FC00000;                                  // canonical qNaN
            end else if (a_inf || b_inf) begin
                if ((a_inf && b_zero) || (b_inf && a_zero))
                    fp16_mul = 32'h7FC00000;                              // inf·0 = NaN
                else
                    fp16_mul = {sy, 8'hFF, 23'b0};                        // signed inf
            end else if (a_zero || b_zero) begin
                fp16_mul = {sy, 31'b0};                                   // signed zero
            end else begin
                // ---- normalise operands to 11-bit significand · 2^E ----
                // Subnormals use a priority-encoded MSB search + a single barrel
                // shift (NOT a shift-by-1 ripple) so the path stays short.
                if (ea == 5'h0) begin                                     // subnormal a (ma != 0)
                    msba = 0;
                    for (k = 0; k < 10; k = k + 1) if (ma[k]) msba = k;   // highest set bit
                    siga = {1'b0, ma} << (10 - msba); Ea = -24 - (10 - msba);
                end else begin                                           // normal a
                    siga = {1'b1, ma}; Ea = ea - 25;
                end
                if (eb == 5'h0) begin                                     // subnormal b (mb != 0)
                    msbb = 0;
                    for (k = 0; k < 10; k = k + 1) if (mb[k]) msbb = k;
                    sigb = {1'b0, mb} << (10 - msbb); Eb = -24 - (10 - msbb);
                end else begin                                           // normal b
                    sigb = {1'b1, mb}; Eb = eb - 25;
                end
                P   = siga * sigb;                                        // [2^20, 2^22)
                Ep  = Ea + Eb;
                msb = P[21] ? 21 : 20;                                    // top set bit
                eb32 = msb + Ep + 127;                                    // fp32 biased exp (always 79..158)
                Pal  = P << (23 - msb);                                   // align MSB to bit23
                fp16_mul = {sy, eb32[7:0], Pal[22:0]};                    // exact fp32 normal
            end
        end
    endfunction

    // =======================================================================
    // FP32 + FP32 → FP32, correctly-rounded round-to-nearest-even.  Full IEEE
    // adder: special-value handling, alignment with guard/round/sticky, add or
    // subtract by sign, renormalise, RTNE, overflow→inf, underflow→subnormal.
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
        reg [31:0] lowmask;
        reg        dropped, guard, roundb, sticky, roundup;
        reg [24:0] kept;
        reg [7:0]  EF;
        begin
            sa = a[31]; ea = a[30:23]; ma = a[22:0];
            sb = b[31]; eb = b[30:23]; mb = b[22:0];
            a_nan = (ea == 8'hFF) && (|ma);  a_inf = (ea == 8'hFF) && (~|ma);
            b_nan = (eb == 8'hFF) && (|mb);  b_inf = (eb == 8'hFF) && (~|mb);
            if (a_nan || b_nan) begin
                fp32_add = 32'h7FC00000;
            end else if (a_inf && b_inf) begin
                fp32_add = (sa == sb) ? a : 32'h7FC00000;                 // inf-inf = NaN
            end else if (a_inf) begin
                fp32_add = a;
            end else if (b_inf) begin
                fp32_add = b;
            end else begin
                siga = (ea == 8'h0) ? {1'b0, ma} : {1'b1, ma};
                sigb = (eb == 8'h0) ? {1'b0, mb} : {1'b1, mb};
                eea  = (ea == 8'h0) ? 1 : ea;                             // subnormals share exp -126
                eeb  = (eb == 8'h0) ? 1 : eb;
                // ---- order so `big` is the larger magnitude ----
                if (eea > eeb || (eea == eeb && siga >= sigb)) begin
                    E = eea; d = eea - eeb; big = {siga, 3'b0}; small0 = {sigb, 3'b0}; sbig = sa; ssmall = sb;
                end else begin
                    E = eeb; d = eeb - eea; big = {sigb, 3'b0}; small0 = {siga, 3'b0}; sbig = sb; ssmall = sa;
                end
                // ---- align smaller right by d, collecting sticky in bit0 ----
                if (d == 0) begin
                    small_sh = small0;
                end else if (d > 27) begin
                    small_sh = (|small0) ? 28'b1 : 28'b0;
                end else begin
                    small_sh = small0 >> d;
                    if (|(small0 & ((28'b1 << d) - 28'b1))) small_sh[0] = 1'b1;
                end
                // ---- add or subtract ----
                sres = sbig;
                if (sbig == ssmall) summ = big + small_sh;               // big >= small_sh always ⇒ no borrow
                else                summ = big - small_sh;
                if (summ == 28'b0) begin
                    fp32_add = 32'h00000000;                             // exact cancellation ⇒ +0.0 (RTNE)
                end else begin
                    // ---- renormalise ----
                    if (summ[27]) begin                                  // add carry: shift right 1
                        dropped = summ[0]; summ = summ >> 1; summ[0] = summ[0] | dropped; E = E + 1;
                    end
                    // subtract case: strip leading zeros with a priority-encoded MSB
                    // search + single barrel shift (clamped so E cannot drop below 1,
                    // i.e. results underflow into the subnormal encoding).
                    msbp = 0;
                    for (s = 0; s < 27; s = s + 1) if (summ[s]) msbp = s; // highest set bit (0..26)
                    want  = 26 - msbp;
                    avail = E - 1;
                    sh    = (want <= avail) ? want : avail;
                    summ  = summ << sh; E = E - sh;
                    // ---- RTNE on the low guard/round/sticky ----
                    kept   = {1'b0, summ[26:3]};                         // 24-bit significand
                    guard  = summ[2]; roundb = summ[1]; sticky = summ[0];
                    roundup = guard & (roundb | sticky | kept[0]);
                    kept   = kept + {24'b0, roundup};
                    if (kept[24]) begin kept = kept >> 1; E = E + 1; end // rounding carry
                    if (E >= 255) begin
                        fp32_add = {sres, 8'hFF, 23'b0};                 // overflow → inf
                    end else begin
                        EF = kept[23] ? E[7:0] : 8'h0;                   // subnormal if MSB clear
                        fp32_add = {sres, EF, kept[22:0]};
                    end
                end
            end
        end
    endfunction

    // =======================================================================
    // FP32 → FP16, round-to-nearest-even, overflow→inf, underflow→subnormal/0.
    // Verified bit-exact to numpy float16(float32) over 200k values incl the
    // overflow / subnormal / round-half boundaries.
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
            if (e == 8'hFF) begin
                fp32_to_fp16 = (|m) ? {s, 5'h1F, 10'b1000000000} : {s, 5'h1F, 10'b0};  // nan : inf
            end else if (e == 8'h0) begin
                fp32_to_fp16 = {s, 15'b0};                               // fp32 zero/subnormal → fp16 0
            end else begin
                sig = {1'b1, m};
                he  = e - 112;                                           // target fp16 biased exponent
                if (he >= 31) begin
                    fp32_to_fp16 = {s, 5'h1F, 10'b0};                    // overflow → inf
                end else begin
                    drop = (he <= 0) ? (14 - he) : 13;
                    if (drop > 25) drop = 25;
                    kept    = (sig >> drop);
                    guard   = (sig >> (drop - 1)) & 1'b1;                // sig>>24 == 0 ⇒ guard 0 for drop=25
                    sticky  = |(sig & (((32'b1 << (drop - 1)) - 32'b1)));
                    roundup = guard & (sticky | kept[0]);
                    kept    = kept + {12'b0, roundup};
                    if (he <= 0) begin
                        // subnormal magnitude sits directly in the low 15 bits; a
                        // round-up carry into bit10 seamlessly becomes the smallest
                        // normal (exp=1, mant=0), so no special case is needed.
                        fp32_to_fp16 = {s, 2'b00, kept};
                    end else begin
                        if (kept[11]) begin he = he + 1; kept = kept >> 1; end
                        if (he >= 31) fp32_to_fp16 = {s, 5'h1F, 10'b0};
                        else          fp32_to_fp16 = {s, he[4:0], kept[9:0]};
                    end
                end
            end
        end
    endfunction

    // ---- Stage 1: per-lane fp16×fp16→fp32 product (combinational), registered below ----
    wire [PW-1:0] prod [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_mul
            wire [FW-1:0] v_lane = v_data[gi*FW +: FW];
            assign prod[gi] = fp16_mul(a_data, v_lane);
        end
    endgenerate

    // Stage-1 pipeline registers: the fp32 product + a 1-cycle-delayed control copy.
    reg [PW-1:0] prod_reg [0:N-1];
    reg          v1;      // s_valid, delayed one cycle
    reg          last1;   // (s_valid & s_last), delayed one cycle

    // ---- Stage 2: accumulate the REGISTERED product into the fp32 accumulators ----
    reg  [PW-1:0] acc [0:N-1];
    wire [PW-1:0] acc_next [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_acc
            assign acc_next[gi] = fp32_add(acc[gi], prod_reg[gi]);
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            c_valid <= 1'b0;
            c_data  <= {N*FW{1'b0}};
            v1      <= 1'b0;
            last1   <= 1'b0;
            for (i = 0; i < N; i = i + 1) begin
                acc[i]      <= {PW{1'b0}};       // +0.0
                prod_reg[i] <= {PW{1'b0}};
            end
        end else begin
            c_valid <= 1'b0;                     // default: no result this cycle

            // Stage 1: latch this token's product + its control one cycle.
            v1    <= s_valid;
            last1 <= s_valid & s_last;
            if (s_valid)
                for (i = 0; i < N; i = i + 1)
                    prod_reg[i] <= prod[i];

            // Stage 2: accumulate the PREVIOUS token's fp32 product. On the pipelined
            // last, round the fp32 accumulator to fp16, emit, and clear (+0.0).
            if (v1) begin
                if (last1) begin
                    for (i = 0; i < N; i = i + 1) begin
                        c_data[i*FW +: FW] <= fp32_to_fp16(acc_next[i]);
                        acc[i]             <= {PW{1'b0}};
                    end
                    c_valid <= 1'b1;
                end else begin
                    for (i = 0; i < N; i = i + 1)
                        acc[i] <= acc_next[i];
                end
            end
        end
    end

endmodule
