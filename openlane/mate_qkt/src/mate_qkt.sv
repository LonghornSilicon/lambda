// mate_qkt.sv
//
// Synthesizable Q·Kᵀ decode-scoring engine — the score-generation datapath of the
// LonghornSilicon "Lambda" MatE matrix engine. For DECODE there is ONE query token;
// this block scores it against L cached keys:
//
//     score[l] = round_fp16( Σ_d  Q[d] · K[l][d] )      (l = 0 .. N-1 keys)
//
// realising the ACU MAC-array reference's FP16 Q·Kᵀ contract (sw/reference_model/
// mac_array_ref.{hpp,py} `matmul_fp16` — used as `matmul_fp16(Q, K.T)` in
// integration_example.py, fp16_accumulator_bits=32):
//   - Q arrives INT8 (per-tensor/-tile symmetric quant, |q| ≤ 127) and is promoted
//     to fp16 EXACTLY (any integer with |q| ≤ 2048 is representable in binary16).
//   - K is per-channel-dequantized to fp16 by the KVE key path (cq_dequant_f16 =
//     round_fp16(code · fp16 per-channel scale)); MatE consumes it as fp16.
//   - each fp16(Q[d]) × fp16(K[l][d]) product is EXACT in fp32, the head-dim (D)
//     reduction is accumulated in fp32 with round-to-nearest-even at every add, and
//     the emitted score is rounded to fp16 (RTNE, overflow→inf) exactly ONCE.
//
// This is the same fp16 datapath as `mate_pv_fp16` (fp16_mul / fp32_add /
// fp32_to_fp16 are byte-identical), reused here for the Q·Kᵀ reduction. The only
// difference is the reduced operand: the P·V tile reduces attention probs · V over
// TOKENS; the Q·Kᵀ tile reduces one query's Q · K over head-dim CHANNELS, with the
// int8 Q code promoted to fp16 at the input. Accumulation is sequential in channel
// arrival order — the faithful streaming-MAC realisation of the reference's "fp32
// accumulate, round-to-fp16-once" semantics (see docs/mate_qkt_rtl.md). numpy BLAS
// `@` sums in a different pairwise order, so it agrees to a few ULP — within the
// FP16 path's rel_err < 5e-3.
//
// INTERFACE (house style — streaming valid/last, registered-product pipeline, like
// mate_pv / mate_pv_fp16): present one HEAD-DIM CHANNEL per clock with s_valid=1 —
// the query's scalar INT8 code Q[d] on a_data, and the d-th channel of every key
// (the N-wide packed fp16 vector K[0..N-1][d]) on k_data; assert s_last=1 on the
// final channel (d = D-1). c_valid pulses when the row of scores is ready, with the
// N fp16 scores on c_data. Accumulators auto-reset (to +0.0) after the row.
//
// PIPELINED (mirrors mate_pv): the per-lane int8→fp16→fp32 product is registered
// (prod_reg) between the multiply and the fp32 accumulate. Scores emerge 2 cycles
// after s_last (value unchanged — same sequential Σ, just delayed).
//
// Latency  : 2 cycles after s_last.  Throughput: 1 channel/cycle, 1 score-row per D.
// Synthesis: registered fp32 product + registered fp32 accumulators; no latches.

`timescale 1ns/1ps

module mate_qkt #(
    parameter integer N   = 8,    // number of cached keys scored in parallel (lanes)
    parameter integer QW  = 8,    // INT8 query code width (signed)
    parameter integer FW  = 16,   // fp16 key/score width (IEEE-754 binary16)
    parameter integer PW  = 32    // internal fp32 product / accumulator width (binary32)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    s_valid,   // a head-dim channel is being presented
    input  wire signed [QW-1:0]    a_data,    // Q[d]        : this channel's INT8 query code
    input  wire        [N*FW-1:0]  k_data,    // K[0..N-1][d]: this channel of every key (packed fp16)
    input  wire                    s_last,    // last head-dim channel of the query (d = D-1)

    output reg                     c_valid,   // pulses 2 cycles after s_last
    output reg  [N*FW-1:0]         c_data     // N fp16 Q·Kᵀ scores (round-to-fp16)
);

    genvar gi;
    integer i;

    // =======================================================================
    // INT8 → FP16 (exact).  Promote a signed int8 query code to binary16; any
    // integer with |q| ≤ 2048 is exactly representable, so |q| ≤ 127 always is.
    // =======================================================================
    function automatic [15:0] int8_to_fp16;
        input signed [7:0] q;
        reg        s;
        reg [7:0]  mag;         // |q| (0..128)
        reg [16:0] magsh;       // mag left-aligned so the leading 1 sits at bit10
        integer    p, k;
        reg [4:0]  be;
        begin
            if (q == 8'sd0) begin
                int8_to_fp16 = 16'h0000;
            end else begin
                s   = q[7];
                mag = q[7] ? (~q + 1'b1) : q;                 // two's-complement |q|
                p = 0;
                for (k = 0; k < 8; k = k + 1) if (mag[k]) p = k;   // highest set bit (0..7)
                be    = p + 15;                              // fp16 biased exponent
                magsh = mag << (10 - p);                     // MSB (the implicit 1) → bit10
                int8_to_fp16 = {s, be, magsh[9:0]};          // drop implicit bit, keep 10 mantissa
            end
        end
    endfunction

    // =======================================================================
    // FP16 × FP16 → FP32 (exact).  Signed IEEE-754 binary16 multiply; result is
    // an fp32 bit-pattern.  Handles zero / subnormal / inf / nan operands.
    // (Byte-identical to mate_pv_fp16.fp16_mul.)
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
                if (ea == 5'h0) begin                                     // subnormal a (ma != 0)
                    msba = 0;
                    for (k = 0; k < 10; k = k + 1) if (ma[k]) msba = k;
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
    // FP32 + FP32 → FP32, correctly-rounded round-to-nearest-even.
    // (Byte-identical to mate_pv_fp16.fp32_add.)
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
                    // search + single barrel shift (clamped so E cannot drop below 1).
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
    // (Byte-identical to mate_pv_fp16.fp32_to_fp16.)
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
                        fp32_to_fp16 = {s, 2'b00, kept};                 // subnormal→normal transition is seamless
                    end else begin
                        if (kept[11]) begin he = he + 1; kept = kept >> 1; end
                        if (he >= 31) fp32_to_fp16 = {s, 5'h1F, 10'b0};
                        else          fp32_to_fp16 = {s, he[4:0], kept[9:0]};
                    end
                end
            end
        end
    endfunction

    // ---- Stage 1: per-key int8(Q)→fp16 × fp16(K) → fp32 product, registered below ----
    wire [15:0]   q_fp16 = int8_to_fp16(a_data);
    wire [PW-1:0] prod [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_mul
            wire [FW-1:0] k_lane = k_data[gi*FW +: FW];
            assign prod[gi] = fp16_mul(q_fp16, k_lane);
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

            // Stage 1: latch this channel's product + its control one cycle.
            v1    <= s_valid;
            last1 <= s_valid & s_last;
            if (s_valid)
                for (i = 0; i < N; i = i + 1)
                    prod_reg[i] <= prod[i];

            // Stage 2: accumulate the PREVIOUS channel's fp32 product. On the pipelined
            // last, round each fp32 score accumulator to fp16, emit, and clear (+0.0).
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
