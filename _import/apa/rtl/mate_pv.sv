// mate_pv.sv
//
// Synthesizable INT8 P·V MAC tile — the token-reduction vector-MAC core of the
// LonghornSilicon "Lambda" MatE matrix engine. Computes one attention output row
//
//     o[n] = Σ_t  A[t] · V[t][n]        (n = 0 .. N-1 head-dim channels)
//
// in signed INT8 × INT8 → signed INT32, NO saturation — bit-exact to the ACU MAC
// array reference sw/reference_model/mac_array_ref.{hpp,py} `matmul_int8` for M=1.
//
// Why INT32 (not INT24): the P·V tile reduces over the TOKEN dimension, so the
// accumulator width scales with context length. A maximally-flat causal row of
// length L drives every code to ±127, so |acc| ≤ 127·127·L → 14+ceil(log2 L) bits;
// INT24 overflows past ~520 tokens, INT32 covers ~133k. The K-axis (hidden-dim)
// GEMM accumulators are INT24; the token-reduction P·V accumulator is INT32.
// See adaptive-precision-attention/analysis/pv_accumulator_width.py + arch.yml.
//
// This is the ACU's INT8 tile (precision_controller.d_fp16 == 0). The FP16 tile is
// tolerance-only (see MAC_ARRAY_DESIGN.md) and is not in this integer datapath.
//
// Interface (house style — streaming valid/last, 1-cycle result pulse, like
// precision_controller): present one token per clock with s_valid=1, its scalar
// A-code on a_data and its N-wide packed V-row on v_data; assert s_last=1 on the
// final token of the row. c_valid pulses when the row's Σ is ready, with the N
// int32 results on c_data. Accumulators auto-reset after the row for the next one.
//
// PIPELINED: the per-lane product is registered (prod_reg) between the multiply
// and the accumulate, so the ~15 ns combinational mult→add path is split into two
// ~half stages (mult in stage 1, int32 add in stage 2) — this lets the block close
// a ~2× faster clock. The trade is one extra cycle of latency: the row sum is
// emitted 2 cycles after s_last (was 1). The VALUE is unchanged — same Σ_t A·V,
// just delayed — so it stays bit-exact to matmul_int8. Downstream uses the c_valid
// handshake, so the extra cycle is transparent.
//
// Latency  : 2 cycles after s_last.  Throughput: 1 token/cycle, 1 row per K tokens.
// Synthesis: registered product + registered int32 accumulators; no latches.

`timescale 1ns/1ps

module mate_pv #(
    parameter integer N     = 8,    // head-dim channels computed in parallel (lanes)
    parameter integer AW    = 8,    // A (attention-prob) code width, signed
    parameter integer VW    = 8,    // V (value) code width, signed
    parameter integer ACC_W = 32    // token-reduction accumulator width (INT32)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    s_valid,   // a token is being presented
    input  wire signed [AW-1:0]    a_data,    // A[t]      : this token's attention code
    input  wire        [N*VW-1:0]  v_data,    // V[t][0..N-1]: this token's value row (packed signed)
    input  wire                    s_last,    // last token of the output row

    output reg                     c_valid,   // pulses cycle after s_last
    output reg  signed [N*ACC_W-1:0] c_data   // N int32 dot-product results
);

    genvar gi;
    integer i;

    // ---- Stage 1: per-lane signed product (combinational), registered below ----
    // int8×int8 → int16 product; sign-extended in the stage-2 add. Registering the
    // product is what breaks the long mult→add path.
    wire signed [AW+VW-1:0] prod [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_mul
            wire signed [VW-1:0] v_lane = $signed(v_data[gi*VW +: VW]);
            assign prod[gi] = a_data * v_lane;                         // signed int16
        end
    endgenerate

    // Stage-1 pipeline registers: the product and a 1-cycle-delayed copy of the
    // control (valid / last) that travels with it.
    reg signed [AW+VW-1:0] prod_reg [0:N-1];
    reg                    v1;      // s_valid, delayed one cycle
    reg                    last1;   // (s_valid & s_last), delayed one cycle

    // ---- Stage 2: accumulate the REGISTERED product into the int32 accumulators ----
    reg signed [ACC_W-1:0] acc [0:N-1];
    wire signed [ACC_W-1:0] acc_next [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_acc
            assign acc_next[gi] = acc[gi] + $signed(prod_reg[gi]);     // sign-extended add
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            c_valid <= 1'b0;
            c_data  <= {N*ACC_W{1'b0}};
            v1      <= 1'b0;
            last1   <= 1'b0;
            for (i = 0; i < N; i = i + 1) begin
                acc[i]      <= {ACC_W{1'b0}};
                prod_reg[i] <= {(AW+VW){1'b0}};
            end
        end else begin
            c_valid <= 1'b0;                       // default: no result this cycle

            // Stage 1: latch this token's product + its control one cycle.
            v1    <= s_valid;
            last1 <= s_valid & s_last;
            if (s_valid)
                for (i = 0; i < N; i = i + 1)
                    prod_reg[i] <= prod[i];

            // Stage 2: accumulate the PREVIOUS token's product (nonblocking reads of
            // v1/last1/prod_reg see last cycle's values). On the pipelined last, emit
            // Σ_t A·V and clear for the next row.
            if (v1) begin
                if (last1) begin
                    for (i = 0; i < N; i = i + 1) begin
                        c_data[i*ACC_W +: ACC_W] <= acc_next[i];
                        acc[i]                   <= {ACC_W{1'b0}};
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
