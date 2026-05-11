// precision_controller.sv
//
// Streaming precision controller for adaptive attention.
//
// Decision (no division gate):
//   max(|S|) * N  >  THRESHOLD * sum(|S|)
//
// LHS = {max_acc, LOG2_N zeros}   — free left-shift, wire routing only
// RHS = THRESHOLD * sum_acc       — constant multiply → shifts + adder tree
//
// Interface:
//   Feed one signed score per clock with s_valid=1.
//   Assert s_last=1 on the final score of each tile.
//   d_valid pulses high the cycle after s_last; d_fp16 carries the decision.
//   Accumulators auto-reset for the next tile on s_last.
//
// Latency  : 1 cycle after s_last
// Throughput: 1 score/cycle, 1 decision per N = BLOCK_M*BLOCK_N cycles
//
// Synthesis targets:
//   Vivado  : xczu9eg (ZCU102) or xczu7ev (ZCU104) — target 200 MHz
//   Cadence Genus : see constraints/timing.sdc — target 500 MHz (TSMC 28nm)

`timescale 1ns/1ps

module precision_controller #(
    parameter integer BLOCK_M     = 64,
    parameter integer BLOCK_N     = 64,
    parameter integer SCORE_WIDTH = 8,
    parameter integer THRESHOLD   = 10
) (
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          s_valid,
    input  wire signed [SCORE_WIDTH-1:0] s_data,
    input  wire                          s_last,

    output reg                           d_valid,
    output reg                           d_fp16
);

    // Derived constants
    localparam integer N      = BLOCK_M * BLOCK_N;
    localparam integer LOG2_N = $clog2(N);
    // Sum accumulator: SCORE_WIDTH + LOG2_N bits (max value = 127 * 4096 = 520192)
    localparam integer SUM_W  = SCORE_WIDTH + LOG2_N;
    // Comparison width: sum * THRESHOLD needs 4 more bits (THRESHOLD=10 < 16 = 2^4)
    localparam integer CMP_W  = SUM_W + 4;

    // Absolute value of incoming score (two's complement negation)
    wire [SCORE_WIDTH-1:0] abs_score;
    assign abs_score = s_data[SCORE_WIDTH-1] ? (~s_data + 1'b1) : s_data;

    // Accumulators
    reg [SCORE_WIDTH-1:0] max_acc;
    reg [SUM_W-1:0]       sum_acc;

    // Combinatorial next-state (includes current score so decision is final)
    wire [SCORE_WIDTH-1:0] max_next;
    wire [SUM_W-1:0]       sum_next;
    assign max_next = (s_valid && (abs_score > max_acc)) ? abs_score : max_acc;
    assign sum_next = s_valid ? (sum_acc + {{(SUM_W-SCORE_WIDTH){1'b0}}, abs_score})
                              : sum_acc;

    // LHS = max_next * N = max_next left-shifted by LOG2_N (free: wire routing)
    // RHS = THRESHOLD * sum_next = 10 * sum = (sum<<3) + (sum<<1)
    // Both widened to CMP_W for safe comparison
    wire [CMP_W-1:0] lhs, rhs;
    assign lhs = {{(CMP_W - SCORE_WIDTH - LOG2_N){1'b0}}, max_next, {LOG2_N{1'b0}}};
    assign rhs = ({{(CMP_W-SUM_W-3){1'b0}}, sum_next, 3'b000})   // sum << 3 (*8)
               + ({{(CMP_W-SUM_W-1){1'b0}}, sum_next, 1'b0  });   // sum << 1 (*2)

    always @(posedge clk) begin
        if (!rst_n) begin
            max_acc <= {SCORE_WIDTH{1'b0}};
            sum_acc <= {SUM_W{1'b0}};
            d_valid <= 1'b0;
            d_fp16  <= 1'b0;
        end else begin
            d_valid <= 1'b0;

            if (s_valid) begin
                max_acc <= max_next;
                sum_acc <= sum_next;

                if (s_last) begin
                    d_fp16  <= (lhs > rhs);
                    d_valid <= 1'b1;
                    // Reset for next tile (last-write wins over max/sum updates above)
                    max_acc <= {SCORE_WIDTH{1'b0}};
                    sum_acc <= {SUM_W{1'b0}};
                end
            end
        end
    end

endmodule
