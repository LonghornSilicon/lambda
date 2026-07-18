// token_importance_unit.sv
//
// H2O heavy-hitter eviction core for the LonghornSilicon KV cache (block 3).
//
// Maintains N_SLOTS KV-cache slots, each holding an accumulated attention-mass
// score (the "heavy-hitter oracle" statistic). Three operations:
//
//   ACC  (acc_valid) : score[acc_slot] += acc_weight   (saturating) — the per-step
//                      attention mass a cached token just received. 1 weight/cycle.
//   LOAD (ld_valid)  : score[ld_slot] := 0, valid := 1 — install a fresh token.
//   EVICT(evict_req) : serially scan the slots and return the VALID slot with the
//                      MINIMUM accumulated score — the token to evict — then free it.
//
// This is the eviction datapath; the recent-window protection and the budget
// bookkeeping are a thin control-layer wrapper (cf. how block 1 shipped just the
// ratio gate). ACC/LOAD complete in one cycle from S_IDLE; EVICT takes N_SLOTS+1
// cycles (a serialized argmin — one comparator, no wide combinational min tree, so
// it place-and-routes at a real clock).
//
// FF count derivation (closed form):
//   score[N_SLOTS] : N_SLOTS * SCORE_WIDTH
//   valid[N_SLOTS] : N_SLOTS
//   state          : 2
//   scan_idx       : SLOT_WIDTH
//   min_score      : SCORE_WIDTH
//   min_idx        : SLOT_WIDTH
//   evict_valid    : 1
//   evict_slot     : SLOT_WIDTH
//   Total: N_SLOTS*(SCORE_WIDTH+1) + SCORE_WIDTH + 3*SLOT_WIDTH + 3
//
// For N_SLOTS = 8, SCORE_WIDTH = 16, SLOT_WIDTH = 3:
//   8*17 + 16 + 3*3 + 3 = 136 + 16 + 9 + 3 = 164 FFs (register-count derivation)
//
// Synthesized (yosys 0.33, `synth -flatten`): 167 FFs — the +3 is one slot-index
// register yosys keeps un-merged. The CI FF-count gate pins the synthesized value
// (167); the derivation above is the analytic bound it tracks.
//
`timescale 1ns/1ps

module token_importance_unit #(
    parameter  integer N_SLOTS      = 8,
    parameter  integer SCORE_WIDTH  = 16,
    parameter  integer WEIGHT_WIDTH = 8,
    localparam integer SLOT_WIDTH   = (N_SLOTS <= 1) ? 1 : $clog2(N_SLOTS)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // Accumulate a token's received attention mass
    input  wire                          acc_valid,
    input  wire [SLOT_WIDTH-1:0]         acc_slot,
    input  wire [WEIGHT_WIDTH-1:0]       acc_weight,

    // Install a fresh token (resets that slot's score, marks it valid)
    input  wire                          ld_valid,
    input  wire [SLOT_WIDTH-1:0]         ld_slot,

    // Eviction request: pulse evict_req; evict_valid pulses with the victim slot
    input  wire                          evict_req,
    output reg                           evict_valid,
    output reg  [SLOT_WIDTH-1:0]         evict_slot,

    output wire                          busy
);
    localparam [SCORE_WIDTH-1:0] SCORE_MAX = {SCORE_WIDTH{1'b1}};
    localparam [SLOT_WIDTH-1:0]  LAST_IDX  = N_SLOTS - 1;

    // FSM
    localparam [1:0] S_IDLE = 2'd0, S_SCAN = 2'd1, S_DONE = 2'd2;
    reg [1:0] state;

    // Slot state
    reg [SCORE_WIDTH-1:0] score [0:N_SLOTS-1];
    reg                   valid [0:N_SLOTS-1];

    // Scan registers
    reg [SLOT_WIDTH-1:0]  scan_idx;
    reg [SCORE_WIDTH-1:0] min_score;
    reg [SLOT_WIDTH-1:0]  min_idx;

    assign busy = (state != S_IDLE);

    // Saturating add for the accumulate path
    wire [SCORE_WIDTH:0] sum_ext = {1'b0, score[acc_slot]} +
                                   {{(SCORE_WIDTH-WEIGHT_WIDTH+1){1'b0}}, acc_weight};
    wire [SCORE_WIDTH-1:0] sum_sat = sum_ext[SCORE_WIDTH] ? SCORE_MAX
                                                          : sum_ext[SCORE_WIDTH-1:0];

    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            scan_idx    <= {SLOT_WIDTH{1'b0}};
            min_score   <= {SCORE_WIDTH{1'b0}};
            min_idx     <= {SLOT_WIDTH{1'b0}};
            evict_valid <= 1'b0;
            evict_slot  <= {SLOT_WIDTH{1'b0}};
            for (k = 0; k < N_SLOTS; k = k + 1) begin
                score[k] <= {SCORE_WIDTH{1'b0}};
                valid[k] <= 1'b0;
            end
        end else begin
            evict_valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    // ACC and LOAD can happen the same cycle (different slots typical)
                    if (acc_valid) score[acc_slot] <= sum_sat;
                    if (ld_valid) begin
                        score[ld_slot] <= {SCORE_WIDTH{1'b0}};
                        valid[ld_slot] <= 1'b1;
                    end
                    if (evict_req) begin
                        state     <= S_SCAN;
                        scan_idx  <= {SLOT_WIDTH{1'b0}};
                        // seed min with slot 0 if valid, else max sentinel
                        min_score <= valid[0] ? score[0] : SCORE_MAX;
                        min_idx   <= {SLOT_WIDTH{1'b0}};
                    end
                end
                S_SCAN: begin
                    // compare slot scan_idx, then advance
                    if (valid[scan_idx] && (score[scan_idx] < min_score)) begin
                        min_score <= score[scan_idx];
                        min_idx   <= scan_idx;
                    end
                    if (scan_idx == LAST_IDX) begin
                        state <= S_DONE;
                    end else begin
                        scan_idx <= scan_idx + 1'b1;
                    end
                end
                S_DONE: begin
                    evict_valid     <= 1'b1;
                    evict_slot      <= min_idx;
                    valid[min_idx]  <= 1'b0;   // free the evicted slot
                    state           <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
