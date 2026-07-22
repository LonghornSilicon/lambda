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
// For N_SLOTS = 8, SCORE_WIDTH = 8, SLOT_WIDTH = 3:
//   8*9 + 8 + 3*3 + 3 = 72 + 8 + 9 + 3 = 92 FFs (register-count derivation)
//
// SCORE_WIDTH = 8 is set from the accumulator-bit-width study
// (docs/findings/h2o-deep-analysis.md): 8 bits is loss-free for the eviction
// ranking on Qwen2 (-0.002 acc_norm). Fewer score bits also shrink the argmin
// read-mux, which is what drove the Sky130 max-transition closure.
// The CI FF-count gate pins the synthesized value; the derivation is the analytic
// bound it tracks (yosys keeps a few slot-index FFs un-merged, ~+3).
//
`timescale 1ns/1ps

module token_importance_unit #(
    parameter  integer N_SLOTS      = 8,
    parameter  integer SCORE_WIDTH  = 8,
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

    // TIER HANDSHAKE to the KV Cache Engine (block 2).
    // Per-slot importance tier, driven combinationally from the accumulated mass:
    //   tier_keep[k] = 1  -> heavy hitter: KVCE keeps its VALUE at high precision (CQ-8)
    //   tier_keep[k] = 0  -> demote:       KVCE stores its VALUE at CQ-4
    // Eviction (dropping K+V) is signalled separately via evict_slot above.
    // NB: the tier is a per-token VALUE-precision lever only — keys stay uniform
    // per-channel (per-token key demotion collapses GQA; see docs). Emitted as N
    // parallel comparators (no read-mux) so it adds no fanout to the argmin path.
    input  wire [SCORE_WIDTH-1:0]        tier_threshold,
    output wire [N_SLOTS-1:0]            tier_keep,

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

    // Per-slot importance tier: parallel comparators, one per slot (no read-mux, so
    // no added fanout on score[]). A valid slot at/above the threshold is a KEEP.
    genvar gi;
    generate
        for (gi = 0; gi < N_SLOTS; gi = gi + 1) begin : g_tier
            assign tier_keep[gi] = valid[gi] && (score[gi] >= tier_threshold);
        end
    endgenerate

    // Saturating add — instantiated PER SLOT below (distributed accumulators). There
    // is deliberately no shared sum result broadcast to all slots: a single adder
    // output feeding N_SLOTS register-input muxes was the high-fanout net that blew
    // max-transition in Sky130. Each slot owns its adder (a tiny WEIGHT_WIDTH add),
    // driven only by its own score and the broadcast acc_weight (easy to buffer).
    function [SCORE_WIDTH-1:0] sat_add;
        input [SCORE_WIDTH-1:0] s;
        input [WEIGHT_WIDTH-1:0] w;
        reg [SCORE_WIDTH:0] ext;
        begin
            ext = {1'b0, s} + {{(SCORE_WIDTH-WEIGHT_WIDTH+1){1'b0}}, w};
            sat_add = ext[SCORE_WIDTH] ? SCORE_MAX : ext[SCORE_WIDTH-1:0];
        end
    endfunction

    integer k;
    always @(posedge clk) begin
        // Only the control FFs are reset: state (must start IDLE), valid[] (slots
        // must start empty), evict_valid (no spurious victim). The score[] datapath
        // and the scan scratch regs (scan_idx/min_*) are intentionally NOT reset —
        // a slot's score is zeroed by LOAD before it is ever read (its valid bit is
        // 0 until then), and the scan regs are seeded at evict_req. This keeps rst_n
        // fanout ~11 instead of ~113, so no delay-buffer reset tree (and its slew
        // violations) is needed.
        if (!rst_n) begin
            state       <= S_IDLE;
            evict_valid <= 1'b0;
            for (k = 0; k < N_SLOTS; k = k + 1)
                valid[k] <= 1'b0;
        end else begin
            evict_valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    // Distributed per-slot update: each slot owns its adder, so no
                    // shared result fans out across the register file. LOAD (zero +
                    // set valid) has priority over ACC on the same slot, matching the
                    // original last-write-wins ordering.
                    for (k = 0; k < N_SLOTS; k = k + 1) begin
                        if (ld_valid && (ld_slot == k[SLOT_WIDTH-1:0])) begin
                            score[k] <= {SCORE_WIDTH{1'b0}};
                            valid[k] <= 1'b1;
                        end else if (acc_valid && (acc_slot == k[SLOT_WIDTH-1:0])) begin
                            score[k] <= sat_add(score[k], acc_weight);
                        end
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
