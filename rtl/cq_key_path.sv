// cq_key_path.sv — streaming per-channel grouped KEY codec (contract §3-§4).
//
// The key datapath the top will instantiate. Keys can't be scaled until their
// whole group is seen, so this buffers the group (residual_buffer, fp16), takes
// the per-channel max over the group (amax_unit, key mode), converts to fp16
// per-channel scales (D× cq_scale_unit, banked in scale_bank), then walks the
// buffered tokens quantizing each keep-channel (D× cq_quant_unit) and packing
// the INT4 keep codes (contract §5). Outlier channels (outlier_mask) are excluded
// from the INT4 path and carried FP16 in a sidecar by the top (contract §4).
//
// Group FSM:  COLLECT (buffer + amax accumulate) -> SCALE (freeze per-channel
// scales) -> EMIT (one buffered token per cycle: quantize keep channels + pack)
// -> DONE. Decompress is combinational (D× cq_dequant_unit, per-channel scale).
//
// Verified vs golden key_scales/key_payload/expected_k_hat/sidecar (full g=G and
// partial g<G groups, CQ-4/CQ-4+) by tb_key_path.sv (make sim_kpath). The cq
// cores are behavioral golden-equivalents (fp16 hardware lowering is P4b).

`include "cq_fp_pkg.sv"
`default_nettype none

module cq_key_path #(
    parameter int D  = 64,     // head dim
    parameter int DW = 16,     // fp16 element width
    parameter int G  = 128     // key group size
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [D-1:0]            outlier_mask,   // bit c = 1 -> outlier (excluded from INT4)

    // ---- compress: stream key tokens (group_start on 1st, group_last on last) ----
    input  wire                    in_valid,
    input  wire [D*DW-1:0]         in_vec,
    input  wire                    group_start,
    input  wire                    group_last,

    // ---- per-group outputs ----
    output reg                     group_valid,    // pulses when the group finished emitting
    output wire [D*DW-1:0]         scales_bus,     // D per-channel fp16 scales (keep valid)
    output reg  [$clog2(G+1)-1:0]  g_out,          // tokens in the group

    // ---- per-token payload emit (during EMIT) ----
    output wire                    tok_valid,
    output wire [$clog2(G)-1:0]    tok_idx,
    output wire [(D/2)*8-1:0]      tok_pay,        // packed keep-channel INT4 codes (nk/2 bytes)
    output wire [D*8-1:0]          tok_codes,      // compacted keep codes (byte i = i-th keep)

    // ---- decompress: combinational, per-channel dequant ----
    input  wire [D*8-1:0]          dec_codes,      // per original channel (byte c)
    input  wire [D*DW-1:0]         dec_scales,     // per original channel scale
    output wire [D*32-1:0]         dec_hat
);

    localparam [1:0] S_COLLECT = 2'd0, S_SCALE = 2'd1, S_EMIT = 2'd2, S_DONE = 2'd3;
    reg [1:0]                 state;
    reg [$clog2(G+1)-1:0]     icnt;      // tokens seen so far in the current group
    reg [$clog2(G+1)-1:0]     g_cnt;     // frozen group size
    reg [$clog2(G)-1:0]       emit_cnt;

    wire collecting = (state == S_COLLECT);

    // ---- residual buffer (fp16 group hold) ----
    wire [$clog2(G+1)-1:0] rb_fill;
    wire [D*DW-1:0]        rb_rdvec;
    residual_buffer #(.DIM(D), .DW(DW), .G(G)) u_rb (
        .clk(clk), .rst_n(rst_n),
        .wr_valid(collecting & in_valid), .wr_vec(in_vec),
        .clear(collecting & in_valid & group_start), .fill(rb_fill),
        .rd_idx(emit_cnt), .rd_vec(rb_rdvec)
    );

    // ---- per-channel amax over the group ----
    wire [D*DW-1:0] amax_chan;
    wire            amax_ov;
    amax_unit #(.DIM(D), .DW(DW)) u_amax (
        .clk(clk), .rst_n(rst_n),
        .in_valid(collecting & in_valid), .vec(in_vec), .mode_channel(1'b1),
        .group_start(collecting & in_valid & group_start),
        .group_done (collecting & in_valid & group_last),
        .scale_token(), .scale_chan(amax_chan), .out_valid(amax_ov)
    );

    // ---- amax -> fp16 scale, per channel (keys are INT4) ----
    wire [D*DW-1:0] csc;    // computed per-channel scales
    genvar c;
    generate
        for (c = 0; c < D; c = c + 1) begin : g_scale
            cq_scale_unit u_sc (
                .amax_f16(amax_chan[c*DW +: DW]), .bits(4'd4),
                .scale_f16(csc[c*DW +: DW])
            );
        end
    endgenerate

    // ---- scale bank: freeze the group's per-channel scales in S_SCALE ----
    scale_bank #(.DIM(D), .DW(DW)) u_sb (
        .clk(clk), .rst_n(rst_n),
        .wr(state == S_SCALE), .scales_in(csc), .scales_out(scales_bus)
    );

    // ---- quantize the current buffered token's channels (parallel) ----
    wire signed [7:0] code_c [0:D-1];
    generate
        for (c = 0; c < D; c = c + 1) begin : g_quant
            cq_quant_unit u_q (
                .x_f16(rb_rdvec[c*DW +: DW]), .scale_f16(scales_bus[c*DW +: DW]),
                .bits(4'd4), .code(code_c[c])
            );
        end
    endgenerate

    // ---- gather keep-channel codes + pack INT4 (little-endian nibble order) ----
    reg [(D/2)*8-1:0]     pay_c;
    reg [D*8-1:0]         codes_c;
    reg [$clog2(D):0]     kidx;
    integer               cc;
    always @* begin
        pay_c   = '0;
        codes_c = '0;
        kidx    = '0;
        for (cc = 0; cc < D; cc = cc + 1) begin
            if (!outlier_mask[cc]) begin
                codes_c[kidx*8 +: 8] = code_c[cc];
                if (kidx[0] == 1'b0) pay_c[(kidx>>1)*8     +: 4] = code_c[cc][3:0];
                else                 pay_c[(kidx>>1)*8 + 4 +: 4] = code_c[cc][3:0];
                kidx = kidx + 1;
            end
        end
    end

    assign tok_valid = (state == S_EMIT);
    assign tok_idx   = emit_cnt;
    assign tok_pay   = pay_c;
    assign tok_codes = codes_c;

    // ---- FSM ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_COLLECT;
            icnt        <= '0;
            g_cnt       <= '0;
            emit_cnt    <= '0;
            group_valid <= 1'b0;
            g_out       <= '0;
        end else begin
            group_valid <= 1'b0;
            case (state)
                S_COLLECT: begin
                    if (in_valid) begin
                        icnt <= group_start ? 'd1 : icnt + 'd1;
                        if (group_last) begin
                            g_cnt <= group_start ? 'd1 : icnt + 'd1;
                            state <= S_SCALE;
                        end
                    end
                end
                S_SCALE: begin
                    // csc valid (amax_ov), scale_bank latches this edge
                    emit_cnt <= '0;
                    g_out    <= g_cnt;
                    state    <= S_EMIT;
                end
                S_EMIT: begin
                    if (emit_cnt == g_cnt - 'd1) state <= S_DONE;
                    else                         emit_cnt <= emit_cnt + 'd1;
                end
                S_DONE: begin
                    group_valid <= 1'b1;
                    state       <= S_COLLECT;
                end
                default: state <= S_COLLECT;
            endcase
        end
    end

    // ---- decompress: per-channel dequant (outlier channels handled by the top) ----
    generate
        for (c = 0; c < D; c = c + 1) begin : g_dequant
            cq_dequant_unit u_d (
                .code(dec_codes[c*8 +: 8]), .scale_f16(dec_scales[c*DW +: DW]),
                .xhat_f32(dec_hat[c*32 +: 32])
            );
        end
    endgenerate

endmodule

`default_nettype wire
