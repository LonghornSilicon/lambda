// cq_key_path.sv — streaming per-channel grouped KEY codec (contract §3-§4).
//
// The key datapath the top instantiates. Keys can't be scaled until their whole
// group is seen, so this buffers the group (residual_buffer, fp16), takes the
// per-channel max over the group (amax_unit, key mode), converts to fp16
// per-channel scales (D× cq_scale_unit, banked in scale_bank), then walks the
// buffered tokens quantizing each channel and packing the INT4 keep codes
// (contract §5). Outlier channels (outlier_mask) are excluded from the INT4 path
// and carried FP16 in a sidecar by the top (contract §4).
//
// Group FSM:  COLLECT (buffer + amax accumulate) -> SCALE (freeze per-channel
// scales) -> [per token: QSTART/QWAIT over D channels on ONE shared quant unit]
// -> TEMIT (pulse tok_valid with the packed token) -> DONE. Decompress is
// combinational (D× cq_dequant_unit, per-channel scale).
//
// AREA (P4b serialization): the quant core carries the fp16 divider, so instead
// of D parallel combinational quant units (cq_quant_comb, D dividers -> a single
// giant divide cone that hung OpenLane's resizer) this SERIALIZES one shared
// cq_quant_unit_syn across the D channels of each buffered token — exactly like
// cq_value_path's S_QSTART/S_QWAIT handshake. Per-channel codes shift into a flat
// register (channel c -> byte c after D shifts, same trick as the value path),
// then the keep-channel compaction packs INT4 from it. ~D fewer dividers, at
// D-quant-cycles per emitted token.
//
// Verified vs golden key_scales/key_payload/expected_k_hat/sidecar (full g=G and
// partial g<G groups, CQ-4/CQ-4+) by tb_key_path.sv (make sim_kpath). The cq
// cores are the synthesizable fp16 fixed-function units in cq_units_syn.sv (P4b).

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

    // ---- per-token payload emit (pulses once per token during TEMIT) ----
    output wire                    tok_valid,
    output wire [$clog2(G)-1:0]    tok_idx,
    output wire [(D/2)*8-1:0]      tok_pay,        // packed keep-channel INT4 codes (nk/2 bytes)
    output wire [D*8-1:0]          tok_codes,      // compacted keep codes (byte i = i-th keep)
    output wire [D*DW-1:0]         emit_vec,       // the emitting token's raw fp16 (for outlier sidecar capture)

    // ---- decompress: ONE shared dequant, indexed by dec_idx (one channel/beat) ----
    input  wire [D*8-1:0]          dec_codes,      // per original channel (byte c)
    input  wire [D*DW-1:0]         dec_scales,     // per original channel scale
    input  wire [$clog2(D)-1:0]    dec_idx,        // channel to reconstruct
    output wire [31:0]             dec_hat         // that channel's fp32
);
    localparam int CW = $clog2(D);

    // ---- FSM ----
    localparam [2:0] S_COLLECT = 3'd0, S_SCALE = 3'd1, S_QSTART = 3'd2,
                     S_QWAIT   = 3'd3, S_TEMIT = 3'd4, S_DONE = 3'd5;
    reg [2:0]                 state;
    reg [$clog2(G+1)-1:0]     icnt;      // tokens seen so far in the current group
    reg [$clog2(G+1)-1:0]     g_cnt;     // frozen group size
    reg [$clog2(G)-1:0]       emit_cnt;  // current token being emitted
    reg [CW:0]                ch;        // channel counter within the current token (0..D-1)
    reg [CW:0]                sch;       // channel counter for the scale walk (0..D-1)

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

    // ---- amax -> fp16 scale: ONE shared scale unit, walked over the D channels in
    // S_SCALE (writes scale_bank one channel/cycle). Serializing this avoids D
    // parallel fixed-point dividers, which blew up yosys synth (share/alumacc) the
    // same way the parallel quant/dequant did — mirror the value-path serialization.
    wire [DW-1:0] scale_one;
    cq_scale_unit_syn u_sc (
        .amax_f16(amax_chan[sch*DW +: DW]), .bits(4'd4), .scale_f16(scale_one)
    );

    // ---- scale bank: per-channel write during the S_SCALE walk ----
    scale_bank #(.DIM(D), .DW(DW)) u_sb (
        .clk(clk), .rst_n(rst_n),
        .wr_en(state == S_SCALE), .wr_idx(sch[CW-1:0]), .wr_scale(scale_one),
        .scales_out(scales_bus)
    );

    // ---- ONE shared quantizer, walked across the D channels of the current token.
    // Per channel: pulse q_start with (rb_rdvec[ch], scales_bus[ch]); wait q_done;
    // shift the code into codes_all. After D shifts channel c sits at byte c (same
    // in-order shift alignment as cq_value_path). No indexed write -> checker-clean.
    reg               q_start;
    wire              q_done;
    wire signed [7:0] q_code;
    cq_quant_unit_syn u_q (
        .clk(clk), .rst_n(rst_n), .start(q_start),
        .x_f16(rb_rdvec[ch*DW +: DW]), .scale_f16(scales_bus[ch*DW +: DW]),
        .bits(4'd4), .code(q_code), .done(q_done)
    );

    reg [D*8-1:0] codes_all;   // byte c holds channel c's INT4 code (all D channels)

    // ---- gather keep-channel codes + pack INT4 (little-endian nibble order) ----
    // Combinational compaction of the registered per-channel codes: keep channel i
    // -> byte i of tok_codes, nibble i of tok_pay. Outlier channels excluded.
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
                codes_c[kidx*8 +: 8] = codes_all[cc*8 +: 8];
                if (kidx[0] == 1'b0) pay_c[(kidx>>1)*8     +: 4] = codes_all[cc*8 +: 4];
                else                 pay_c[(kidx>>1)*8 + 4 +: 4] = codes_all[cc*8 +: 4];
                kidx = kidx + 1;
            end
        end
    end

    assign tok_valid = (state == S_TEMIT);   // one-cycle pulse per token
    assign tok_idx   = emit_cnt;
    assign tok_pay   = pay_c;
    assign tok_codes = codes_c;
    assign emit_vec  = rb_rdvec;              // token emit_cnt's fp16 (valid throughout its emit)

    // ---- FSM ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_COLLECT;
            icnt        <= '0;
            g_cnt       <= '0;
            emit_cnt    <= '0;
            ch          <= '0;
            sch         <= '0;
            codes_all   <= '0;
            q_start     <= 1'b0;
            group_valid <= 1'b0;
            g_out       <= '0;
        end else begin
            group_valid <= 1'b0;
            q_start     <= 1'b0;
            case (state)
                S_COLLECT: begin
                    if (in_valid) begin
                        icnt <= group_start ? 'd1 : icnt + 'd1;
                        if (group_last) begin
                            g_cnt <= group_start ? 'd1 : icnt + 'd1;
                            sch   <= '0;
                            state <= S_SCALE;
                        end
                    end
                end
                S_SCALE: begin
                    // walk the shared scale unit over the D channels; scale_bank
                    // writes bank[sch] <= scale_one each cycle (wr_en = state==S_SCALE).
                    if (sch == D-1) begin
                        emit_cnt <= '0;
                        g_out    <= g_cnt;
                        ch       <= '0;
                        state    <= S_QSTART;
                    end else begin
                        sch <= sch + 1'b1;
                    end
                end
                S_QSTART: begin
                    q_start <= 1'b1;                 // kick the shared quant core for channel `ch`
                    state   <= S_QWAIT;
                end
                S_QWAIT: if (q_done) begin
                    codes_all <= {q_code, codes_all[D*8-1:8]};
                    if (ch == D-1) begin
                        ch    <= '0;
                        state <= S_TEMIT;
                    end else begin
                        ch    <= ch + 1'b1;
                        state <= S_QSTART;
                    end
                end
                S_TEMIT: begin
                    // tok_valid pulses this cycle (combinational); tok_pay/tok_codes
                    // are packed from the now-complete codes_all.
                    if (emit_cnt == g_cnt - 'd1) state <= S_DONE;
                    else begin
                        emit_cnt <= emit_cnt + 'd1;
                        state    <= S_QSTART;         // next token, channels restart at 0
                    end
                end
                S_DONE: begin
                    group_valid <= 1'b1;
                    state       <= S_COLLECT;
                end
                default: state <= S_COLLECT;
            endcase
        end
    end

    // ---- decompress: ONE shared dequant, indexed by dec_idx (one channel/beat).
    // The top streams the D reconstructed words out one per cycle, so a single
    // dequant suffices — same as cq_value_path, and it avoids D parallel dequant
    // cones (which also stress yosys synth). Outlier channels handled by the top.
    cq_dequant_unit_syn u_d (
        .code(dec_codes[dec_idx*8 +: 8]), .scale_f16(dec_scales[dec_idx*DW +: DW]),
        .xhat_f32(dec_hat)
    );

endmodule

`default_nettype wire
