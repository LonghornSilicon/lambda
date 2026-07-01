// cq_value_path.sv — streaming per-token VALUE codec (contract §2).
//
// The value datapath as a single reusable block (the top instantiates it):
//   compress:   token -> amax_unit -> cq_scale_unit -> [1 shared quant, D cycles] -> pack
//   decompress: (codes,scale,idx) -> [1 shared dequant, combinational] -> fp32 word
// Per-token scaling over D dims — no residual buffer (that is the key path).
// `bits` is a runtime input (8 = CQ-8, 4 = CQ-4/CQ-4+); D is a parameter.
//
// AREA (P4b): the quant core carries the fp16 divider, so instead of D parallel
// quant units (D dividers) this SERIALIZES one shared quant unit over D cycles,
// and the decompress side shares ONE dequant unit indexed by dec_idx (the top
// already streams the D reconstructed words out one per cycle). ~64x fewer
// dividers vs the parallel form, at D-cycle-per-token compress latency.
//
// Cores are the synthesizable fp16 fixed-function units in cq_units_syn.sv.
// Verified vs the golden val_scales / val_payload / expected_v_hat by
// tb_value_path.sv (make sim_vpath).

`default_nettype none

module cq_value_path #(
    parameter int D  = 64,     // head dim
    parameter int DW = 16      // fp16 element width
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [3:0]        bits,       // 4 or 8

    // ---- compress: stream tokens in (present 1 token/in_valid when !busy) ----
    input  wire              in_valid,
    input  wire [D*DW-1:0]   in_vec,
    output wire              busy,       // high while a token is being compressed
    output reg               out_valid,  // 1-cycle pulse when a token is done
    output reg  [DW-1:0]     out_scale,  // fp16 per-token scale
    output reg  [D*8-1:0]    out_codes,  // D signed codes (int4 in low nibble)
    output reg  [D*8-1:0]    out_pay,    // packed: int4 -> D/2 bytes, int8 -> D bytes

    // ---- decompress: one channel per dec_idx (combinational) ----
    input  wire [D*8-1:0]         dec_codes,
    input  wire [DW-1:0]          dec_scale,
    input  wire [$clog2(D)-1:0]   dec_idx,
    output wire [31:0]            dec_hat
);
    localparam int CW = $clog2(D);

    // ---- FSM ----------------------------------------------------------------
    localparam [1:0] S_IDLE = 2'd0, S_WAIT = 2'd1, S_QUANT = 2'd2, S_EMIT = 2'd3;
    reg [1:0]   state;
    reg [CW:0]  ch;
    reg [3:0]   bits_r;
    assign busy = (state != S_IDLE);

    wire start = in_valid && (state == S_IDLE);

    // ---- amax over the token (value mode) -> fp16 scale ---------------------
    wire [DW-1:0] amax;
    wire          amax_valid;
    amax_unit #(.DIM(D), .DW(DW)) u_amax (
        .clk(clk), .rst_n(rst_n),
        .in_valid(start), .vec(in_vec),
        .mode_channel(1'b0), .group_start(1'b0), .group_done(1'b0),
        .scale_token(amax), .scale_chan(), .out_valid(amax_valid)
    );
    wire [DW-1:0] scale;
    cq_scale_unit_syn u_scale (.amax_f16(amax), .bits(bits_r), .scale_f16(scale));

    reg [D*DW-1:0] vec_reg;
    reg [DW-1:0]   scale_reg;

    // ---- one shared quantizer, walked across the D channels -----------------
    // Codes accumulate directly into the flat out_codes vector (no reg-array ->
    // no memory inference); out_pay is packed from it in S_EMIT.
    wire signed [7:0] q_code;
    cq_quant_unit_syn u_q (
        .x_f16(vec_reg[ch*DW +: DW]), .scale_f16(scale_reg), .bits(bits_r),
        .code(q_code)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; ch <= '0; bits_r <= 4'd8;
            out_valid <= 1'b0; out_scale <= '0; out_codes <= '0; out_pay <= '0;
            vec_reg <= '0; scale_reg <= '0;
        end else begin
            out_valid <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    vec_reg <= in_vec;
                    bits_r  <= bits;
                    state   <= S_WAIT;
                end
                S_WAIT: if (amax_valid) begin      // amax ready 1 cycle after start
                    scale_reg <= scale;
                    ch        <= '0;
                    out_pay   <= '0;               // int4 leaves the upper half 0
                    state     <= S_QUANT;
                end
                S_QUANT: begin
                    // full 8-bit code (for decompress) + packed nibble/byte in place:
                    // contract §5 puts code ch at nibble ch -> bit ch*4 (int4).
                    out_codes[ch*8 +: 8] <= q_code;
                    if (bits_r == 4) out_pay[ch*4 +: 4] <= q_code[3:0];
                    else             out_pay[ch*8 +: 8] <= q_code;
                    if (ch == D-1) state <= S_EMIT;
                    else           ch    <= ch + 1'b1;
                end
                S_EMIT: begin
                    out_scale <= scale_reg;
                    out_valid <= 1'b1;
                    state     <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ---- decompress: one shared dequant unit, indexed by dec_idx ------------
    cq_dequant_unit_syn u_d (
        .code(dec_codes[dec_idx*8 +: 8]), .scale_f16(dec_scale),
        .xhat_f32(dec_hat)
    );

endmodule

`default_nettype wire
