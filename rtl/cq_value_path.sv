// cq_value_path.sv — streaming per-token VALUE codec (contract §2).
//
// The value datapath as a single reusable block (the top will instantiate it):
//   compress:  token -> amax_unit -> cq_scale_unit -> D× cq_quant_unit -> pack
//   decompress: codes + scale -> D× cq_dequant_unit -> fp32
// Per-token scaling over D dims — no residual buffer (that is the key path).
// `bits` is a runtime input (8 = CQ-8, 4 = CQ-4/CQ-4+); D is a parameter.
//
// The quant/dequant/scale cores are the behavioral golden-equivalents
// (cq_units.sv, `real` math); amax_unit + packing are synthesizable. The fp16
// hardware lowering of the cores is P4b. Verified end-to-end vs the golden
// val_scales / val_payload / expected_v_hat by tb_value_path.sv (make sim_vpath).

// (synthesizable: instantiates only the fp16 fixed-function cores in
//  cq_units_syn.sv — no `real`, no cq_fp_pkg import here.)
`default_nettype none

module cq_value_path #(
    parameter int D  = 64,     // head dim
    parameter int DW = 16      // fp16 element width
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [3:0]        bits,       // 4 or 8

    // ---- compress: stream tokens in ----
    input  wire              in_valid,
    input  wire [D*DW-1:0]   in_vec,
    output reg               out_valid,
    output reg  [DW-1:0]     out_scale,  // fp16 per-token scale
    output reg  [D*8-1:0]    out_codes,  // D signed codes (int4 in low nibble)
    output reg  [D*8-1:0]    out_pay,    // packed: int4 -> D/2 bytes, int8 -> D bytes

    // ---- decompress: combinational (codes + scale -> fp32) ----
    input  wire [D*8-1:0]    dec_codes,
    input  wire [DW-1:0]     dec_scale,
    output wire [D*32-1:0]   dec_hat
);

    // ---- amax over the token (value mode) -> fp16 scale ----------------------
    wire [DW-1:0] amax;
    wire          amax_valid;
    amax_unit #(.DIM(D), .DW(DW)) u_amax (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .vec(in_vec),
        .mode_channel(1'b0), .group_start(1'b0), .group_done(1'b0),
        .scale_token(amax), .scale_chan(), .out_valid(amax_valid)
    );

    wire [DW-1:0] scale;
    cq_scale_unit_syn u_scale (.amax_f16(amax), .bits(bits), .scale_f16(scale));

    // Register the token so it stays aligned with its (1-cycle-late) scale — the
    // producer presents each token exactly once (no 2-cycle hold requirement).
    reg [D*DW-1:0] vec_reg;
    always @(posedge clk) if (in_valid) vec_reg <= in_vec;

    // ---- quantize the token's D elements against its scale (parallel) --------
    wire signed [7:0] code [0:D-1];
    genvar i;
    generate
        for (i = 0; i < D; i = i + 1) begin : g_quant
            cq_quant_unit_syn u_q (
                .x_f16(vec_reg[i*DW +: DW]), .scale_f16(scale), .bits(bits),
                .code(code[i])
            );
        end
    endgenerate

    // ---- pack (contract §5): int4 two-per-byte, int8 one-per-byte ------------
    reg [D*8-1:0] codes_c, pay_c;
    integer j;
    always @* begin
        codes_c = '0;
        pay_c   = '0;
        for (j = 0; j < D; j = j + 1)
            codes_c[j*8 +: 8] = code[j];
        if (bits == 4) begin
            for (j = 0; j < D/2; j = j + 1)
                pay_c[j*8 +: 8] = {code[2*j+1][3:0], code[2*j][3:0]};
        end else begin
            for (j = 0; j < D; j = j + 1)
                pay_c[j*8 +: 8] = code[j];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_scale <= '0;
            out_codes <= '0;
            out_pay   <= '0;
        end else begin
            out_valid <= amax_valid;
            if (amax_valid) begin
                out_scale <= scale;
                out_codes <= codes_c;
                out_pay   <= pay_c;
            end
        end
    end

    // ---- decompress: D dequant units, combinational --------------------------
    generate
        for (i = 0; i < D; i = i + 1) begin : g_dequant
            cq_dequant_unit_syn u_d (
                .code(dec_codes[i*8 +: 8]), .scale_f16(dec_scale),
                .xhat_f32(dec_hat[i*32 +: 32])
            );
        end
    endgenerate

endmodule

`default_nettype wire
