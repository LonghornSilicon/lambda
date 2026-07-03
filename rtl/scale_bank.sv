// scale_bank.sv — per-channel quantization-scale storage for ChannelQuant keys.
//
// Holds the DIM per-channel fp16 scales for the current key group (contract §3.1).
// Written one channel per cycle during the scale walk (cq_key_path serializes the
// amax->scale conversion onto ONE shared cq_scale_unit rather than DIM parallel
// dividers), read in full (parallel) so every channel's dequant/quant lane sees
// its scale at once. Depth = DIM (= D, contract §7 SCALE_BANK_DEPTH). Instantiated
// by cq_key_path. (Per-token value scales are emitted directly by cq_value_path
// and stored by the top, so they are not banked here.)

`default_nettype none

module scale_bank #(
    parameter int DIM = 64,    // head_dim D -> per-channel bank depth
    parameter int DW  = 16     // fp16 scale width
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     wr_en,     // write one channel this cycle
    input  wire [$clog2(DIM)-1:0]   wr_idx,    // which channel
    input  wire [DW-1:0]            wr_scale,  // its fp16 scale
    output wire [DIM*DW-1:0]        scales_out // parallel read (all channels)
);

    reg [DW-1:0] bank [0:DIM-1];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (i = 0; i < DIM; i = i + 1) bank[i] <= '0;
        else if (wr_en)
            bank[wr_idx] <= wr_scale;
    end

    genvar g;
    generate
        for (g = 0; g < DIM; g = g + 1) begin : g_rd
            assign scales_out[g*DW +: DW] = bank[g];
        end
    endgenerate

endmodule

`default_nettype wire
