// scale_bank.sv — per-channel quantization-scale storage for ChannelQuant keys.
//
// Holds the DIM per-channel fp16 scales frozen for the current key group
// (contract §3.1): written in one shot from the amax->scale conversion at group
// flush, read in full (parallel) so every channel's dequant/quant lane sees its
// scale at once. Depth = DIM (= D, contract §7 SCALE_BANK_DEPTH). Instantiated
// by cq_key_path. (Per-token value scales are emitted directly by cq_value_path
// and stored by the top, so they are not banked here.)

`default_nettype none

module scale_bank #(
    parameter int DIM = 64,    // head_dim D -> per-channel bank depth
    parameter int DW  = 16     // fp16 scale width
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                wr,           // latch all DIM channel scales
    input  wire [DIM*DW-1:0]   scales_in,
    output wire [DIM*DW-1:0]   scales_out    // parallel read (all channels)
);

    reg [DW-1:0] bank [0:DIM-1];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (i = 0; i < DIM; i = i + 1) bank[i] <= '0;
        else if (wr)
            for (i = 0; i < DIM; i = i + 1) bank[i] <= scales_in[i*DW +: DW];
    end

    genvar g;
    generate
        for (g = 0; g < DIM; g = g + 1) begin : g_rd
            assign scales_out[g*DW +: DW] = bank[g];
        end
    endgenerate

endmodule

`default_nettype wire
