// chip_core_kv.sv — Lambda KV-cache COPROCESSOR workshop-slot core override.
//
// Drop-in replacement for the Chipathon 2026 GF180 padring fork's
// `src/chip_core.sv` (pad interface copied verbatim). Instantiates the compact
// KV-cache compression coprocessor (lambda_kv_coproc: KVE CQ-3 value compressor +
// TIU H2O importance + precision gate), which fits the 2051x2051 workshop core
// with wide margin — unlike the full fp16 attention datapath. Host talks to it
// over the same 4-wire SPI slave on the first four bidir pads.
//
// PAD MAP (workshop slot, 20 bidir pads)
//   bidir[0] spi_sclk  bidir[1] spi_cs_n  bidir[2] spi_mosi  bidir[3] spi_miso
//   bidir[19:4] observation ; input[0] spare ; analog[59:0] pass-through.

`default_nettype none

module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
    `endif
    input  wire clk,
    input  wire rst_n,

    input  wire [NUM_INPUT_PADS-1:0] input_in,
    output wire [NUM_INPUT_PADS-1:0] input_pu,
    output wire [NUM_INPUT_PADS-1:0] input_pd,

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,

    inout  wire [NUM_ANALOG_PADS-1:0] analog
);
    assign input_pu = '0;
    assign input_pd = '0;

    localparam [NUM_BIDIR_PADS-1:0] OE_MASK = ~(3'b111); // pads 0,1,2 = input
    assign bidir_oe = OE_MASK;
    assign bidir_cs = '0;
    assign bidir_sl = '0;
    assign bidir_ie = ~bidir_oe;
    assign bidir_pu = '0;
    assign bidir_pd = '0;

    wire        spi_miso;
    wire [NUM_BIDIR_PADS-1:0] obs_out;

    lambda_kv_coproc #(
        .D          (2),    // value channels per token (WHT dim)
        .L          (2),    // cache slots / tokens per compress step
        .ADDR_WIDTH (16),
        .NUM_BIDIR  (NUM_BIDIR_PADS)
    ) u_coproc (
        .clk      (clk),
        .rst_n    (rst_n),
        .spi_sclk (bidir_in[0]),
        .spi_cs_n (bidir_in[1]),
        .spi_mosi (bidir_in[2]),
        .spi_miso (spi_miso),
        .obs_out  (obs_out)
    );

    assign bidir_out = { obs_out[NUM_BIDIR_PADS-1:4], spi_miso, 3'b000 };

    logic _unused;
    assign _unused = &{1'b0, input_in, bidir_in[NUM_BIDIR_PADS-1:3]};

endmodule

`default_nettype wire
