// chip_core.sv — Lambda ACU workshop-slot override for the Chipathon 2026
// GF180 padring fork (Mauricio-xx/chipathon-2026-gf180mcu-padring).
//
// Drop-in replacement for the fork's `src/chip_core.sv`. The pad interface
// below is COPIED VERBATIM from the fork's chip_core / the multi-macro
// template's chip_core_multi.sv (same NUM_INPUT_PADS / NUM_BIDIR_PADS /
// NUM_ANALOG_PADS parameterization and the input_*/bidir_*/analog port list)
// so it plugs straight into the workshop slot. In SLOT_WORKSHOP the fork sets
// NUM_INPUT_PADS=1, NUM_BIDIR_PADS=20, NUM_ANALOG_PADS=60.
//
// It instantiates the Lambda ACU top (`lambda_acu`), which contains the serial
// host loader and the decode-attention datapath macro sites. Because the ACU
// datapath is much wider than the pads, the host talks to it over a 4-wire SPI
// slave carried on the first four bidir pads; the rest are debug/observation.
//
// PAD MAP (workshop slot, 20 bidir pads)
//   bidir[0]  spi_sclk   INPUT   host serial clock
//   bidir[1]  spi_cs_n   INPUT   active-low frame select
//   bidir[2]  spi_mosi   INPUT   host -> chip
//   bidir[3]  spi_miso   OUTPUT  chip -> host
//   bidir[19:4]          OUTPUT  observation (ACU state + result byte)
//   input[0]             spare (tied through; reserved for external strobe)
//   analog[59:0]         pass-through, unconnected at core level
//
// STATUS: SKELETON — see lambda_acu.sv. Buildable-forward, not fake-complete.

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

    input  wire clk,       // clock
    input  wire rst_n,     // reset (active low)

    input  wire [NUM_INPUT_PADS-1:0] input_in,   // Input value
    output wire [NUM_INPUT_PADS-1:0] input_pu,   // Pull-up
    output wire [NUM_INPUT_PADS-1:0] input_pd,   // Pull-down

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,   // Input value
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,  // Output value
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,   // Output enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,   // Input type (0=CMOS, 1=Schmitt)
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,   // Slew rate (0=fast, 1=slow)
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,   // Input enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,   // Pull-up
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,   // Pull-down

    inout  wire [NUM_ANALOG_PADS-1:0] analog     // Analog
);

    // ---- input pads: disable pulls (spare input pad, reserved) -------------
    assign input_pu = '0;
    assign input_pd = '0;

    // ---- bidir pad direction: [2:0] are host->chip inputs, rest outputs ----
    // oe=1 drives outward. SPI sclk/cs_n/mosi are inputs (oe=0); miso + the
    // observation bus are outputs (oe=1).
    localparam [NUM_BIDIR_PADS-1:0] OE_MASK = ~(3'b111); // pads 0,1,2 = input
    assign bidir_oe = OE_MASK;
    assign bidir_cs = '0;              // CMOS input buffers
    assign bidir_sl = '0;              // fast slew
    assign bidir_ie = ~bidir_oe;       // input-enable = complement of oe
    assign bidir_pu = '0;
    assign bidir_pd = '0;

    // ---- Lambda ACU top ----------------------------------------------------
    wire        spi_miso;
    wire [NUM_BIDIR_PADS-1:0] obs_out;

    lambda_acu #(
        .DH         (4),    // head-dim channels (P·V lanes, Q·Kᵀ reduction)
        .L          (2),    // cached tokens / keys per decode step
        .ADDR_WIDTH (16),
        .NUM_BIDIR  (NUM_BIDIR_PADS)
    ) u_lambda_acu (
        .clk      (clk),
        .rst_n    (rst_n),
        .spi_sclk (bidir_in[0]),
        .spi_cs_n (bidir_in[1]),
        .spi_mosi (bidir_in[2]),
        .spi_miso (spi_miso),
        .obs_out  (obs_out)
    );

    // ---- drive the output pads --------------------------------------------
    //   [3]     = spi_miso
    //   [19:4]  = observation bus (top bits of obs_out)
    //   [2:0]   = input pads, driven 0 (oe=0 so value is ignored)
    assign bidir_out = { obs_out[NUM_BIDIR_PADS-1:4], spi_miso, 3'b000 };

    // analog pads float through, unconnected at the core level.

    // keep unused input pad from being optimised away
    logic _unused;
    assign _unused = &{1'b0, input_in, bidir_in[NUM_BIDIR_PADS-1:3]};

endmodule

`default_nettype wire
