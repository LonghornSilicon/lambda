// chip_core_wrap.sv — thin elaboration wrapper for the smoke test.
//
// chip_core takes NUM_INPUT_PADS/NUM_BIDIR_PADS/NUM_ANALOG_PADS with no
// defaults (they come from the padring slot at integration time). This wrapper
// pins them to the SLOT_WORKSHOP values (1 / 20 / 60) and gives cocotb a
// concrete top to elaborate and poke. NOT part of the hardened design.

`default_nettype none

module chip_core_wrap (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [0:0]  input_in,
    input  wire [19:0] bidir_in,
    output wire [19:0] bidir_out,
    output wire [19:0] bidir_oe,
    inout  wire [59:0] analog
);
    wire [0:0]  input_pu, input_pd;
    wire [19:0] bidir_cs, bidir_sl, bidir_ie, bidir_pu, bidir_pd;

    chip_core #(
        .NUM_INPUT_PADS (1),
        .NUM_BIDIR_PADS (20),
        .NUM_ANALOG_PADS(60)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .input_in  (input_in),
        .input_pu  (input_pu),
        .input_pd  (input_pd),
        .bidir_in  (bidir_in),
        .bidir_out (bidir_out),
        .bidir_oe  (bidir_oe),
        .bidir_cs  (bidir_cs),
        .bidir_sl  (bidir_sl),
        .bidir_ie  (bidir_ie),
        .bidir_pu  (bidir_pu),
        .bidir_pd  (bidir_pd),
        .analog    (analog)
    );
endmodule

`default_nettype wire
