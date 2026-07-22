// tb_realdata.sv — replay testbench
//
// Streams INT8-quantized score tiles produced by analysis/gen_rtl_testvectors.py
// into the precision_controller and checks the d_fp16 output against the
// integer reference recorded in expected.hex.
//
// Run from rtl/ so $readmemh resolves the relative paths:
//   iverilog -g2012 -o sim_realdata.out precision_controller.sv tb/tb_realdata.sv
//   ./sim_realdata.out
//
// NUM_TILES must match what gen_rtl_testvectors.py wrote (currently 143).
// Override at compile time with -PNUM_TILES=<n> or via Makefile.

`timescale 1ns/1ps

module tb_realdata;

    localparam integer BLOCK_M     = 64;
    localparam integer BLOCK_N     = 64;
    localparam integer SCORE_WIDTH = 8;
    localparam integer THRESHOLD   = 10;
    localparam integer N           = BLOCK_M * BLOCK_N;  // 4096

    parameter integer NUM_TILES    = 143;
    parameter         SCORES_HEX   = "tb/testvectors/scores.hex";
    parameter         EXPECTED_HEX = "tb/testvectors/expected.hex";

    reg clk   = 1'b0;
    reg rst_n = 1'b0;
    always #2.5 clk = ~clk;  // 200 MHz

    reg                          s_valid = 1'b0;
    reg signed [SCORE_WIDTH-1:0] s_data  = '0;
    reg                          s_last  = 1'b0;
    wire                         d_valid;
    wire                         d_fp16;

    precision_controller #(
        .BLOCK_M(BLOCK_M), .BLOCK_N(BLOCK_N),
        .SCORE_WIDTH(SCORE_WIDTH), .THRESHOLD(THRESHOLD)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_data(s_data), .s_last(s_last),
        .d_valid(d_valid), .d_fp16(d_fp16)
    );

    // $readmemh stores raw bit patterns (unsigned); the DUT input is signed
    // and the same width, so a same-width assignment carries the sign bit.
    reg [SCORE_WIDTH-1:0] scores   [0:NUM_TILES*N - 1];
    reg [0:0]             expected [0:NUM_TILES - 1];

    integer t, i;
    integer tests_pass = 0;
    integer tests_fail = 0;
    integer first_fail = -1;
    integer exp_bit, got_bit;

    initial begin
        $readmemh(SCORES_HEX,   scores);
        $readmemh(EXPECTED_HEX, expected);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (t = 0; t < NUM_TILES; t = t + 1) begin
            // Stream N scores; assert s_last on the last beat.
            for (i = 0; i < N; i = i + 1) begin
                @(posedge clk);
                s_valid <= 1'b1;
                s_data  <= scores[t*N + i];
                s_last  <= (i == N - 1) ? 1'b1 : 1'b0;
            end

            // DUT registers d_valid one cycle after s_last.
            @(posedge clk);
            #1;  // past NBA region

            s_valid <= 1'b0;
            s_last  <= 1'b0;
            s_data  <= '0;

            exp_bit = expected[t][0];
            got_bit = d_fp16 ? 1 : 0;

            if (!d_valid) begin
                $display("[ERROR] tile %0d: d_valid not asserted (latency bug)", t);
                tests_fail = tests_fail + 1;
                if (first_fail < 0) first_fail = t;
            end else if (got_bit === exp_bit) begin
                tests_pass = tests_pass + 1;
            end else begin
                $display("[FAIL] tile %3d  exp=%0d got=%0d", t, exp_bit, got_bit);
                tests_fail = tests_fail + 1;
                if (first_fail < 0) first_fail = t;
            end

            // Brief idle gap between tiles
            repeat (2) @(posedge clk);
        end

        repeat (4) @(posedge clk);
        $display("");
        $display("================================================");
        $display(" Replay TB:  %0d tiles   pass=%0d   fail=%0d",
                 NUM_TILES, tests_pass, tests_fail);
        if (tests_fail == 0)
            $display(" ALL TESTS PASSED");
        else
            $display(" *** FAILURES: %0d   first failing tile = %0d ***",
                     tests_fail, first_fail);
        $display("================================================");
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("TIMEOUT"); $finish;
    end

endmodule
