// tb_precision_controller.sv — iverilog-compatible self-checking testbench
//
// Compile + run:
//   iverilog -g2012 -o sim.out precision_controller.sv tb/tb_precision_controller.sv
//   ./sim.out
//
// Vivado xsim:
//   xvlog -sv precision_controller.sv tb/tb_precision_controller.sv
//   xelab tb_precision_controller -s sim && xsim sim -runall
//
// Cadence Xcelium:
//   xrun -sv precision_controller.sv tb/tb_precision_controller.sv

`timescale 1ns/1ps

module tb_precision_controller;

    localparam integer BLOCK_M     = 64;
    localparam integer BLOCK_N     = 64;
    localparam integer SCORE_WIDTH = 8;
    localparam integer THRESHOLD   = 10;
    localparam integer N           = BLOCK_M * BLOCK_N;  // 4096

    reg clk   = 0;
    reg rst_n = 0;
    always #2.5 clk = ~clk;  // 200 MHz

    reg                          s_valid = 0;
    reg signed [SCORE_WIDTH-1:0] s_data  = 0;
    reg                          s_last  = 0;
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

    integer tests_run  = 0;
    integer tests_pass = 0;
    integer tests_fail = 0;

    reg signed [SCORE_WIDTH-1:0] tile [0:N-1];
    integer ref_max, ref_sum, ref_abs, ref_exp;
    integer i, j, seed;
    integer got;

    // Software reference
    task compute_ref;
        begin
            ref_max = 0; ref_sum = 0;
            for (i = 0; i < N; i = i + 1) begin
                ref_abs = (tile[i] < 0) ? -tile[i] : tile[i];
                if (ref_abs > ref_max) ref_max = ref_abs;
                ref_sum = ref_sum + ref_abs;
            end
            // Same formula as DUT: max*N > THRESHOLD*sum → FP16
            ref_exp = (ref_max * N > THRESHOLD * ref_sum) ? 1 : 0;
        end
    endtask

    // Send tile and check.
    // DUT latency is exactly 1 cycle after s_last.
    // Use #1 after posedge to read post-NBA values and avoid race with d_valid clear.
    task send_and_check;
        input integer test_id;
        begin
            // Stream N scores; s_last fires on the last one
            for (i = 0; i < N; i = i + 1) begin
                @(posedge clk);
                s_valid = 1'b1;
                s_data  = tile[i];
                s_last  = (i == N - 1) ? 1'b1 : 1'b0;
            end

            // One more posedge: DUT samples s_last=1 and registers d_valid=1
            @(posedge clk);
            #1; // advance past NBA region so d_valid is readable

            // Deassert (DUT resets accumulators, will clear d_valid next cycle)
            s_valid = 1'b0;
            s_last  = 1'b0;
            s_data  = 0;

            // Read result — d_valid must be 1 here (exactly 1-cycle latency)
            if (!d_valid) begin
                $display("[ERROR] test %0d: d_valid not asserted (latency bug)", test_id);
                tests_fail = tests_fail + 1;
                tests_run  = tests_run  + 1;
                repeat(2) @(posedge clk);
            end else begin
                got = d_fp16 ? 1 : 0;
                tests_run = tests_run + 1;
                if (got === ref_exp) begin
                    tests_pass = tests_pass + 1;
                    $display("[PASS] test %3d  exp=%0d got=%0d  max=%0d sum=%0d",
                             test_id, ref_exp, got, ref_max, ref_sum);
                end else begin
                    tests_fail = tests_fail + 1;
                    $display("[FAIL] test %3d  exp=%0d got=%0d  max=%0d sum=%0d",
                             test_id, ref_exp, got, ref_max, ref_sum);
                end
                repeat(2) @(posedge clk);
            end
        end
    endtask

    task fill_const;
        input signed [SCORE_WIDTH-1:0] val;
        begin
            for (i = 0; i < N; i = i + 1) tile[i] = val;
        end
    endtask

    initial begin
        seed = 42;
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // 1: Uniform +2 → INT8
        fill_const(8'sh02); compute_ref; send_and_check(1);

        // 2: Uniform -3 → INT8
        fill_const(-8'sh03); compute_ref; send_and_check(2);

        // 3: Single positive spike → FP16
        fill_const(8'sh01);
        tile[N/2] = 8'sh60;  // 0x60 = 96
        compute_ref; send_and_check(3);

        // 4: Single negative spike → FP16
        fill_const(8'sh01);
        tile[0] = -8'sh60;
        compute_ref; send_and_check(4);

        // 5: All-zero → INT8 (safe default)
        fill_const(8'sh00); compute_ref; send_and_check(5);

        // 6: All +127 (uniform max) → INT8
        fill_const(8'sh7F); compute_ref; send_and_check(6);

        // 7: Two max spikes in background of 1s
        fill_const(8'sh01);
        tile[10] = 8'sh7F; tile[20] = 8'sh7F;
        compute_ref; send_and_check(7);

        // 8: Spike at index 0 (first element)
        fill_const(8'sh02);
        tile[0] = 8'sh7F;
        compute_ref; send_and_check(8);

        // 9: Spike at last index
        fill_const(8'sh02);
        tile[N-1] = 8'sh7F;
        compute_ref; send_and_check(9);

        // 10: Alternating ±3 → INT8
        for (i = 0; i < N; i = i + 1)
            tile[i] = (i % 2 == 0) ? 8'sh03 : -8'sh03;
        compute_ref; send_and_check(10);

        // 11-110: 100 random tiles (LCG)
        for (j = 0; j < 100; j = j + 1) begin
            for (i = 0; i < N; i = i + 1) begin
                seed = seed * 1664525 + 1013904223;
                tile[i] = $signed((seed >>> 16) % 7) - 3;
            end
            seed = seed * 1664525 + 1013904223;
            if (((seed >>> 16) & 8'hFF) < 26) begin  // ~10% spike
                seed = seed * 1664525 + 1013904223;
                i = ((seed >>> 4) & 32'hFFF) % N;
                seed = seed * 1664525 + 1013904223;
                tile[i] = $signed(40 + ((seed >>> 16) % 87));
            end
            compute_ref; send_and_check(11 + j);
        end

        repeat(4) @(posedge clk);
        $display("");
        $display("================================================");
        $display(" Tests: %0d  Pass: %0d  Fail: %0d",
                 tests_run, tests_pass, tests_fail);
        if (tests_fail == 0)
            $display(" ALL TESTS PASSED");
        else
            $display(" *** FAILURES: %0d ***", tests_fail);
        $display("================================================");
        $finish;
    end

    initial begin
        #500_000_000;
        $display("TIMEOUT"); $finish;
    end

endmodule
