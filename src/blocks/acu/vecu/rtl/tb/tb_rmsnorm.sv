// tb_rmsnorm.sv — bit-exact check of rmsnorm vs the RMSNorm LUT golden
// (gen_rmsnorm_vectors.py). Streams each row's D fp16 (element, gain) pairs in
// (LOAD), then collects the D normalized elements (EMIT) and compares bit-exact.
`timescale 1ns/1ps
module tb_rmsnorm;
    localparam integer D  = 16;
    localparam integer FW = 16;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg              s_valid, s_last;
    reg  [FW-1:0]    s_data, g_data;
    wire             y_valid, y_last, busy;
    wire [FW-1:0]    y_data;

    rmsnorm #(.D(D), .FW(FW)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_data(s_data), .g_data(g_data), .s_last(s_last),
        .y_valid(y_valid), .y_data(y_data), .y_last(y_last), .busy(busy));

    integer fd, code, ROWS, LEN, r, j, xv, gv, cexp, got_cnt, guard;
    integer rows_pass, errors;
    reg [FW-1:0] xin [0:D-1];
    reg [FW-1:0] gin [0:D-1];
    reg [FW-1:0] gold [0:D-1];

    initial begin
        fd = $fopen("rmsnorm_vectors.txt", "r");
        if (fd == 0) begin $display("FATAL: rmsnorm_vectors.txt missing (run gen_rmsnorm_vectors.py)"); $finish; end
        code = $fscanf(fd, "%d\n", ROWS);

        s_valid = 0; s_last = 0; s_data = 0; g_data = 0;
        rst_n = 0; repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
        rows_pass = 0; errors = 0;

        for (r = 0; r < ROWS; r = r + 1) begin
            code = $fscanf(fd, "%d\n", LEN);
            for (j = 0; j < LEN; j = j + 1) begin
                code = $fscanf(fd, "%d", xv); xin[j] = xv[FW-1:0];
            end
            for (j = 0; j < LEN; j = j + 1) begin
                code = $fscanf(fd, "%d", gv); gin[j] = gv[FW-1:0];
            end
            for (j = 0; j < LEN; j = j + 1) begin
                code = $fscanf(fd, "%d", cexp); gold[j] = cexp[FW-1:0];
            end
            // stream (element, gain) pairs
            for (j = 0; j < LEN; j = j + 1) begin
                @(negedge clk);
                s_data = xin[j]; g_data = gin[j];
                s_valid = 1; s_last = (j == LEN-1);
            end
            @(negedge clk); s_valid = 0; s_last = 0;
            // collect LEN emitted elements on y_valid
            got_cnt = 0; guard = 0;
            while (got_cnt < LEN && guard < (LEN * 12 + 128)) begin
                @(negedge clk);
                if (y_valid) begin
                    if (y_data !== gold[got_cnt]) begin
                        errors = errors + 1;
                        $display("  row %0d y%0d: got %04x exp %04x", r, got_cnt, y_data, gold[got_cnt]);
                    end
                    if ((got_cnt == LEN-1) && (y_last !== 1'b1)) begin
                        errors = errors + 1; $display("  row %0d: y_last not set on final element", r);
                    end
                    got_cnt = got_cnt + 1;
                end
                guard = guard + 1;
            end
            if (got_cnt !== LEN) begin
                errors = errors + 1;
                $display("  row %0d: only %0d elements emitted", r, got_cnt);
            end else rows_pass = rows_pass + 1;
            @(negedge clk);
        end
        $fclose(fd);

        $display("Tests: %0d  Pass: %0d  Errors: %0d", ROWS, rows_pass, errors);
        if (errors == 0) $display("ALL TESTS PASSED — rmsnorm bit-exact vs RMSNorm LUT golden");
        else             $display("FAILED — %0d mismatches", errors);
        $finish;
    end
endmodule
