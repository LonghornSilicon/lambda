// tb_vecu_softmax.sv — bit-exact check of vecu_softmax vs the LUT online-softmax
// golden (gen_vecu_softmax_vectors.py). Streams each row's L fp16 scores in (LOAD),
// then collects the L fp16 weights streamed out (EMIT) and compares bit-exact.
`timescale 1ns/1ps
module tb_vecu_softmax;
    localparam integer N  = 520;   // buffer depth (>= max L in the vector file)
    localparam integer FW = 16;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg              s_valid, s_last;
    reg  [FW-1:0]    s_data;
    wire             w_valid, w_last, busy;
    wire [FW-1:0]    w_data;

    vecu_softmax #(.N(N), .FW(FW)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_data(s_data), .s_last(s_last),
        .w_valid(w_valid), .w_data(w_data), .w_last(w_last), .busy(busy));

    integer fd, code, ROWS, L, r, j, wv, cexp, got_cnt, guard;
    integer rows_pass, errors;
    reg [FW-1:0] gold [0:N-1];

    initial begin
        fd = $fopen("vecu_softmax_vectors.txt", "r");
        if (fd == 0) begin $display("FATAL: vecu_softmax_vectors.txt missing (run gen_vecu_softmax_vectors.py)"); $finish; end
        code = $fscanf(fd, "%d\n", ROWS);

        s_valid = 0; s_last = 0; s_data = 0;
        rst_n = 0; repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
        rows_pass = 0; errors = 0;

        for (r = 0; r < ROWS; r = r + 1) begin
            code = $fscanf(fd, "%d\n", L);
            // stream L scores
            for (j = 0; j < L; j = j + 1) begin
                @(negedge clk);
                code = $fscanf(fd, "%d", wv);
                s_data = wv[FW-1:0];
                s_valid = 1; s_last = (j == L-1);
            end
            @(negedge clk); s_valid = 0; s_last = 0;
            // read expected weights
            for (j = 0; j < L; j = j + 1) begin
                code = $fscanf(fd, "%d", cexp);
                gold[j] = cexp[FW-1:0];
            end
            // collect L emitted weights on w_valid
            got_cnt = 0; guard = 0;
            // micro-sequenced core: COMPUTE ~8 cyc/score + EMIT ~8 cyc/weight => ~16*L
            while (got_cnt < L && guard < (L * 20 + 64)) begin
                @(negedge clk);
                if (w_valid) begin
                    if (w_data !== gold[got_cnt]) begin
                        errors = errors + 1;
                        if (got_cnt < 4) $display("  row %0d w%0d: got %04x exp %04x", r, got_cnt, w_data, gold[got_cnt]);
                    end
                    if ((got_cnt == L-1) && (w_last !== 1'b1)) begin
                        errors = errors + 1; $display("  row %0d: w_last not set on final weight", r);
                    end
                    got_cnt = got_cnt + 1;
                end
                guard = guard + 1;
            end
            if (got_cnt !== L) begin
                errors = errors + 1;
                $display("  row %0d (L=%0d): only %0d weights emitted", r, L, got_cnt);
            end else rows_pass = rows_pass + 1;
            @(negedge clk);   // gap between rows
        end
        $fclose(fd);

        $display("Tests: %0d  Pass: %0d  Errors: %0d", ROWS, rows_pass, errors);
        if (errors == 0) $display("ALL TESTS PASSED — vecu_softmax bit-exact vs LUT online-softmax golden");
        else             $display("FAILED — %0d mismatches", errors);
        $finish;
    end
endmodule
