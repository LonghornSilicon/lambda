// tb_rope.sv — bit-exact check of rope vs the RoPE LUT golden
// (gen_rope_vectors.py). Streams each row's HEAD_DIM fp16 channels in (LOAD) with
// its position, then collects the HEAD_DIM rotated channels (EMIT) and compares
// bit-exact.
`timescale 1ns/1ps
module tb_rope;
    localparam integer HEAD_DIM = 8;
    localparam integer MAX_POS  = 16;
    localparam integer FW       = 16;
    localparam integer POSW     = $clog2(MAX_POS);

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg              s_valid, s_last;
    reg  [FW-1:0]    s_data;
    reg  [POSW-1:0]  pos;
    wire             y_valid, y_last, busy;
    wire [FW-1:0]    y_data;

    rope #(.HEAD_DIM(HEAD_DIM), .MAX_POS(MAX_POS), .FW(FW)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_data(s_data), .s_last(s_last), .pos(pos),
        .y_valid(y_valid), .y_data(y_data), .y_last(y_last), .busy(busy));

    integer fd, code, ROWS, r, j, wv, cexp, got_cnt, guard;
    integer rows_pass, errors, p;
    reg [FW-1:0] gold [0:HEAD_DIM-1];

    initial begin
        fd = $fopen("rope_vectors.txt", "r");
        if (fd == 0) begin $display("FATAL: rope_vectors.txt missing (run gen_rope_vectors.py)"); $finish; end
        code = $fscanf(fd, "%d\n", ROWS);

        s_valid = 0; s_last = 0; s_data = 0; pos = 0;
        rst_n = 0; repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
        rows_pass = 0; errors = 0;

        for (r = 0; r < ROWS; r = r + 1) begin
            code = $fscanf(fd, "%d\n", p);
            // stream HEAD_DIM channels with the row's position
            for (j = 0; j < HEAD_DIM; j = j + 1) begin
                @(negedge clk);
                code = $fscanf(fd, "%d", wv);
                s_data = wv[FW-1:0];
                pos = p[POSW-1:0];
                s_valid = 1; s_last = (j == HEAD_DIM-1);
            end
            @(negedge clk); s_valid = 0; s_last = 0;
            // read expected rotated channels
            for (j = 0; j < HEAD_DIM; j = j + 1) begin
                code = $fscanf(fd, "%d", cexp);
                gold[j] = cexp[FW-1:0];
            end
            // collect HEAD_DIM emitted channels on y_valid
            got_cnt = 0; guard = 0;
            while (got_cnt < HEAD_DIM && guard < (HEAD_DIM * 12 + 64)) begin
                @(negedge clk);
                if (y_valid) begin
                    if (y_data !== gold[got_cnt]) begin
                        errors = errors + 1;
                        $display("  row %0d y%0d: got %04x exp %04x", r, got_cnt, y_data, gold[got_cnt]);
                    end
                    if ((got_cnt == HEAD_DIM-1) && (y_last !== 1'b1)) begin
                        errors = errors + 1; $display("  row %0d: y_last not set on final channel", r);
                    end
                    got_cnt = got_cnt + 1;
                end
                guard = guard + 1;
            end
            if (got_cnt !== HEAD_DIM) begin
                errors = errors + 1;
                $display("  row %0d: only %0d channels emitted", r, got_cnt);
            end else rows_pass = rows_pass + 1;
            @(negedge clk);
        end
        $fclose(fd);

        $display("Tests: %0d  Pass: %0d  Errors: %0d", ROWS, rows_pass, errors);
        if (errors == 0) $display("ALL TESTS PASSED — rope bit-exact vs RoPE LUT golden");
        else             $display("FAILED — %0d mismatches", errors);
        $finish;
    end
endmodule
